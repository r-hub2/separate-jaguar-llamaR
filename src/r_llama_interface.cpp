#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <climits>
#include <map>
#include <memory>

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Rinternals.h defines length() as a macro which conflicts with C++ methods
#ifdef length
#undef length
#endif

#include "r_llama_ptr.h"    // type-checked externalptr arguments
#include "r_llama_throw.h"  // llamar_error(): throws instead of longjmp-ing

#include "llama.h"
#include "llama-ext.h"  // [llamaR] llama_get_memory_breakdown (master moved it here)
// [llamaR] llama_context::get_cparams(), to report the flash-attention type a
// context actually resolved to. There is no public getter for it in llama.h.
#include "llama-context.h"
#include <ggml-backend.h>
#include <ggml-cpu.h>

// ============================================================
// Logging control
// ============================================================

static int log_verbosity = 1;  // 0 = silent, 1 = errors only, 2 = normal, 3 = verbose

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (log_verbosity == 0) return;
    if (log_verbosity == 1 && level != GGML_LOG_LEVEL_ERROR) return;
    if (log_verbosity == 2 && level == GGML_LOG_LEVEL_DEBUG) return;
    // verbosity 3 = show everything
    Rprintf("%s", text);
}

// ============================================================
// Backend initialization (lazy, once)
// ============================================================

static bool backend_initialized = false;

static void ensure_backend_init(void) {
    if (!backend_initialized) {
        llama_log_set(llama_log_callback, NULL);
        llama_backend_init();
        backend_initialized = true;
    }
}

// ============================================================
// Finalizers — auto-free on GC
// ============================================================

// ------------------------------------------------------------
// External-pointer arguments (helpers in r_llama_ptr.h)
// ------------------------------------------------------------

// The wording of `what` is what user-facing error messages have always said.
static inline llama_model * llamar_model_arg(SEXP x) {
    return (llama_model *) llamar_ptr_arg(x, "model");
}
static inline llama_context * llamar_ctx_arg(SEXP x) {
    return (llama_context *) llamar_ptr_arg(x, "context");
}
static inline llama_adapter_lora * llamar_lora_arg(SEXP x) {
    return (llama_adapter_lora *) llamar_ptr_arg(x, "LoRA adapter");
}

// Throwing counterparts, for the entry points that hold a std::vector /
// std::string / RAII guard when the argument is validated: a longjmp there
// would skip the destructor. Those entry points wrap their body in
// LLAMAR_ENTRYPOINT_BEGIN/END; everything else uses the plain versions above,
// whose longjmp lets R reset the protection stack (and keeps rchk quiet).
static inline llama_model * llamar_model_arg_throw(SEXP x) {
    return (llama_model *) llamar_ptr_arg_throw(x, "model");
}
static inline llama_context * llamar_ctx_arg_throw(SEXP x) {
    return (llama_context *) llamar_ptr_arg_throw(x, "context");
}
static inline llama_adapter_lora * llamar_lora_arg_throw(SEXP x) {
    return (llama_adapter_lora *) llamar_ptr_arg_throw(x, "LoRA adapter");
}

static void model_finalizer(SEXP x) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(x);
    if (model) {
        llama_model_free(model);
        R_SetExternalPtrAddr(x, NULL);
    }
}

static void context_finalizer(SEXP x) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(x);
    if (ctx) {
        llama_free(ctx);
        R_SetExternalPtrAddr(x, NULL);
    }
}

// Rewrite a byte range so that it is valid UTF-8, replacing anything that is
// not with U+FFFD. Needed because a model can emit byte-fragment tokens
// (<0x85>, <0xBE>, ...) that never combine into a character: an orphaned
// continuation byte, or a lead byte followed by another lead instead of its
// continuation bytes. Holding those back would stall the stream forever, and
// passing them to R tagged CE_UTF8 yields a string that breaks every consumer
// that has to re-encode it — a JSON response body above all.
static std::string utf8_sanitize(const char * data, size_t n) {
    static const char * REPLACEMENT = "\xEF\xBF\xBD";  // U+FFFD
    std::string out;
    out.reserve(n);
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char) data[i];
        size_t need;
        if      ((c & 0x80) == 0x00) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else {                       // continuation byte with no lead, or 0xF8+
            out += REPLACEMENT;
            i++;
            continue;
        }
        if (i + need > n) {          // truncated at the end of the range
            out += REPLACEMENT;
            i++;
            continue;
        }
        bool ok = true;
        for (size_t k = 1; k < need; k++) {
            if (((unsigned char) data[i + k] & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) {                   // lead byte not followed by its body
            out += REPLACEMENT;
            i++;
            continue;
        }
        out.append(data + i, need);
        i += need;
    }
    return out;
}

// Streaming generation state (token-by-token), see r_llama_gen_* below.
struct llama_gen_state {
    llama_context *      ctx   = NULL;   // borrowed, not owned
    const llama_vocab *  vocab = NULL;   // borrowed
    llama_sampler *      smpl  = NULL;   // owned, freed by finalizer
    int                  n_remaining = 0;
    bool                 done  = false;
    std::string          utf8_buf;       // bytes of an incomplete trailing UTF-8 char
};

static inline llama_gen_state * llamar_gen_state_arg(SEXP x) {
    return (llama_gen_state *) llamar_ptr_arg(x, "generation state");
}

// Throwing counterpart; see llamar_model_arg_throw above.
static inline llama_gen_state * llamar_gen_state_arg_throw(SEXP x) {
    return (llama_gen_state *) llamar_ptr_arg_throw(x, "generation state");
}

static void gen_state_finalizer(SEXP x) {
    llama_gen_state * st = (llama_gen_state *) R_ExternalPtrAddr(x);
    if (st) {
        if (st->smpl) llama_sampler_free(st->smpl);
        delete st;
        R_SetExternalPtrAddr(x, NULL);
    }
}

// ============================================================
// Version
// ============================================================

extern "C" SEXP r_llama_version(void) {
    return Rf_mkString("0.1.1");
}

extern "C" SEXP r_llama_supports_gpu(void) {
    ensure_backend_init();
    return Rf_ScalarLogical(llama_supports_gpu_offload() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_set_verbosity(SEXP r_level) {
    int level = INTEGER(r_level)[0];
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    log_verbosity = level;
    return R_NilValue;
}

extern "C" SEXP r_llama_get_verbosity(void) {
    return Rf_ScalarInteger(log_verbosity);
}

// ============================================================
// Time / NUMA / Backend devices
// ============================================================

extern "C" SEXP r_llama_time_us(void) {
    return Rf_ScalarReal((double) llama_time_us());
}

extern "C" SEXP r_llama_numa_init(SEXP r_strategy) {
    ensure_backend_init();
    int strategy = INTEGER(r_strategy)[0];
    if (strategy < 0 || strategy >= GGML_NUMA_STRATEGY_COUNT)
        Rf_error("llamaR: invalid NUMA strategy %d (valid: 0..%d)", strategy,
                 GGML_NUMA_STRATEGY_COUNT - 1);
    llama_numa_init((enum ggml_numa_strategy) strategy);
    return R_NilValue;
}

extern "C" SEXP r_llama_backend_devices(void) {
    ensure_backend_init();
    size_t n = ggml_backend_dev_count();

    SEXP names_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
    SEXP descs_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
    SEXP types_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));

    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        SET_STRING_ELT(names_vec, (R_xlen_t) i, Rf_mkChar(ggml_backend_dev_name(dev)));
        SET_STRING_ELT(descs_vec, (R_xlen_t) i, Rf_mkChar(ggml_backend_dev_description(dev)));

        enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        const char * type_str = "unknown";
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU)   type_str = "cpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_GPU)   type_str = "gpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_IGPU)  type_str = "igpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) type_str = "accel";
        SET_STRING_ELT(types_vec, (R_xlen_t) i, Rf_mkChar(type_str));
    }

    // Build data.frame
    SEXP df = PROTECT(Rf_allocVector(VECSXP, 3));
    SET_VECTOR_ELT(df, 0, names_vec);
    SET_VECTOR_ELT(df, 1, descs_vec);
    SET_VECTOR_ELT(df, 2, types_vec);

    SEXP col_names = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(col_names, 0, Rf_mkChar("name"));
    SET_STRING_ELT(col_names, 1, Rf_mkChar("description"));
    SET_STRING_ELT(col_names, 2, Rf_mkChar("type"));
    Rf_setAttrib(df, R_NamesSymbol, col_names);

    SEXP row_names = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(row_names)[0] = NA_INTEGER;
    INTEGER(row_names)[1] = -(int) n;
    Rf_setAttrib(df, R_RowNamesSymbol, row_names);
    Rf_setAttrib(df, R_ClassSymbol, Rf_mkString("data.frame"));

    UNPROTECT(6);
    return df;
}

// ============================================================
// Model: load / free / info
// ============================================================

// Fill in llama_model_params from the R-side arguments shared by the
// single-file and split loaders. `devs` receives the resolved device list and
// must outlive the load call, since mparams.devices points into it.
static void llamar_model_params_from_sexp(struct llama_model_params & mparams,
                                          std::vector<ggml_backend_dev_t> & devs,
                                          SEXP r_n_gpu_layers, SEXP r_devices,
                                          SEXP r_split_mode, SEXP r_use_mmap,
                                          SEXP r_use_mlock) {
    mparams.n_gpu_layers = INTEGER(r_n_gpu_layers)[0];
    mparams.split_mode   = (enum llama_split_mode) INTEGER(r_split_mode)[0];
    mparams.use_mmap     = LOGICAL(r_use_mmap)[0];
    mparams.use_mlock    = LOGICAL(r_use_mlock)[0];

    // device selection
    if (!Rf_isNull(r_devices)) {
        int n_devs = Rf_length(r_devices);
        size_t n_available = ggml_backend_dev_count();
        for (int i = 0; i < n_devs; i++) {
            const char * dev_name = CHAR(STRING_ELT(r_devices, i));
            // try by name first
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
            if (!dev) {
                // try by type keyword: "cpu", "gpu"
                if (strcmp(dev_name, "cpu") == 0) {
                    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                } else if (strcmp(dev_name, "gpu") == 0) {
                    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                    if (!dev)
                        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
                } else {
                    // try as numeric index (0-based)
                    char * endptr;
                    long idx = strtol(dev_name, &endptr, 10);
                    if (*endptr == '\0' && idx >= 0 && (size_t) idx < n_available)
                        dev = ggml_backend_dev_get((size_t) idx);
                }
            }
            if (!dev) llamar_error("llamaR: device not found: '%s'", dev_name);
            devs.push_back(dev);
        }
        devs.push_back(nullptr);  // NULL-terminated
        mparams.devices = devs.data();
    }
}

extern "C" SEXP r_llama_load_model(SEXP r_path, SEXP r_n_gpu_layers, SEXP r_devices,
                                   SEXP r_split_mode, SEXP r_use_mmap, SEXP r_use_mlock) {
    LLAMAR_ENTRYPOINT_BEGIN
    ensure_backend_init();

    const char * path = CHAR(STRING_ELT(r_path, 0));

    struct llama_model_params mparams = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    llamar_model_params_from_sexp(mparams, devs, r_n_gpu_layers, r_devices,
                                  r_split_mode, r_use_mmap, r_use_mlock);

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        llamar_error("llamaR: failed to load model from '%s'", path);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, model_finalizer);
    UNPROTECT(1);
    return result;
    LLAMAR_ENTRYPOINT_END
}

// Load a model whose GGUF is split across several files with a naming scheme
// llama_model_load_from_file cannot infer. The paths must already be in order.
extern "C" SEXP r_llama_load_model_from_splits(SEXP r_paths, SEXP r_n_gpu_layers,
                                               SEXP r_devices, SEXP r_split_mode,
                                               SEXP r_use_mmap, SEXP r_use_mlock) {
    LLAMAR_ENTRYPOINT_BEGIN
    ensure_backend_init();

    if (TYPEOF(r_paths) != STRSXP) {
        llamar_error("llamaR: paths must be a character vector");
    }
    const int n_paths = (int) Rf_length(r_paths);
    if (n_paths == 0) llamar_error("llamaR: paths must be non-empty");

    std::vector<const char *> paths;
    paths.reserve(n_paths);
    for (int i = 0; i < n_paths; i++) {
        paths.push_back(CHAR(STRING_ELT(r_paths, i)));
    }

    struct llama_model_params mparams = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    llamar_model_params_from_sexp(mparams, devs, r_n_gpu_layers, r_devices,
                                  r_split_mode, r_use_mmap, r_use_mlock);

    llama_model * model = llama_model_load_from_splits(paths.data(),
                                                       (size_t) n_paths, mparams);
    if (!model) {
        llamar_error("llamaR: failed to load model from %d splits (first: '%s')",
                     n_paths, paths[0]);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, model_finalizer);
    UNPROTECT(1);
    return result;
    LLAMAR_ENTRYPOINT_END
}

// llama_split_path/_prefix take a zero-based split_no and print split_no + 1,
// so the number in the file name is one-based. R counts from one throughout,
// and this number is visible in the file name rather than being an internal
// index, so the R side is one-based and converts here.
// (NB: the comments in llama.h show a zero-based call with a one-based result;
// the implementation in llama.cpp is what these follow.)
static int llamar_split_no_from_r(SEXP r_split_no, SEXP r_split_count) {
    const int split_no    = INTEGER(r_split_no)[0];
    const int split_count = INTEGER(r_split_count)[0];
    if (split_no == NA_INTEGER || split_count == NA_INTEGER) {
        llamar_error("llamaR: split_no and split_count must not be NA");
    }
    if (split_count < 1) {
        llamar_error("llamaR: split_count must be at least 1, got %d", split_count);
    }
    if (split_no < 1 || split_no > split_count) {
        llamar_error("llamaR: split_no must be between 1 and split_count (%d), got %d",
                     split_count, split_no);
    }
    return split_no - 1;
}

// Build the path of one chunk of a split GGUF from its prefix.
extern "C" SEXP r_llama_split_path(SEXP r_prefix, SEXP r_split_no, SEXP r_split_count) {
    LLAMAR_ENTRYPOINT_BEGIN
    const char * prefix = CHAR(STRING_ELT(r_prefix, 0));
    const int split_no    = llamar_split_no_from_r(r_split_no, r_split_count);
    const int split_count = INTEGER(r_split_count)[0];

    // PATH_MAX is not portable enough to rely on here; llama.cpp itself uses a
    // fixed buffer for this, and the pattern only appends "-%05d-of-%05d.gguf".
    std::vector<char> buf(4096, '\0');
    const int32_t len = llama_split_path(buf.data(), buf.size(), prefix,
                                         split_no, split_count);
    if (len <= 0) llamar_error("llamaR: failed to build split path (prefix too long?)");
    return Rf_mkString(buf.data());
    LLAMAR_ENTRYPOINT_END
}

// Recover the prefix from a split path, when split_no/split_count match it.
extern "C" SEXP r_llama_split_prefix(SEXP r_path, SEXP r_split_no, SEXP r_split_count) {
    LLAMAR_ENTRYPOINT_BEGIN
    const char * path = CHAR(STRING_ELT(r_path, 0));
    const int split_no    = llamar_split_no_from_r(r_split_no, r_split_count);
    const int split_count = INTEGER(r_split_count)[0];

    std::vector<char> buf(4096, '\0');
    const int32_t len = llama_split_prefix(buf.data(), buf.size(), path,
                                           split_no, split_count);
    // A mismatch is a legitimate answer ("this path is not that chunk"), not an
    // error, so report it as NA rather than failing.
    if (len <= 0) return Rf_ScalarString(NA_STRING);
    return Rf_mkString(buf.data());
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_free_model(SEXP r_model) {
    // Freeing an already-freed handle is a no-op, so NULL is accepted here;
    // the type still has to be right before anything is written back to it.
    llama_model * model = (llama_model *) llamar_ptr_addr_or_null(r_model, "model");
    if (model) {
        llama_model_free(model);
        R_SetExternalPtrAddr(r_model, NULL);
    }
    return R_NilValue;
}

extern "C" SEXP r_llama_model_info(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 7));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(llama_model_n_ctx_train(model)));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(llama_model_n_embd(model)));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(llama_vocab_n_tokens(vocab)));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(llama_model_n_layer(model)));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(llama_model_n_head(model)));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(llama_model_n_head_kv(model)));
    SET_VECTOR_ELT(result, 6, Rf_mkString(desc));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 7));
    SET_STRING_ELT(names, 0, Rf_mkChar("n_ctx_train"));
    SET_STRING_ELT(names, 1, Rf_mkChar("n_embd"));
    SET_STRING_ELT(names, 2, Rf_mkChar("n_vocab"));
    SET_STRING_ELT(names, 3, Rf_mkChar("n_layer"));
    SET_STRING_ELT(names, 4, Rf_mkChar("n_head"));
    SET_STRING_ELT(names, 5, Rf_mkChar("n_head_kv"));
    SET_STRING_ELT(names, 6, Rf_mkChar("desc"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

// ============================================================
// Context: new / free
// ============================================================

extern "C" SEXP r_llama_new_context(SEXP r_model, SEXP r_n_ctx,
                                    SEXP r_n_threads, SEXP r_n_threads_batch,
                                    SEXP r_n_batch, SEXP r_n_ubatch,
                                    SEXP r_n_seq_max,
                                    SEXP r_flash_attn, SEXP r_embedding) {
    llama_model * model = llamar_model_arg(r_model);

    bool embedding = LOGICAL(r_embedding)[0];

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = (uint32_t) INTEGER(r_n_ctx)[0];
    cparams.n_threads        = INTEGER(r_n_threads)[0];
    cparams.n_threads_batch  = INTEGER(r_n_threads_batch)[0];
    cparams.n_batch          = (uint32_t) INTEGER(r_n_batch)[0];
    cparams.n_ubatch         = (uint32_t) INTEGER(r_n_ubatch)[0];
    cparams.n_seq_max        = (uint32_t) INTEGER(r_n_seq_max)[0];
    cparams.flash_attn_type  = (enum llama_flash_attn_type) INTEGER(r_flash_attn)[0];
    cparams.embeddings       = embedding;
    cparams.kv_unified       = false; // match llama-bench / upstream default
    cparams.swa_full         = false; // match llama-bench
    cparams.no_perf          = false; // enable llama_perf_context() counters

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        Rf_error("llamaR: failed to create context");
    }

    if (embedding) {
        llama_set_causal_attn(ctx, false);
    }

    // tag = embedding flag, prot = r_model (keeps model alive)
    SEXP tag = PROTECT(Rf_ScalarLogical(embedding));
    SEXP result = PROTECT(R_MakeExternalPtr(ctx, tag, r_model));
    R_RegisterCFinalizer(result, context_finalizer);
    UNPROTECT(2);
    return result;
}

// Helper: check if context was created in embedding mode
static bool ctx_is_embedding(SEXP r_ctx) {
    SEXP tag = R_ExternalPtrTag(r_ctx);
    if (tag != R_NilValue && TYPEOF(tag) == LGLSXP) {
        return LOGICAL(tag)[0];
    }
    return false;
}

extern "C" SEXP r_llama_free_context(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) llamar_ptr_addr_or_null(r_ctx, "context");
    if (ctx) {
        llama_free(ctx);
        R_SetExternalPtrAddr(r_ctx, NULL);
    }
    return R_NilValue;
}

// ============================================================
// Tokenize / Detokenize
// ============================================================

extern "C" SEXP r_llama_tokenize(SEXP r_ctx, SEXP r_text, SEXP r_add_special,
                                 SEXP r_parse_special) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text          = CHAR(STRING_ELT(r_text, 0));
    bool         add_special   = LOGICAL(r_add_special)[0] != 0;
    bool         parse_special = LOGICAL(r_parse_special)[0] != 0;
    int          text_len      = (int) strlen(text);

    // first pass: get required buffer size (returns negative on "need more space")
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, add_special, parse_special);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int actual = llama_tokenize(vocab, text, text_len, tokens.data(), n_tokens, add_special, parse_special);
    if (actual < 0) {
        llamar_error("llamaR: tokenization failed");
    }

    SEXP r_result = PROTECT(Rf_allocVector(INTSXP, actual));
    for (int i = 0; i < actual; i++) {
        INTEGER(r_result)[i] = tokens[i];
    }
    UNPROTECT(1);
    return r_result;
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_detokenize(SEXP r_ctx, SEXP r_tokens) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_tokens = LENGTH(r_tokens);
    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = INTEGER(r_tokens)[i];
    }

    // first pass: get required buffer size
    int text_len = llama_detokenize(vocab, tokens.data(), n_tokens, NULL, 0, true, false);
    if (text_len < 0) text_len = -text_len;

    std::vector<char> text(text_len + 1);
    int actual = llama_detokenize(vocab, tokens.data(), n_tokens, text.data(), text_len, true, false);
    if (actual < 0) actual = 0;
    text[actual] = '\0';

    return Rf_mkString(text.data());
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Generate: prompt → encode → decode loop → text
// ============================================================

// Add a grammar sampler to the chain. If trigger patterns/tokens are supplied
// (from llama_chat_build for lazy formats like Mistral/Ministral), use the
// lazy-patterns sampler so the grammar only constrains output after a trigger
// (e.g. [TOOL_CALLS]); otherwise use the plain grammar sampler. r_trig_pat /
// r_trig_tok may be NULL or empty for non-lazy grammars.
static void llamar_add_grammar_sampler(llama_sampler * smpl,
                                       const llama_vocab * vocab,
                                       const char * grammar,
                                       SEXP r_trig_pat, SEXP r_trig_tok) {
    if (!grammar || strlen(grammar) == 0) return;

    const int n_pat = Rf_isNull(r_trig_pat) ? 0 : (int) Rf_length(r_trig_pat);
    const int n_tok = Rf_isNull(r_trig_tok) ? 0 : (int) Rf_length(r_trig_tok);

    // Lazy-patterns sampler only when triggers are supplied (the grammar then
    // waits for a trigger before constraining). The caller passes triggers only
    // for lazy grammars; a non-lazy grammar must constrain from the first token
    // and so falls through to the plain grammar sampler.
    if (n_pat == 0 && n_tok == 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar, "root"));
        return;
    }

    std::vector<const char *> patterns;
    patterns.reserve(n_pat);
    for (int i = 0; i < n_pat; i++) {
        patterns.push_back(CHAR(STRING_ELT(r_trig_pat, i)));
    }
    std::vector<llama_token> tokens;
    tokens.reserve(n_tok);
    for (int i = 0; i < n_tok; i++) {
        tokens.push_back((llama_token) INTEGER(r_trig_tok)[i]);
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_grammar_lazy_patterns(
        vocab, grammar, "root",
        patterns.data(), patterns.size(),
        tokens.data(), tokens.size()));
}

// ============================================================
// Sampler chain: one builder shared by generate / gen_begin /
// gen_begin_at / generate_batch
// ============================================================

// Plain-old-data view of the sampling parameters. Deliberately free of SEXP so
// the chain can later be built off the R thread (see the batching work in
// TODO.md); llamar_sampler_params_from_sexp does all the R-side reading.
struct llamar_sampler_params {
    float    temp           = 0.8f;
    int      top_k          = 50;
    float    top_p          = 0.9f;
    float    min_p          = 0.0f;
    float    typical_p      = 1.0f;
    uint32_t seed           = 42;
    int      min_keep       = 1;

    float    repeat_penalty = 1.0f;
    int      repeat_last_n  = 64;
    float    freq_penalty   = 0.0f;
    float    pres_penalty   = 0.0f;

    int      mirostat       = 0;
    float    mirostat_tau   = 5.0f;
    float    mirostat_eta   = 0.1f;

    // dynamic temperature (temp_ext); range 0 makes it equivalent to plain temp
    float    dynatemp_range    = 0.0f;
    float    dynatemp_exponent = 1.0f;

    // XTC
    float    xtc_probability = 0.0f;   // 0 = disabled
    float    xtc_threshold   = 0.1f;

    // top-n-sigma
    float    top_n_sigma     = -1.0f;  // < 0 = disabled

    // DRY
    float    dry_multiplier     = 0.0f;  // 0 = disabled
    float    dry_base           = 1.75f;
    int      dry_allowed_length = 2;
    int      dry_penalty_last_n = -1;
    std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };

    // adaptive-p: replaces dist at the end of the chain when enabled
    float    adaptive_target = -1.0f;  // < 0 = disabled
    float    adaptive_decay  = 0.9f;

    // fill-in-the-middle
    bool     infill = false;

    // logit bias: parallel token/value arrays
    std::vector<llama_logit_bias> logit_bias;
};

// Look up a named element of an R list; returns R_NilValue when absent.
static SEXP llamar_list_elt(SEXP list, const char * name) {
    if (Rf_isNull(list) || !Rf_isNewList(list)) return R_NilValue;
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (Rf_isNull(names)) return R_NilValue;
    const int n = (int) Rf_length(list);
    for (int i = 0; i < n; i++) {
        const char * nm = CHAR(STRING_ELT(names, i));
        if (strcmp(nm, name) == 0) {
            SEXP v = VECTOR_ELT(list, i);
            return Rf_isNull(v) ? R_NilValue : v;
        }
    }
    return R_NilValue;
}

static void llamar_get_float(SEXP list, const char * name, float * out) {
    SEXP v = llamar_list_elt(list, name);
    if (Rf_isNull(v) || Rf_length(v) < 1) return;
    double d = Rf_asReal(v);
    if (!ISNA(d)) *out = (float) d;
}

static void llamar_get_int(SEXP list, const char * name, int * out) {
    SEXP v = llamar_list_elt(list, name);
    if (Rf_isNull(v) || Rf_length(v) < 1) return;
    int i = Rf_asInteger(v);
    if (i != NA_INTEGER) *out = i;
}

static void llamar_get_uint(SEXP list, const char * name, uint32_t * out) {
    SEXP v = llamar_list_elt(list, name);
    if (Rf_isNull(v) || Rf_length(v) < 1) return;
    int i = Rf_asInteger(v);
    if (i != NA_INTEGER) *out = (uint32_t) i;
}

static void llamar_get_bool(SEXP list, const char * name, bool * out) {
    SEXP v = llamar_list_elt(list, name);
    if (Rf_isNull(v) || Rf_length(v) < 1) return;
    int b = Rf_asLogical(v);
    if (b != NA_LOGICAL) *out = (b == TRUE);
}

// Read a sampler-parameter list (as built by llama_sampler_params() in R) into
// the POD struct. Missing or NULL fields keep their defaults, so a partial list
// is legal and an empty/NULL list yields the defaults.
static llamar_sampler_params llamar_sampler_params_from_sexp(SEXP r_params) {
    llamar_sampler_params p;
    if (Rf_isNull(r_params)) return p;
    if (!Rf_isNewList(r_params)) {
        llamar_error("llamaR: sampler parameters must be a named list");
    }

    llamar_get_float(r_params, "temp",           &p.temp);
    llamar_get_int  (r_params, "top_k",          &p.top_k);
    llamar_get_float(r_params, "top_p",          &p.top_p);
    llamar_get_float(r_params, "min_p",          &p.min_p);
    llamar_get_float(r_params, "typical_p",      &p.typical_p);
    llamar_get_uint (r_params, "seed",           &p.seed);
    llamar_get_int  (r_params, "min_keep",       &p.min_keep);

    llamar_get_float(r_params, "repeat_penalty", &p.repeat_penalty);
    llamar_get_int  (r_params, "repeat_last_n",  &p.repeat_last_n);
    llamar_get_float(r_params, "frequency_penalty", &p.freq_penalty);
    llamar_get_float(r_params, "presence_penalty",  &p.pres_penalty);

    llamar_get_int  (r_params, "mirostat",       &p.mirostat);
    llamar_get_float(r_params, "mirostat_tau",   &p.mirostat_tau);
    llamar_get_float(r_params, "mirostat_eta",   &p.mirostat_eta);

    llamar_get_float(r_params, "dynatemp_range",    &p.dynatemp_range);
    llamar_get_float(r_params, "dynatemp_exponent", &p.dynatemp_exponent);

    llamar_get_float(r_params, "xtc_probability", &p.xtc_probability);
    llamar_get_float(r_params, "xtc_threshold",   &p.xtc_threshold);

    llamar_get_float(r_params, "top_n_sigma",     &p.top_n_sigma);

    llamar_get_float(r_params, "dry_multiplier",     &p.dry_multiplier);
    llamar_get_float(r_params, "dry_base",           &p.dry_base);
    llamar_get_int  (r_params, "dry_allowed_length", &p.dry_allowed_length);
    llamar_get_int  (r_params, "dry_penalty_last_n", &p.dry_penalty_last_n);

    SEXP r_breakers = llamar_list_elt(r_params, "dry_sequence_breakers");
    if (!Rf_isNull(r_breakers)) {
        if (TYPEOF(r_breakers) != STRSXP) {
            llamar_error("llamaR: dry_sequence_breakers must be a character vector");
        }
        p.dry_sequence_breakers.clear();
        const int n = (int) Rf_length(r_breakers);
        p.dry_sequence_breakers.reserve(n);
        for (int i = 0; i < n; i++) {
            p.dry_sequence_breakers.push_back(CHAR(STRING_ELT(r_breakers, i)));
        }
    }

    llamar_get_float(r_params, "adaptive_target", &p.adaptive_target);
    llamar_get_float(r_params, "adaptive_decay",  &p.adaptive_decay);

    llamar_get_bool (r_params, "infill", &p.infill);

    // logit_bias: a named numeric vector is not usable here (names are strings),
    // so it arrives as a list(token = <int>, bias = <double>) of equal length.
    SEXP r_lb = llamar_list_elt(r_params, "logit_bias");
    if (!Rf_isNull(r_lb)) {
        SEXP r_tok = llamar_list_elt(r_lb, "token");
        SEXP r_val = llamar_list_elt(r_lb, "bias");
        if (Rf_isNull(r_tok) || Rf_isNull(r_val)) {
            llamar_error("llamaR: logit_bias must be a list with 'token' and 'bias'");
        }
        SEXP tok = PROTECT(Rf_coerceVector(r_tok, INTSXP));
        SEXP val = PROTECT(Rf_coerceVector(r_val, REALSXP));
        const int n = (int) Rf_length(tok);
        if ((int) Rf_length(val) != n) {
            UNPROTECT(2);  // balance the stack before unwinding
            llamar_error("llamaR: logit_bias 'token' and 'bias' must have equal length");
        }
        p.logit_bias.reserve(n);
        for (int i = 0; i < n; i++) {
            llama_logit_bias lb;
            lb.token = (llama_token) INTEGER(tok)[i];
            lb.bias  = (float) REAL(val)[i];
            p.logit_bias.push_back(lb);
        }
        UNPROTECT(2);
    }

    return p;
}

// The sampler argument of the generation functions is either a parameter list
// (the declarative path, llama_sampler_params()) or a chain handle built by
// hand through the chain API. These two live at the bottom of this file; the
// generation entry points need them here.
struct llamar_sampler_handle;
static llamar_sampler_handle * llamar_chain_handle_get(SEXP r_chain);
static llamar_sampler_handle * llamar_chain_handle_get_throw(SEXP r_chain);
static llama_sampler * llamar_handle_chain_ptr(llamar_sampler_handle * h);

// True when the sampler argument is a chain handle rather than a parameter
// list. A NULL/absent argument and a plain list both mean "build it yourself".
static bool llamar_is_chain_arg(SEXP r_sampler) {
    return !Rf_isNull(r_sampler) && TYPEOF(r_sampler) == EXTPTRSXP;
}

// Take a caller-supplied chain and hand back a private copy for this
// generation. Generation mutates sampler state (mirostat's mu, adaptive-p's
// moving average, the penalty samplers' history, the grammar stack), so the
// caller's chain is never used directly: the copy keeps the two independent
// and sidesteps every question about which side frees the chain and when.
//
// reset_state clears the copy's accumulated state, which is the default and
// matches how the parameter-list path behaves (a freshly built chain each
// call). Passing FALSE carries the caller's state into the generation, for
// deliberately continuing where a previous one left off.
// Only the generation entry points call this, and they all wrap their body, so
// the throwing flavour is the right one here.
static llama_sampler * llamar_chain_copy_for_generation(SEXP r_sampler, bool reset_state) {
    llamar_sampler_handle * h = llamar_chain_handle_get_throw(r_sampler);
    llama_sampler * copy = llama_sampler_clone(llamar_handle_chain_ptr(h));
    if (!copy) llamar_error("llamaR: this sampler chain cannot be copied");
    if (reset_state) llama_sampler_reset(copy);
    return copy;
}

// Build the full sampler chain onto smpl, in the same order llama.cpp's
// common/sampling.cpp uses: grammar first (so it constrains logits before
// anything else), then logit_bias, then penalties → dry → top_n_sigma → top_k →
// typical → top_p → min_p → xtc → temp, and finally a token-selecting sampler
// (dist / adaptive_p / greedy, or the mirostat pair).
//
// seed_override lets llama_generate_batch give each sequence its own seed
// without copying the whole parameter struct; pass p.seed to keep it as-is.
static void llamar_build_sampler_chain(llama_sampler * smpl,
                                       const llama_model * model,
                                       const llama_vocab * vocab,
                                       const llamar_sampler_params & p,
                                       uint32_t seed,
                                       const char * grammar,
                                       SEXP r_trig_pat, SEXP r_trig_tok) {
    llamar_add_grammar_sampler(smpl, vocab, grammar, r_trig_pat, r_trig_tok);

    if (!p.logit_bias.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab),
            (int32_t) p.logit_bias.size(),
            p.logit_bias.data()));
    }

    if (p.mirostat != 0) {
        // Mirostat replaces the truncation samplers entirely; it needs a
        // temperature above zero to have a distribution to work on.
        const float t = p.temp > 0.0f ? p.temp : 0.8f;
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(t));
        if (p.mirostat == 1) {
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab), seed, p.mirostat_tau, p.mirostat_eta, 100));
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
                seed, p.mirostat_tau, p.mirostat_eta));
        }
        return;
    }

    const size_t min_keep = (size_t) (p.min_keep > 0 ? p.min_keep : 1);

    if (p.repeat_penalty != 1.0f || p.freq_penalty != 0.0f || p.pres_penalty != 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            p.repeat_last_n, p.repeat_penalty, p.freq_penalty, p.pres_penalty));
    }

    if (p.dry_multiplier > 0.0f) {
        std::vector<const char *> breakers;
        breakers.reserve(p.dry_sequence_breakers.size());
        for (const auto & s : p.dry_sequence_breakers) breakers.push_back(s.c_str());
        llama_sampler_chain_add(smpl, llama_sampler_init_dry(
            vocab, llama_model_n_ctx_train(model),
            p.dry_multiplier, p.dry_base,
            p.dry_allowed_length, p.dry_penalty_last_n,
            breakers.data(), breakers.size()));
    }

    if (p.top_n_sigma >= 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_top_n_sigma(p.top_n_sigma));
    if (p.top_k > 0)
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p.top_k));
    if (p.typical_p < 1.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(p.typical_p, min_keep));
    if (p.top_p < 1.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p.top_p, min_keep));
    if (p.min_p > 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(p.min_p, min_keep));
    if (p.xtc_probability > 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
            p.xtc_probability, p.xtc_threshold, min_keep, seed));

    if (p.infill)
        llama_sampler_chain_add(smpl, llama_sampler_init_infill(vocab));

    if (p.temp > 0.0f) {
        // temp_ext with range 0 behaves exactly like plain temp
        llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(
            p.temp, p.dynatemp_range, p.dynatemp_exponent));
    }

    // Final, token-selecting sampler.
    if (p.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else if (p.adaptive_target >= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_adaptive_p(
            p.adaptive_target, p.adaptive_decay, seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
}

extern "C" SEXP r_llama_generate(SEXP r_ctx, SEXP r_prompt,
                                  SEXP r_max_new_tokens, SEXP r_params,
                                  SEXP r_grammar, SEXP r_with_timings,
                                  SEXP r_trigger_patterns, SEXP r_trigger_tokens,
                                  SEXP r_sampler_reset) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * prompt         = CHAR(STRING_ELT(r_prompt, 0));
    int          max_new_tokens = INTEGER(r_max_new_tokens)[0];
    // A chain handle carries its own samplers, so the parameter struct stays at
    // its defaults and is unused in that case.
    const bool   chain_arg      = llamar_is_chain_arg(r_params);
    const llamar_sampler_params sp =
        chain_arg ? llamar_sampler_params{} : llamar_sampler_params_from_sexp(r_params);
    const char * grammar        = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));
    bool         with_timings   = (Rf_asLogical(r_with_timings) == TRUE);
    const bool   sampler_reset  = (Rf_asLogical(r_sampler_reset) != FALSE);

    int prompt_len = (int) strlen(prompt);

    using clk = std::chrono::steady_clock;
    auto t_total_start = clk::now();
    double t_tokenize_ms = 0, t_build_sampler_ms = 0, t_kv_clear_ms = 0;
    double t_prefill_dispatch_ms = 0, t_prefill_sync_ms = 0;
    double t_gpu_sync_ms = 0, t_sample_ms = 0, t_decode_dispatch_ms = 0;
    double t_post_decode_sync_ms = 0;
    double t_detokenize_ms = 0;
    int    n_iterations = 0;
    int    n_splits_prefill = 0;
    int    n_splits_decode  = 0;

    auto tic = clk::now();

    // --- tokenize prompt ---
    // parse_special = true: the prompt has already been through the chat
    // template, so role markers like [INST]/<|im_start|> are control tokens
    // and must be parsed as such, not split into literal characters.
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) {
        llamar_error("llamaR: prompt produced zero tokens");
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    int actual = llama_tokenize(vocab, prompt, prompt_len,
                                prompt_tokens.data(), n_tokens, true, true);
    if (actual < 0) llamar_error("llamaR: tokenization failed");
    n_tokens = actual;

    if (with_timings) {
        t_tokenize_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

    // --- build sampler chain ---
    // A caller-supplied chain is copied rather than used in place; see
    // llamar_chain_copy_for_generation.
    llama_sampler * smpl;
    if (chain_arg) {
        smpl = llamar_chain_copy_for_generation(r_params, sampler_reset);
    } else {
        auto sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llamar_build_sampler_chain(smpl, model, vocab, sp, sp.seed, grammar,
                                   r_trigger_patterns, r_trigger_tokens);
    }

    if (with_timings) {
        t_build_sampler_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

    // --- clear KV cache ---
    llama_memory_clear(llama_get_memory(ctx), true);

    if (with_timings) {
        t_kv_clear_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

    // --- prefill: dispatch (chunked by n_batch) ---
    // A single llama_decode carries at most n_batch tokens; split long
    // prompts so we never trip llama.cpp's n_tokens <= n_batch assert.
    struct llama_batch batch;
    {
        int n_batch = (int) llama_n_batch(ctx);
        if (n_batch <= 0) n_batch = n_tokens;
        for (int off = 0; off < n_tokens; off += n_batch) {
            int n_chunk = (n_tokens - off < n_batch) ? (n_tokens - off) : n_batch;
            batch = llama_batch_get_one(prompt_tokens.data() + off, n_chunk);
            if (llama_decode(ctx, batch) != 0) {
                llama_sampler_free(smpl);
                llamar_error("llamaR: failed to process prompt (chunk at offset %d of %d)",
                             off, n_tokens);
            }
        }
    }

    if (with_timings) {
        t_prefill_dispatch_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
        // honest GPU wait for prefill, isolated from first sample
        llama_synchronize(ctx);
        t_prefill_sync_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        n_splits_prefill = llama_n_splits(ctx);
    }

    // --- autoregressive decode loop ---
    std::vector<llama_token> generated;
    llama_token current_token;

    for (int i = 0; i < max_new_tokens; i++) {
        if (with_timings) {
            // before sampling, ensure GPU is done with the previous decode dispatch
            // (skipped on i=0 because prefill_sync already drained it)
            if (i > 0) {
                auto t0 = clk::now();
                llama_synchronize(ctx);
                t_gpu_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            }
        }

        auto t_s0 = with_timings ? clk::now() : clk::time_point{};
        current_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, current_token)) {
            if (with_timings) {
                t_sample_ms += std::chrono::duration<double, std::milli>(clk::now() - t_s0).count();
                n_iterations = i + 1;
            }
            break;
        }

        generated.push_back(current_token);
        // NOTE: llama_sampler_sample() already calls llama_sampler_accept()
        // internally, so we must NOT accept again here. A double-accept is
        // harmless for stateless samplers (penalties) but corrupts the grammar
        // sampler: it advances the grammar stack twice per token, so the second
        // advance has no valid transition and aborts with "Unexpected empty
        // grammar stack". This broke all grammar-constrained generation.

        if (with_timings) {
            t_sample_ms += std::chrono::duration<double, std::milli>(clk::now() - t_s0).count();
        }

        auto t_d0 = with_timings ? clk::now() : clk::time_point{};
        batch = llama_batch_get_one(&current_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            llamar_error("llamaR: failed during token generation");
        }
        if (with_timings) {
            t_decode_dispatch_ms += std::chrono::duration<double, std::milli>(clk::now() - t_d0).count();
            // immediate post-dispatch drain: how much GPU work remains right
            // after llama_decode returns? compare to t_gpu_sync_ms which is
            // measured at the *start* of the next iteration (after batch_get_one etc.)
            auto t_p0 = clk::now();
            llama_synchronize(ctx);
            t_post_decode_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - t_p0).count();
            if (i == 0) {
                n_splits_decode = llama_n_splits(ctx);
            }
            n_iterations = i + 1;
        }
    }

    llama_sampler_free(smpl);

    // --- detokenize generated tokens ---
    if (generated.empty()) {
        SEXP empty = PROTECT(Rf_mkString(""));
        if (with_timings) {
            double t_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total_start).count();
            const char * names[] = {
                "t_tokenize_ms", "t_build_sampler_ms", "t_kv_clear_ms",
                "t_prefill_dispatch_ms", "t_prefill_sync_ms",
                "t_gpu_sync_ms", "t_sample_ms", "t_decode_dispatch_ms",
                "t_post_decode_sync_ms", "t_detokenize_ms",
                "n_iterations", "n_splits_prefill", "n_splits_decode",
                "t_total_ms", NULL
            };
            SEXP timings = PROTECT(Rf_allocVector(REALSXP, 14));
            REAL(timings)[0]  = t_tokenize_ms;
            REAL(timings)[1]  = t_build_sampler_ms;
            REAL(timings)[2]  = t_kv_clear_ms;
            REAL(timings)[3]  = t_prefill_dispatch_ms;
            REAL(timings)[4]  = t_prefill_sync_ms;
            REAL(timings)[5]  = t_gpu_sync_ms;
            REAL(timings)[6]  = t_sample_ms;
            REAL(timings)[7]  = t_decode_dispatch_ms;
            REAL(timings)[8]  = t_post_decode_sync_ms;
            REAL(timings)[9]  = 0.0;  // detokenize (empty path)
            REAL(timings)[10] = (double) n_iterations;
            REAL(timings)[11] = (double) n_splits_prefill;
            REAL(timings)[12] = (double) n_splits_decode;
            REAL(timings)[13] = t_total_ms;
            SEXP nm = PROTECT(Rf_allocVector(STRSXP, 14));
            for (int k = 0; k < 14; k++) SET_STRING_ELT(nm, k, Rf_mkChar(names[k]));
            Rf_setAttrib(timings, R_NamesSymbol, nm);
            Rf_setAttrib(empty, Rf_install("timings"), timings);
            UNPROTECT(2);
        }
        UNPROTECT(1);
        return empty;
    }

    if (with_timings) tic = clk::now();

    int text_len = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                    NULL, 0, false, false);
    if (text_len < 0) text_len = -text_len;

    std::vector<char> text(text_len + 1);
    int result = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                  text.data(), text_len, false, false);
    if (result < 0) result = 0;
    text[result] = '\0';

    if (with_timings) {
        t_detokenize_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
    }

    // Detokenized output can contain byte-fragment tokens that do not form
    // valid UTF-8; sanitize before it reaches R (see utf8_sanitize).
    const std::string clean_text = utf8_sanitize(text.data(), (size_t) result);
    SEXP r_text = PROTECT(Rf_ScalarString(
        Rf_mkCharLenCE(clean_text.data(), (int) clean_text.size(), CE_UTF8)));
    if (with_timings) {
        double t_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total_start).count();
        const char * names[] = {
            "t_tokenize_ms", "t_build_sampler_ms", "t_kv_clear_ms",
            "t_prefill_dispatch_ms", "t_prefill_sync_ms",
            "t_gpu_sync_ms", "t_sample_ms", "t_decode_dispatch_ms",
            "t_post_decode_sync_ms", "t_detokenize_ms",
            "n_iterations", "n_splits_prefill", "n_splits_decode",
            "t_total_ms", NULL
        };
        SEXP timings = PROTECT(Rf_allocVector(REALSXP, 14));
        REAL(timings)[0]  = t_tokenize_ms;
        REAL(timings)[1]  = t_build_sampler_ms;
        REAL(timings)[2]  = t_kv_clear_ms;
        REAL(timings)[3]  = t_prefill_dispatch_ms;
        REAL(timings)[4]  = t_prefill_sync_ms;
        REAL(timings)[5]  = t_gpu_sync_ms;
        REAL(timings)[6]  = t_sample_ms;
        REAL(timings)[7]  = t_decode_dispatch_ms;
        REAL(timings)[8]  = t_post_decode_sync_ms;
        REAL(timings)[9]  = t_detokenize_ms;
        REAL(timings)[10] = (double) n_iterations;
        REAL(timings)[11] = (double) n_splits_prefill;
        REAL(timings)[12] = (double) n_splits_decode;
        REAL(timings)[13] = t_total_ms;
        SEXP nm = PROTECT(Rf_allocVector(STRSXP, 14));
        for (int k = 0; k < 14; k++) SET_STRING_ELT(nm, k, Rf_mkChar(names[k]));
        Rf_setAttrib(timings, R_NamesSymbol, nm);
        Rf_setAttrib(r_text, Rf_install("timings"), timings);
        UNPROTECT(2);
    }
    UNPROTECT(1);
    return r_text;
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Streaming generation: begin / next / end (token-by-token)
// ============================================================
//
// Splits r_llama_generate into three calls so callers can pull one chunk of
// text at a time (e.g. to push into an SSE stream). State lives in an
// externalptr with a GC finalizer that frees the sampler chain.

// Length in bytes of the trailing run of bytes in `s` that forms an
// incomplete (truncated) UTF-8 sequence. Returns 0 when `s` ends on a
// complete character. Only the final, still-growing code point is held back.
static size_t utf8_incomplete_tail(const std::string & s) {
    size_t n = s.size();
    // Scan back over continuation bytes (10xxxxxx) to find the last lead byte.
    size_t i = n;
    while (i > 0) {
        unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) != 0x80) {  // not a continuation byte: this is the lead
            i--;
            break;
        }
        i--;
    }
    if (i >= n) return 0;  // last byte was itself a lead with no body, handled below
    unsigned char lead = (unsigned char) s[i];
    size_t need;
    if      ((lead & 0x80) == 0x00) need = 1;  // 0xxxxxxx
    else if ((lead & 0xE0) == 0xC0) need = 2;  // 110xxxxx
    else if ((lead & 0xF0) == 0xE0) need = 3;  // 1110xxxx
    else if ((lead & 0xF8) == 0xF0) need = 4;  // 11110xxx
    else return 0;                             // stray continuation byte: emit as-is
    size_t have = n - i;
    return have < need ? have : 0;  // hold back only if the char is unfinished
}

extern "C" SEXP r_llama_gen_begin(SEXP r_ctx, SEXP r_prompt,
                                  SEXP r_max_new_tokens, SEXP r_params,
                                  SEXP r_grammar,
                                  SEXP r_trigger_patterns, SEXP r_trigger_tokens,
                                  SEXP r_sampler_reset) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * prompt         = CHAR(STRING_ELT(r_prompt, 0));
    int          max_new_tokens = INTEGER(r_max_new_tokens)[0];
    const bool   chain_arg      = llamar_is_chain_arg(r_params);
    const llamar_sampler_params sp =
        chain_arg ? llamar_sampler_params{} : llamar_sampler_params_from_sexp(r_params);
    const char * grammar        = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));
    const bool   sampler_reset  = (Rf_asLogical(r_sampler_reset) != FALSE);

    int prompt_len = (int) strlen(prompt);

    // --- tokenize prompt ---
    // parse_special = true: prompt has been through the chat template, so its
    // role markers are control tokens (see r_llama_generate for rationale).
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) llamar_error("llamaR: prompt produced zero tokens");

    std::vector<llama_token> prompt_tokens(n_tokens);
    int actual = llama_tokenize(vocab, prompt, prompt_len,
                                prompt_tokens.data(), n_tokens, true, true);
    if (actual < 0) llamar_error("llamaR: tokenization failed");
    n_tokens = actual;

    // --- build sampler chain (identical to r_llama_generate) ---
    // The state takes ownership of this chain either way: a caller-supplied one
    // is copied first, so the caller's chain is never freed by the finalizer
    // and needs no lifetime guarantees of its own.
    llama_sampler * smpl;
    if (chain_arg) {
        smpl = llamar_chain_copy_for_generation(r_params, sampler_reset);
    } else {
        auto sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llamar_build_sampler_chain(smpl, model, vocab, sp, sp.seed, grammar,
                                   r_trigger_patterns, r_trigger_tokens);
    }

    // --- clear KV cache and prefill the prompt ---
    // Split the prefill into chunks of at most n_batch tokens: a single
    // llama_decode call may carry no more than n_batch tokens (llama.cpp
    // asserts otherwise). Positions continue automatically across calls
    // since the KV cache was just cleared and grows with each decode.
    llama_memory_clear(llama_get_memory(ctx), true);
    int n_batch = (int) llama_n_batch(ctx);
    if (n_batch <= 0) n_batch = n_tokens;
    for (int off = 0; off < n_tokens; off += n_batch) {
        int n_chunk = (n_tokens - off < n_batch) ? (n_tokens - off) : n_batch;
        struct llama_batch batch = llama_batch_get_one(prompt_tokens.data() + off, n_chunk);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            llamar_error("llamaR: failed to process prompt (chunk at offset %d of %d)",
                         off, n_tokens);
        }
    }

    llama_gen_state * st = new llama_gen_state();
    st->ctx         = ctx;
    st->vocab       = vocab;
    st->smpl        = smpl;
    st->n_remaining = max_new_tokens;
    st->done        = false;

    SEXP ptr = PROTECT(R_MakeExternalPtr(st, Rf_install("llama_gen_state"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, gen_state_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
    LLAMAR_ENTRYPOINT_END
}

// Begin generation from an ALREADY-PREFILLED context, without clearing the KV
// cache or tokenizing/decoding a prompt. Used after r_mtmd_eval has decoded
// text+image chunks into the context (logits_last=true leaves logits ready for
// the last position): we just build the sampler chain and hand back a gen_state
// so the usual r_llama_gen_next loop can continue from where the image left off.
// n_past is accepted for API symmetry; the context already tracks its own KV
// position, and r_llama_gen_next samples from the last logits (index -1).
extern "C" SEXP r_llama_gen_begin_at(SEXP r_ctx, SEXP r_n_past,
                                     SEXP r_max_new_tokens, SEXP r_params,
                                     SEXP r_grammar,
                                     SEXP r_trigger_patterns, SEXP r_trigger_tokens,
                                     SEXP r_sampler_reset) {
    LLAMAR_ENTRYPOINT_BEGIN
    (void) r_n_past;
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int          max_new_tokens = INTEGER(r_max_new_tokens)[0];
    const bool   chain_arg      = llamar_is_chain_arg(r_params);
    const llamar_sampler_params sp =
        chain_arg ? llamar_sampler_params{} : llamar_sampler_params_from_sexp(r_params);
    const char * grammar        = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));
    const bool   sampler_reset  = (Rf_asLogical(r_sampler_reset) != FALSE);

    // --- build sampler chain (identical to r_llama_gen_begin) ---
    llama_sampler * smpl;
    if (chain_arg) {
        smpl = llamar_chain_copy_for_generation(r_params, sampler_reset);
    } else {
        auto sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llamar_build_sampler_chain(smpl, model, vocab, sp, sp.seed, grammar,
                                   r_trigger_patterns, r_trigger_tokens);
    }

    // NB: deliberately NO llama_memory_clear / prefill here — the caller
    // (r_mtmd_eval) has already decoded the prompt + image into the KV cache.

    llama_gen_state * st = new llama_gen_state();
    st->ctx         = ctx;
    st->vocab       = vocab;
    st->smpl        = smpl;
    st->n_remaining = max_new_tokens;
    st->done        = false;

    SEXP ptr = PROTECT(R_MakeExternalPtr(st, Rf_install("llama_gen_state"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, gen_state_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
    LLAMAR_ENTRYPOINT_END
}

// One generation step. Returns a length-1 character vector with the next
// chunk of text (possibly ""), or NULL when generation is finished (EOG
// reached or token budget exhausted). Holds back an incomplete trailing
// UTF-8 char until the next call; r_llama_gen_end() flushes any remainder.
extern "C" SEXP r_llama_gen_next(SEXP r_state) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_gen_state * st = llamar_gen_state_arg_throw(r_state);
    if (st->done || st->n_remaining <= 0) {
        st->done = true;
        return R_NilValue;
    }

    llama_token tok = llama_sampler_sample(st->smpl, st->ctx, -1);
    if (llama_vocab_is_eog(st->vocab, tok)) {
        st->done = true;
        return R_NilValue;
    }

    // NOTE: no llama_sampler_accept() here — llama_sampler_sample() already
    // accepted this token. A second accept double-advances the grammar sampler
    // and aborts with "Unexpected empty grammar stack" (see llama_generate).
    st->n_remaining--;

    // detokenize this single token, appending to the UTF-8 carry buffer
    char piece[256];
    int np = llama_token_to_piece(st->vocab, tok, piece, sizeof(piece), 0, false);
    if (np < 0) {
        std::vector<char> big(-np);
        np = llama_token_to_piece(st->vocab, tok, big.data(), (int) big.size(), 0, false);
        if (np > 0) st->utf8_buf.append(big.data(), np);
    } else if (np > 0) {
        st->utf8_buf.append(piece, np);
    }

    // decode the accepted token to advance the context for the next step
    struct llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(st->ctx, batch) != 0) {
        st->done = true;
        llamar_error("llamaR: failed during token generation");
    }

    // Emit everything except a possibly-incomplete trailing UTF-8 char, which
    // is held back until the next token supplies its remaining bytes. What is
    // emitted still goes through utf8_sanitize: a byte sequence can be broken
    // in ways waiting cannot fix (see its comment), and every chunk handed to
    // R must be valid UTF-8.
    size_t tail = utf8_incomplete_tail(st->utf8_buf);
    size_t emit = st->utf8_buf.size() - tail;
    const std::string clean = utf8_sanitize(st->utf8_buf.data(), emit);
    SEXP r_text = PROTECT(Rf_ScalarString(
        Rf_mkCharLenCE(clean.data(), (int) clean.size(), CE_UTF8)));
    st->utf8_buf.erase(0, emit);
    UNPROTECT(1);
    return r_text;
    LLAMAR_ENTRYPOINT_END
}

// Flush any bytes still held in the carry buffer and mark the state done.
// Safe to call multiple times. The sampler is freed by the GC finalizer.
extern "C" SEXP r_llama_gen_end(SEXP r_state) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_gen_state * st = llamar_gen_state_arg_throw(r_state);
    st->done = true;
    // Whatever is left in the carry buffer is an incomplete UTF-8 character:
    // generation stopped mid-character (max_new_tokens reached, say) and the
    // rest will never arrive. Sanitizing turns the stub into U+FFFD instead of
    // handing R bytes that claim to be UTF-8 and are not.
    const std::string clean = utf8_sanitize(st->utf8_buf.data(), st->utf8_buf.size());
    SEXP r_text = PROTECT(Rf_ScalarString(
        Rf_mkCharLenCE(clean.data(), (int) clean.size(), CE_UTF8)));
    st->utf8_buf.clear();
    UNPROTECT(1);
    return r_text;
    LLAMAR_ENTRYPOINT_END
}

// Sampler timings. These live on the sampler chain, which only outlives a
// single call on the streaming path — hence a gen_state, not a context.
extern "C" SEXP r_llama_perf_sampler(SEXP r_state) {
    llama_gen_state * st = llamar_gen_state_arg(r_state);
    if (!st->smpl) Rf_error("llamaR: generation state has no sampler");

    struct llama_perf_sampler_data perf = llama_perf_sampler(st->smpl);

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(perf.t_sample_ms));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(perf.n_sample));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("t_sample_ms"));
    SET_STRING_ELT(names, 1, Rf_mkChar("n_sample"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_perf_sampler_print(SEXP r_state) {
    llama_gen_state * st = llamar_gen_state_arg(r_state);
    if (!st->smpl) Rf_error("llamaR: generation state has no sampler");
    llama_perf_sampler_print(st->smpl);
    return R_NilValue;
}

extern "C" SEXP r_llama_perf_sampler_reset(SEXP r_state) {
    llama_gen_state * st = llamar_gen_state_arg(r_state);
    if (!st->smpl) Rf_error("llamaR: generation state has no sampler");
    llama_perf_sampler_reset(st->smpl);
    return R_NilValue;
}

// ============================================================
// Generate batch: continuous batching for N independent prompts
// ============================================================

// Forward declaration; tokenize_text is defined later in the Embeddings section.
static int tokenize_text(const llama_vocab * vocab, const char * text,
                         std::vector<llama_token> & out);

extern "C" SEXP r_llama_generate_batch(SEXP r_ctx, SEXP r_prompts,
                                       SEXP r_max_new_tokens, SEXP r_params,
                                       SEXP r_grammar,
                                       SEXP r_trigger_patterns, SEXP r_trigger_tokens) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_seq = Rf_length(r_prompts);
    if (n_seq <= 0) llamar_error("llamaR: prompts must be non-empty character vector");

    int max_new_tokens = INTEGER(r_max_new_tokens)[0];

    // Each sequence needs its own chain seeded differently (seed + s), which a
    // single supplied chain cannot provide: copies would share one seed and the
    // sequences would stop being independent, and there is no public setter to
    // re-seed a copy. Refuse outright rather than silently ignoring the chain.
    if (llamar_is_chain_arg(r_params)) {
        llamar_error("llamaR: llama_generate_batch does not accept a sampler chain; "
                     "pass a parameter list from llama_sampler_params() instead");
    }

    const llamar_sampler_params sp = llamar_sampler_params_from_sexp(r_params);
    const char * grammar    = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));

    // Capacity check: all prompts + max_new_tokens for all seqs must fit in n_ctx
    {
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t n_seq_max = llama_n_seq_max(ctx);
        if ((uint32_t) n_seq > n_seq_max) {
            llamar_error("llamaR: %d prompts exceeds context n_seq_max=%u "
                         "(rebuild context with larger n_seq_max)", n_seq, n_seq_max);
        }
        // Capacity check itself happens after tokenization below.
        (void) n_ctx;
    }

    int64_t t_batch_build = 0, t_decode = 0, t_sample = 0;
    int64_t t_prefill = 0, t_tokenize = 0, t_detokenize = 0;
    int64_t t_gpu_sync = 0;

    // --- tokenize all prompts ---
    int64_t t0_tok = llama_time_us();
    std::vector<std::vector<llama_token>> prompt_tokens(n_seq);
    int total_prompt_tokens = 0;
    for (int s = 0; s < n_seq; s++) {
        const char * p = CHAR(STRING_ELT(r_prompts, s));
        if (tokenize_text(vocab, p, prompt_tokens[s]) < 0) {
            llamar_error("llamaR: tokenization failed for prompt %d", s + 1);
        }
        if (prompt_tokens[s].empty()) {
            llamar_error("llamaR: prompt %d produced zero tokens", s + 1);
        }
        total_prompt_tokens += (int) prompt_tokens[s].size();
    }
    t_tokenize += llama_time_us() - t0_tok;

    {
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t need  = (uint32_t) total_prompt_tokens + (uint32_t) (n_seq * max_new_tokens);
        if (need > n_ctx) {
            llamar_error("llamaR: required tokens (%u) exceed context n_ctx=%u "
                         "(prompts=%d, max_new=%d)", need, n_ctx, total_prompt_tokens, max_new_tokens);
        }
    }

    // --- build per-seq sampler chains (seed = base_seed + seq_id) ---
    auto build_sampler = [&](uint32_t s_seed) -> llama_sampler * {
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llamar_build_sampler_chain(smpl, model, vocab, sp, s_seed, grammar,
                                   r_trigger_patterns, r_trigger_tokens);
        return smpl;
    };

    std::vector<llama_sampler *> smpls(n_seq, nullptr);
    for (int s = 0; s < n_seq; s++) {
        smpls[s] = build_sampler(sp.seed + (uint32_t) s);
    }

    auto cleanup_samplers = [&]() {
        for (int s = 0; s < n_seq; s++) {
            if (smpls[s]) { llama_sampler_free(smpls[s]); smpls[s] = nullptr; }
        }
    };

    // --- prefill: pack all prompts into one batch with distinct seq_ids ---
    llama_memory_clear(llama_get_memory(ctx), true);

    struct llama_batch batch = llama_batch_init(total_prompt_tokens, 0, n_seq);
    std::vector<int> last_logits_idx(n_seq, -1);  // position in batch of last prompt token per seq
    std::vector<int> seq_pos(n_seq, 0);            // next position to write for each seq
    {
        int pos = 0;
        for (int s = 0; s < n_seq; s++) {
            int n_p = (int) prompt_tokens[s].size();
            for (int t = 0; t < n_p; t++) {
                batch.token[pos]     = prompt_tokens[s][t];
                batch.pos[pos]       = (llama_pos) t;
                batch.n_seq_id[pos]  = 1;
                batch.seq_id[pos][0] = (llama_seq_id) s;
                batch.logits[pos]    = (t == n_p - 1) ? 1 : 0;
                pos++;
            }
            last_logits_idx[s] = pos - 1;
            seq_pos[s]         = n_p;
        }
        batch.n_tokens = total_prompt_tokens;
    }

    int64_t t0_prefill = llama_time_us();
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        cleanup_samplers();
        llamar_error("llamaR: prefill decode failed");
    }
    t_prefill += llama_time_us() - t0_prefill;
    llama_batch_free(batch);

    // --- per-seq state ---
    std::vector<bool>                       active(n_seq, true);
    std::vector<int>                        n_generated(n_seq, 0);
    std::vector<std::vector<llama_token>>   generated(n_seq);
    std::vector<int>                        finished(n_seq, 0);  // 0=running, 1=eos, 2=max_tokens

    // Sample first token per seq from prefill logits
    llama_memory_t mem = llama_get_memory(ctx);
    int n_active = n_seq;
    for (int s = 0; s < n_seq; s++) {
        llama_token tok = llama_sampler_sample(smpls[s], ctx, last_logits_idx[s]);
        if (llama_vocab_is_eog(vocab, tok) || max_new_tokens <= 0) {
            active[s]   = false;
            finished[s] = (max_new_tokens <= 0) ? 2 : 1;
            n_active--;
            llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
            continue;
        }
        generated[s].push_back(tok);
        // llama_sampler_sample() already accepted tok; no second accept (see
        // the grammar double-advance note in llama_generate).
        n_generated[s] = 1;
    }

    // --- decode loop: one token per active seq per iteration ---
    struct llama_batch dbatch = llama_batch_init(n_seq, 0, n_seq);
    int n_decode_steps = 0;
    int n_splits_first_decode = -1;
    while (n_active > 0) {
        // Build batch from last-generated tokens of each active seq
        int64_t t0_bb = llama_time_us();
        int pos = 0;
        std::vector<int> idx_in_batch(n_seq, -1);
        for (int s = 0; s < n_seq; s++) {
            if (!active[s]) continue;
            llama_token tok = generated[s].back();
            dbatch.token[pos]     = tok;
            dbatch.pos[pos]       = (llama_pos) seq_pos[s];
            dbatch.n_seq_id[pos]  = 1;
            dbatch.seq_id[pos][0] = (llama_seq_id) s;
            dbatch.logits[pos]    = 1;
            idx_in_batch[s]       = pos;
            seq_pos[s]++;
            pos++;
        }
        dbatch.n_tokens = pos;
        t_batch_build += llama_time_us() - t0_bb;

        int64_t t0_dec = llama_time_us();
        if (llama_decode(ctx, dbatch) != 0) {
            llama_batch_free(dbatch);
            cleanup_samplers();
            llamar_error("llamaR: decode failed during batch generation");
        }
        t_decode += llama_time_us() - t0_dec;
        if (n_splits_first_decode < 0) {
            n_splits_first_decode = llama_n_splits(ctx);
        }

        // Explicit GPU sync — separates "wait for GPU" from "CPU sample work"
        int64_t t0_sync = llama_time_us();
        llama_synchronize(ctx);
        t_gpu_sync += llama_time_us() - t0_sync;

        // Sample one token per active seq
        int64_t t0_smp = llama_time_us();
        for (int s = 0; s < n_seq; s++) {
            if (!active[s]) continue;
            llama_token tok = llama_sampler_sample(smpls[s], ctx, idx_in_batch[s]);

            if (llama_vocab_is_eog(vocab, tok)) {
                active[s]   = false;
                finished[s] = 1;
                n_active--;
                llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
                continue;
            }

            generated[s].push_back(tok);
            // llama_sampler_sample() already accepted tok; no second accept
            // (see the grammar double-advance note in llama_generate).
            n_generated[s]++;

            if (n_generated[s] >= max_new_tokens) {
                active[s]   = false;
                finished[s] = 2;
                n_active--;
                llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
            }
        }
        t_sample += llama_time_us() - t0_smp;
        n_decode_steps++;
    }
    llama_batch_free(dbatch);
    cleanup_samplers();

    // --- detokenize per seq, build result list ---
    int64_t t0_detok = llama_time_us();
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_seq));
    for (int s = 0; s < n_seq; s++) {
        SEXP item = PROTECT(Rf_allocVector(VECSXP, 3));

        // text
        SEXP r_text;
        if (generated[s].empty()) {
            r_text = PROTECT(Rf_mkString(""));
        } else {
            int text_len = llama_detokenize(vocab, generated[s].data(), (int) generated[s].size(),
                                            NULL, 0, false, false);
            if (text_len < 0) text_len = -text_len;
            std::vector<char> buf(text_len + 1);
            int written = llama_detokenize(vocab, generated[s].data(), (int) generated[s].size(),
                                           buf.data(), text_len, false, false);
            if (written < 0) written = 0;
            buf[written] = '\0';
            // Sanitize: byte-fragment tokens need not form valid UTF-8.
            const std::string clean = utf8_sanitize(buf.data(), (size_t) written);
            r_text = PROTECT(Rf_ScalarString(
                Rf_mkCharLenCE(clean.data(), (int) clean.size(), CE_UTF8)));
        }
        SET_VECTOR_ELT(item, 0, r_text);
        UNPROTECT(1);

        SET_VECTOR_ELT(item, 1, Rf_ScalarInteger(n_generated[s]));

        const char * reason = (finished[s] == 1) ? "eos"
                            : (finished[s] == 2) ? "max_tokens"
                            : "running";
        SET_VECTOR_ELT(item, 2, Rf_mkString(reason));

        SEXP item_names = PROTECT(Rf_allocVector(STRSXP, 3));
        SET_STRING_ELT(item_names, 0, Rf_mkChar("text"));
        SET_STRING_ELT(item_names, 1, Rf_mkChar("n_tokens"));
        SET_STRING_ELT(item_names, 2, Rf_mkChar("finished_reason"));
        Rf_setAttrib(item, R_NamesSymbol, item_names);
        UNPROTECT(1);

        SET_VECTOR_ELT(result, s, item);
        UNPROTECT(1);
    }
    t_detokenize += llama_time_us() - t0_detok;

    int64_t t_accounted = t_tokenize + t_prefill + t_batch_build + t_decode + t_gpu_sync + t_sample + t_detokenize;
    REprintf("=== Decode loop timing ===\n");
    REprintf("  t_tokenize:    %.1f ms\n", t_tokenize    / 1000.0);
    REprintf("  t_prefill:     %.1f ms\n", t_prefill     / 1000.0);
    REprintf("  t_batch_build: %.1f ms (%d steps)\n", t_batch_build / 1000.0, n_decode_steps);
    REprintf("  t_decode:      %.1f ms (%d steps, dispatch only)\n", t_decode    / 1000.0, n_decode_steps);
    REprintf("  t_gpu_sync:    %.1f ms (%d steps, GPU wait)\n",      t_gpu_sync  / 1000.0, n_decode_steps);
    REprintf("  t_sample:      %.1f ms (%d steps, pure CPU)\n",      t_sample    / 1000.0, n_decode_steps);
    REprintf("  t_detokenize:  %.1f ms\n", t_detokenize  / 1000.0);
    REprintf("  ACCOUNTED:     %.1f ms\n", t_accounted   / 1000.0);
    REprintf("  n_splits (1st decode call): %d\n", n_splits_first_decode);

    UNPROTECT(1);
    return result;
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Embeddings
// ============================================================

extern "C" SEXP r_llama_embeddings(SEXP r_ctx, SEXP r_text) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text     = CHAR(STRING_ELT(r_text, 0));
    int          text_len = (int) strlen(text);

    // tokenize
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int actual = llama_tokenize(vocab, text, text_len, tokens.data(), n_tokens, true, false);
    if (actual < 0) llamar_error("llamaR: tokenization failed");
    n_tokens = actual;

    // switch to embeddings mode, clear cache, run model
    llama_set_embeddings(ctx, true);
    llama_memory_clear(llama_get_memory(ctx), true);

    struct llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        llama_set_embeddings(ctx, false);
        llamar_error("llamaR: failed to compute embeddings (decode returned %d)", ret);
    }

    llama_synchronize(ctx);

    float * emb = llama_get_embeddings_ith(ctx, -1);
    if (!emb) {
        llama_set_embeddings(ctx, false);
        llamar_error("llamaR: embeddings output is NULL — model may not support embeddings");
    }

    int n_embd = llama_model_n_embd(model);

    // Copy out of the context's embedding buffer, then reset the flag, before
    // building the R result — keeps r_result from being live across the
    // llama_set_embeddings call (which rchk flags as allocating).
    std::vector<float> emb_copy(emb, emb + n_embd);
    llama_set_embeddings(ctx, false);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int i = 0; i < n_embd; i++) {
        REAL(r_result)[i] = (double) emb_copy[i];
    }
    UNPROTECT(1);
    return r_result;
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_get_embeddings_ith(SEXP r_ctx, SEXP r_i) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    int32_t i = INTEGER(r_i)[0];
    const llama_model * model = llama_get_model(ctx);
    int n_embd = llama_model_n_embd(model);

    float * emb = llama_get_embeddings_ith(ctx, i);
    if (!emb) Rf_error("llamaR: embeddings NULL for index %d", i);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int j = 0; j < n_embd; j++)
        REAL(r_result)[j] = (double) emb[j];
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_get_embeddings_seq(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    const llama_model * model = llama_get_model(ctx);
    int n_embd = llama_model_n_embd(model);

    float * emb = llama_get_embeddings_seq(ctx, seq_id);
    if (!emb) Rf_error("llamaR: pooled embeddings NULL for seq_id %d (model may not support pooling)", seq_id);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int j = 0; j < n_embd; j++)
        REAL(r_result)[j] = (double) emb[j];
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_get_embeddings(SEXP r_ctx, SEXP r_n_outputs) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    int n_embd    = llama_model_n_embd(model);
    int n_outputs = INTEGER(r_n_outputs)[0];
    if (n_outputs < 1) Rf_error("llamaR: n_outputs must be >= 1");

    float * emb = llama_get_embeddings(ctx);
    if (!emb) Rf_error("llamaR: embeddings NULL (no decode performed, or pooling_type != none)");

    // Return a matrix: n_outputs rows × n_embd cols (R is column-major)
    SEXP r_result = PROTECT(Rf_allocMatrix(REALSXP, n_outputs, n_embd));
    for (int i = 0; i < n_outputs * n_embd; i++)
        REAL(r_result)[i] = (double) emb[i];
    UNPROTECT(1);
    return r_result;
}

// Helper: tokenize a single C-string, returns token count
static int tokenize_text(const llama_vocab * vocab, const char * text,
                         std::vector<llama_token> & out) {
    int text_len = (int) strlen(text);
    int n_tok = llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tok < 0) n_tok = -n_tok;
    out.resize(n_tok);
    int actual = llama_tokenize(vocab, text, text_len, out.data(), n_tok, true, false);
    if (actual < 0) return -1;
    out.resize(actual);
    return actual;
}

// Helper: embed a single text using decode + embeddings_ith(-1)
static void embed_single(llama_context * ctx, const llama_vocab * vocab,
                          const char * text, float * out, int n_embd, int idx) {
    std::vector<llama_token> tokens;
    if (tokenize_text(vocab, text, tokens) < 0)
        llamar_error("llamaR: tokenization failed for text %d", idx + 1);

    llama_memory_clear(llama_get_memory(ctx), true);
    struct llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
    int ret = llama_decode(ctx, batch);
    if (ret != 0)
        llamar_error("llamaR: embed decode failed for text %d (code %d)", idx + 1, ret);
    llama_synchronize(ctx);

    float * emb = llama_get_embeddings_ith(ctx, -1);
    if (!emb)
        llamar_error("llamaR: embeddings NULL for text %d", idx + 1);
    memcpy(out, emb, n_embd * sizeof(float));
}

// Batch embeddings: pooled path (embedding=TRUE) or sequential (embedding=FALSE)
extern "C" SEXP r_llama_embed_batch(SEXP r_ctx, SEXP r_texts) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    bool embedding = ctx_is_embedding(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_texts = Rf_length(r_texts);

    if (n_texts == 0) {
        SEXP r_mat = PROTECT(Rf_allocMatrix(REALSXP, 0, n_embd));
        UNPROTECT(1);
        return r_mat;
    }

    // tokenize all texts
    std::vector<std::vector<llama_token>> all_tokens(n_texts);
    int total_tokens = 0;
    for (int s = 0; s < n_texts; s++) {
        const char * text = CHAR(STRING_ELT(r_texts, s));
        if (tokenize_text(vocab, text, all_tokens[s]) < 0)
            llamar_error("llamaR: tokenization failed for text %d", s + 1);
        total_tokens += (int) all_tokens[s].size();
    }

    SEXP r_mat = PROTECT(Rf_allocMatrix(REALSXP, n_texts, n_embd));
    double * mat_ptr = REAL(r_mat);

    if (embedding) {
        // --- pooled batch: one decode for all texts ---
        struct llama_batch batch = llama_batch_init(total_tokens, 0, n_texts);
        int pos = 0;
        for (int s = 0; s < n_texts; s++) {
            for (int t = 0; t < (int) all_tokens[s].size(); t++) {
                batch.token[pos]      = all_tokens[s][t];
                batch.pos[pos]        = (llama_pos) t;
                batch.n_seq_id[pos]   = 1;
                batch.seq_id[pos][0]  = (llama_seq_id) s;
                batch.logits[pos]     = (t == (int) all_tokens[s].size() - 1) ? 1 : 0;
                pos++;
            }
        }
        batch.n_tokens = total_tokens;

        llama_memory_clear(llama_get_memory(ctx), true);
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            UNPROTECT(1);
            llamar_error("llamaR: batch embedding decode failed (code %d)", ret);
        }
        llama_synchronize(ctx);

        for (int s = 0; s < n_texts; s++) {
            float * emb = llama_get_embeddings_seq(ctx, (llama_seq_id) s);
            if (!emb) {
                UNPROTECT(1);
                llamar_error("llamaR: pooled embeddings NULL for seq %d", s);
            }
            for (int j = 0; j < n_embd; j++)
                mat_ptr[s + j * n_texts] = (double) emb[j];
        }
    } else {
        // --- sequential: one decode per text, last-token embedding ---
        // embed_single() reports failures by throwing, so the embeddings flag
        // has to be restored on that path too. The PROTECT of r_mat is left
        // alone: the throw reaches the boundary, whose Rf_error() longjmps,
        // and R restores the protection stack itself. Unprotecting by hand
        // here would add an exit at a different depth for rchk to trip over.
        llama_set_embeddings(ctx, true);
        std::vector<float> tmp(n_embd);
        try {
            for (int s = 0; s < n_texts; s++) {
                const char * text = CHAR(STRING_ELT(r_texts, s));
                embed_single(ctx, vocab, text, tmp.data(), n_embd, s);
                for (int j = 0; j < n_embd; j++)
                    mat_ptr[s + j * n_texts] = (double) tmp[j];
            }
        } catch (...) {
            llama_set_embeddings(ctx, false);
            throw;
        }
        llama_set_embeddings(ctx, false);
    }

    UNPROTECT(1);
    return r_mat;
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Chat templates
// ============================================================

extern "C" SEXP r_llama_chat_template(SEXP r_model, SEXP r_name) {
    llama_model * model = llamar_model_arg(r_model);

    const char * name = Rf_isNull(r_name) ? NULL : CHAR(STRING_ELT(r_name, 0));
    const char * tmpl = llama_model_chat_template(model, name);

    if (!tmpl) {
        return R_NilValue;
    }
    return Rf_mkString(tmpl);
}

extern "C" SEXP r_llama_chat_apply_template(SEXP r_tmpl, SEXP r_messages, SEXP r_add_ass) {
    LLAMAR_ENTRYPOINT_BEGIN
    const char * tmpl = Rf_isNull(r_tmpl) ? NULL : CHAR(STRING_ELT(r_tmpl, 0));
    bool add_ass = LOGICAL(r_add_ass)[0];

    // r_messages is a list of lists with $role and $content
    int n_msg = Rf_length(r_messages);
    std::vector<llama_chat_message> messages(n_msg);
    std::vector<std::string> roles(n_msg);
    std::vector<std::string> contents(n_msg);

    // Cache the symbols so Rf_install (an allocating call) runs once up front,
    // not inside the loop where r_role/r_content would be live across it.
    SEXP sym_role = Rf_install("role");
    SEXP sym_content = Rf_install("content");

    for (int i = 0; i < n_msg; i++) {
        SEXP msg = VECTOR_ELT(r_messages, i);

        // Resolve role and content independently and extract each string
        // immediately, so no SEXP is held live across another allocating call
        // (PROTECT-wise rchk-clean). r_role/r_content are scoped per branch.
        {
            SEXP r_role = Rf_getAttrib(msg, sym_role);
            if (Rf_isNull(r_role)) r_role = VECTOR_ELT(msg, 0);
            roles[i] = CHAR(STRING_ELT(r_role, 0));
        }
        {
            SEXP r_content = Rf_getAttrib(msg, sym_content);
            if (Rf_isNull(r_content)) r_content = VECTOR_ELT(msg, 1);
            contents[i] = CHAR(STRING_ELT(r_content, 0));
        }

        messages[i].role = roles[i].c_str();
        messages[i].content = contents[i].c_str();
    }

    // First call to get required size
    int size = llama_chat_apply_template(tmpl, messages.data(), n_msg, add_ass, NULL, 0);
    if (size < 0) {
        llamar_error("llamaR: failed to apply chat template");
    }

    std::vector<char> buf(size + 1);
    int actual = llama_chat_apply_template(tmpl, messages.data(), n_msg, add_ass, buf.data(), buf.size());
    if (actual < 0) {
        llamar_error("llamaR: failed to apply chat template");
    }
    buf[actual] = '\0';

    return Rf_mkString(buf.data());
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// LoRA adapters
// ============================================================

static void lora_finalizer(SEXP x) {
    // LoRA adapters are freed with the model, so we don't free here
    // Just clear the pointer
    R_SetExternalPtrAddr(x, NULL);
}

extern "C" SEXP r_llama_lora_load(SEXP r_model, SEXP r_path) {
    llama_model * model = llamar_model_arg(r_model);

    const char * path = CHAR(STRING_ELT(r_path, 0));

    llama_adapter_lora * adapter = llama_adapter_lora_init(model, path);
    if (!adapter) {
        Rf_error("llamaR: failed to load LoRA adapter from '%s'", path);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(adapter, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, lora_finalizer);
    UNPROTECT(1);
    return result;
}

// [llamaR] master replaced the per-adapter set/rm/clear API with a single
// llama_set_adapters_lora(ctx, adapters**, n, scales*) that overwrites the whole
// active set. To keep our documented per-adapter contract (apply many, remove a
// specific one keeping others, clear all) we track the active adapter->scale set
// per context here and rebuild the array on every change.
static std::map<llama_context *, std::map<llama_adapter_lora *, float>> g_lora_active;

static void llamar_lora_sync(llama_context * ctx) {
    auto & active = g_lora_active[ctx];
    if (active.empty()) {
        llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        return;
    }
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> scales;
    adapters.reserve(active.size());
    scales.reserve(active.size());
    for (auto & kv : active) {
        adapters.push_back(kv.first);
        scales.push_back(kv.second);
    }
    llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data());
}

extern "C" SEXP r_llama_lora_apply(SEXP r_ctx, SEXP r_adapter, SEXP r_scale) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    llama_adapter_lora * adapter = llamar_lora_arg_throw(r_adapter);

    float scale = (float) REAL(r_scale)[0];

    g_lora_active[ctx][adapter] = scale;
    llamar_lora_sync(ctx);

    return R_NilValue;
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_lora_remove(SEXP r_ctx, SEXP r_adapter) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    llama_adapter_lora * adapter = llamar_lora_arg_throw(r_adapter);

    auto & active = g_lora_active[ctx];
    auto it = active.find(adapter);
    if (it == active.end()) {
        return Rf_ScalarInteger(-1);  // adapter was not applied
    }
    active.erase(it);
    llamar_lora_sync(ctx);
    return Rf_ScalarInteger(0);
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_lora_clear(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    g_lora_active.erase(ctx);
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
    return R_NilValue;
}

// All GGUF metadata of a LoRA adapter, as a named character vector. Mirrors
// r_llama_model_meta() but reads from the adapter rather than the model.
extern "C" SEXP r_llama_lora_meta(SEXP r_adapter) {
    llama_adapter_lora * adapter = llamar_lora_arg(r_adapter);

    int32_t count = llama_adapter_meta_count(adapter);
    if (count < 0) count = 0;

    SEXP values = PROTECT(Rf_allocVector(STRSXP, count));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, count));

    char buf[1024];
    for (int32_t i = 0; i < count; i++) {
        int32_t klen = llama_adapter_meta_key_by_index(adapter, i, buf, sizeof(buf));
        SET_STRING_ELT(names, i, klen > 0 ? Rf_mkChar(buf) : NA_STRING);

        int32_t vlen = llama_adapter_meta_val_str_by_index(adapter, i, buf, sizeof(buf));
        SET_STRING_ELT(values, i, vlen > 0 ? Rf_mkChar(buf) : NA_STRING);
    }
    Rf_setAttrib(values, R_NamesSymbol, names);

    UNPROTECT(2);
    return values;
}

extern "C" SEXP r_llama_lora_meta_val(SEXP r_adapter, SEXP r_key) {
    llama_adapter_lora * adapter = llamar_lora_arg(r_adapter);

    const char * key = CHAR(STRING_ELT(r_key, 0));

    char buf[1024];
    int32_t len = llama_adapter_meta_val_str(adapter, key, buf, sizeof(buf));
    if (len < 0) return R_NilValue;   // key absent
    return Rf_mkString(buf);
}

// Invocation tokens of an activated LoRA (aLoRA). A plain LoRA has none, in
// which case this returns NULL.
extern "C" SEXP r_llama_lora_alora_invocation_tokens(SEXP r_adapter) {
    llama_adapter_lora * adapter = llamar_lora_arg(r_adapter);

    uint64_t n = llama_adapter_get_alora_n_invocation_tokens(adapter);
    if (n == 0) return R_NilValue;

    const llama_token * toks = llama_adapter_get_alora_invocation_tokens(adapter);
    if (!toks) return R_NilValue;

    SEXP out = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t) n));
    for (uint64_t i = 0; i < n; i++) {
        INTEGER(out)[i] = (int) toks[i];
    }
    UNPROTECT(1);
    return out;
}

// Control vector: a per-layer steering direction added to the residual stream.
// Passing NULL data clears any vector currently applied to the context.
extern "C" SEXP r_llama_apply_adapter_cvec(SEXP r_ctx, SEXP r_data, SEXP r_n_embd,
                                           SEXP r_il_start, SEXP r_il_end) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    int32_t n_embd    = INTEGER(r_n_embd)[0];
    int32_t il_start  = INTEGER(r_il_start)[0];
    int32_t il_end    = INTEGER(r_il_end)[0];

    // NB: the name is llama_set_adapter_cvec, matching llama-context.cpp.
    // llama.h used to declare it as llama_apply_adapter_cvec, which meant the
    // definition was never seen as extern "C" and linked under a mangled name.
    if (Rf_isNull(r_data)) {
        if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, il_start, il_end) != 0) {
            llamar_error("llamaR: failed to clear the control vector");
        }
        return R_NilValue;
    }

    R_xlen_t len = Rf_xlength(r_data);
    // llama.cpp takes float; R gives double, so convert into a temporary.
    std::vector<float> data((size_t) len);
    for (R_xlen_t i = 0; i < len; i++) {
        data[(size_t) i] = (float) REAL(r_data)[i];
    }

    if (llama_set_adapter_cvec(ctx, data.data(), data.size(), n_embd, il_start, il_end) != 0) {
        llamar_error("llamaR: failed to apply the control vector "
                     "(check n_embd and the layer range against the model)");
    }
    return R_NilValue;
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Extended Model Info
// ============================================================

extern "C" SEXP r_llama_model_size(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarReal((double) llama_model_size(model));
}

extern "C" SEXP r_llama_model_n_params(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarReal((double) llama_model_n_params(model));
}

extern "C" SEXP r_llama_model_has_encoder(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_model_has_encoder(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_has_decoder(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_model_has_decoder(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_is_recurrent(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_model_is_recurrent(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_is_hybrid(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_model_is_hybrid(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_is_diffusion(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_model_is_diffusion(model) ? TRUE : FALSE);
}

// Input embedding width. Differs from llama_model_n_embd() on models whose
// input projection is wider than the residual stream (e.g. some VL models).
extern "C" SEXP r_llama_model_n_embd_inp(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarInteger((int) llama_model_n_embd_inp(model));
}

extern "C" SEXP r_llama_model_n_embd_out(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarInteger((int) llama_model_n_embd_out(model));
}

// Sliding-window attention span; 0 when the model attends over the full context.
extern "C" SEXP r_llama_model_n_swa(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarInteger((int) llama_model_n_swa(model));
}

extern "C" SEXP r_llama_model_rope_type(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    // Returned as a string: the enum values are non-contiguous (they alias the
    // GGML_ROPE_TYPE_* bit flags), so a bare integer would be hard to read.
    const char * name;
    switch (llama_model_rope_type(model)) {
        case LLAMA_ROPE_TYPE_NONE:   name = "none";   break;
        case LLAMA_ROPE_TYPE_NORM:   name = "norm";   break;
        case LLAMA_ROPE_TYPE_NEOX:   name = "neox";   break;
        case LLAMA_ROPE_TYPE_MROPE:  name = "mrope";  break;
        case LLAMA_ROPE_TYPE_IMROPE: name = "imrope"; break;
        case LLAMA_ROPE_TYPE_VISION: name = "vision"; break;
        default:                     name = "unknown"; break;
    }
    return Rf_mkString(name);
}

extern "C" SEXP r_llama_model_rope_freq_scale_train(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarReal((double) llama_model_rope_freq_scale_train(model));
}

// Number of classifier outputs (0 unless the model is a classifier/reranker).
extern "C" SEXP r_llama_model_n_cls_out(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarInteger((int) llama_model_n_cls_out(model));
}

// Labels for the classifier outputs, or NULL when the model provides none.
extern "C" SEXP r_llama_model_cls_labels(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);

    uint32_t n = llama_model_n_cls_out(model);
    if (n == 0) return R_NilValue;

    SEXP out = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
    bool any = false;
    for (uint32_t i = 0; i < n; i++) {
        const char * lbl = llama_model_cls_label(model, i);
        if (lbl) {
            SET_STRING_ELT(out, i, Rf_mkChar(lbl));
            any = true;
        } else {
            SET_STRING_ELT(out, i, NA_STRING);
        }
    }
    UNPROTECT(1);
    return any ? out : R_NilValue;
}

extern "C" SEXP r_llama_model_decoder_start_token(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    llama_token tok = llama_model_decoder_start_token(model);
    // -1 means the model does not define one; report that as NA rather than
    // handing back a token id that cannot be detokenized.
    if (tok < 0) return Rf_ScalarInteger(NA_INTEGER);
    return Rf_ScalarInteger((int) tok);
}

// GGUF keys holding the sampling parameters recommended by the model author,
// e.g. "sampling.top_k". Returned as a named character vector so callers can
// look them up with llama_model_meta_val().
extern "C" SEXP r_llama_model_sampling_meta_keys(void) {
    static const struct { enum llama_model_meta_key key; const char * name; } keys[] = {
        { LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,        "sequence"        },
        { LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,           "top_k"           },
        { LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,           "top_p"           },
        { LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,           "min_p"           },
        { LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY, "xtc_probability" },
        { LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,   "xtc_threshold"   },
        { LLAMA_MODEL_META_KEY_SAMPLING_TEMP,            "temp"            },
        { LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,  "penalty_last_n"  },
        { LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,  "penalty_repeat"  },
        { LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,        "mirostat"        },
        { LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,    "mirostat_tau"    },
        { LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,    "mirostat_eta"    },
    };
    const int n = (int) (sizeof(keys) / sizeof(keys[0]));

    SEXP out   = PROTECT(Rf_allocVector(STRSXP, n));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
    for (int i = 0; i < n; i++) {
        const char * s = llama_model_meta_key_str(keys[i].key);
        SET_STRING_ELT(out, i, s ? Rf_mkChar(s) : NA_STRING);
        SET_STRING_ELT(names, i, Rf_mkChar(keys[i].name));
    }
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2);
    return out;
}

extern "C" SEXP r_llama_model_meta(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);

    int32_t count = llama_model_meta_count(model);

    SEXP names  = PROTECT(Rf_allocVector(STRSXP, count));
    SEXP values = PROTECT(Rf_allocVector(STRSXP, count));

    char buf[512];
    for (int32_t i = 0; i < count; i++) {
        int32_t klen = llama_model_meta_key_by_index(model, i, buf, sizeof(buf));
        if (klen > 0) {
            buf[klen] = '\0';
            SET_STRING_ELT(names, i, Rf_mkChar(buf));
        } else {
            SET_STRING_ELT(names, i, Rf_mkChar(""));
        }

        int32_t vlen = llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
        if (vlen > 0) {
            buf[(vlen < (int32_t)sizeof(buf) - 1) ? vlen : (int32_t)sizeof(buf) - 1] = '\0';
            SET_STRING_ELT(values, i, Rf_mkChar(buf));
        } else {
            SET_STRING_ELT(values, i, Rf_mkChar(""));
        }
    }

    Rf_setAttrib(values, R_NamesSymbol, names);
    UNPROTECT(2);
    return values;
}

extern "C" SEXP r_llama_model_meta_val(SEXP r_model, SEXP r_key) {
    llama_model * model = llamar_model_arg(r_model);

    const char * key = CHAR(STRING_ELT(r_key, 0));
    char buf[512];
    int32_t len = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (len < 0) return R_NilValue;
    buf[(len < (int32_t)sizeof(buf) - 1) ? len : (int32_t)sizeof(buf) - 1] = '\0';
    return Rf_mkString(buf);
}

// ============================================================
// Vocabulary Info
// ============================================================

extern "C" SEXP r_llama_vocab_info(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    SEXP result = PROTECT(Rf_allocVector(INTSXP, 11));
    INTEGER(result)[0]  = llama_vocab_bos(vocab);
    INTEGER(result)[1]  = llama_vocab_eos(vocab);
    INTEGER(result)[2]  = llama_vocab_eot(vocab);
    INTEGER(result)[3]  = llama_vocab_sep(vocab);
    INTEGER(result)[4]  = llama_vocab_nl(vocab);
    INTEGER(result)[5]  = llama_vocab_pad(vocab);
    INTEGER(result)[6]  = llama_vocab_fim_pre(vocab);
    INTEGER(result)[7]  = llama_vocab_fim_suf(vocab);
    INTEGER(result)[8]  = llama_vocab_fim_mid(vocab);
    INTEGER(result)[9]  = llama_vocab_fim_rep(vocab);
    INTEGER(result)[10] = llama_vocab_fim_sep(vocab);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 11));
    SET_STRING_ELT(names, 0,  Rf_mkChar("bos"));
    SET_STRING_ELT(names, 1,  Rf_mkChar("eos"));
    SET_STRING_ELT(names, 2,  Rf_mkChar("eot"));
    SET_STRING_ELT(names, 3,  Rf_mkChar("sep"));
    SET_STRING_ELT(names, 4,  Rf_mkChar("nl"));
    SET_STRING_ELT(names, 5,  Rf_mkChar("pad"));
    SET_STRING_ELT(names, 6,  Rf_mkChar("fim_pre"));
    SET_STRING_ELT(names, 7,  Rf_mkChar("fim_suf"));
    SET_STRING_ELT(names, 8,  Rf_mkChar("fim_mid"));
    SET_STRING_ELT(names, 9,  Rf_mkChar("fim_rep"));
    SET_STRING_ELT(names, 10, Rf_mkChar("fim_sep"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_vocab_type(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int vt = (int) llama_vocab_type(vocab);
    const char * name;
    switch (vt) {
        case 0: name = "none";   break;
        case 1: name = "spm";    break;
        case 2: name = "bpe";    break;
        case 3: name = "wpm";    break;
        case 4: name = "ugm";    break;
        case 5: name = "rwkv";   break;
        case 6: name = "plamo2"; break;
        default: name = "unknown"; break;
    }
    return Rf_mkString(name);
}

extern "C" SEXP r_llama_vocab_is_eog(SEXP r_model, SEXP r_token) {
    llama_model * model = llamar_model_arg(r_model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarLogical(llama_vocab_is_eog(vocab, token) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_is_control(SEXP r_model, SEXP r_token) {
    llama_model * model = llamar_model_arg(r_model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarLogical(llama_vocab_is_control(vocab, token) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_get_text(SEXP r_model, SEXP r_token) {
    llama_model * model = llamar_model_arg(r_model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    const char * text = llama_vocab_get_text(vocab, token);
    if (!text) return R_NilValue;
    return Rf_mkString(text);
}

extern "C" SEXP r_llama_vocab_get_score(SEXP r_model, SEXP r_token) {
    llama_model * model = llamar_model_arg(r_model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarReal((double) llama_vocab_get_score(vocab, token));
}

// Token attributes are a bit mask; returned as a character vector of the set
// flags, which is far more usable from R than the raw integer.
extern "C" SEXP r_llama_vocab_get_attr(SEXP r_model, SEXP r_token) {
    llama_model * model = llamar_model_arg(r_model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];

    enum llama_token_attr attr = llama_vocab_get_attr(vocab, token);
    if (attr == LLAMA_TOKEN_ATTR_UNDEFINED) return Rf_allocVector(STRSXP, 0);

    static const struct { enum llama_token_attr flag; const char * name; } flags[] = {
        { LLAMA_TOKEN_ATTR_UNKNOWN,      "unknown"      },
        { LLAMA_TOKEN_ATTR_UNUSED,       "unused"       },
        { LLAMA_TOKEN_ATTR_NORMAL,       "normal"       },
        { LLAMA_TOKEN_ATTR_CONTROL,      "control"      },
        { LLAMA_TOKEN_ATTR_USER_DEFINED, "user_defined" },
        { LLAMA_TOKEN_ATTR_BYTE,         "byte"         },
        { LLAMA_TOKEN_ATTR_NORMALIZED,   "normalized"   },
        { LLAMA_TOKEN_ATTR_LSTRIP,       "lstrip"       },
        { LLAMA_TOKEN_ATTR_RSTRIP,       "rstrip"       },
        { LLAMA_TOKEN_ATTR_SINGLE_WORD,  "single_word"  },
    };
    const int n_flags = (int) (sizeof(flags) / sizeof(flags[0]));

    int n_set = 0;
    for (int i = 0; i < n_flags; i++) {
        if (attr & flags[i].flag) n_set++;
    }

    SEXP out = PROTECT(Rf_allocVector(STRSXP, n_set));
    int j = 0;
    for (int i = 0; i < n_flags; i++) {
        if (attr & flags[i].flag) SET_STRING_ELT(out, j++, Rf_mkChar(flags[i].name));
    }
    UNPROTECT(1);
    return out;
}

extern "C" SEXP r_llama_vocab_get_add_bos(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_vocab_get_add_bos(llama_model_get_vocab(model)) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_get_add_eos(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_vocab_get_add_eos(llama_model_get_vocab(model)) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_get_add_sep(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    return Rf_ScalarLogical(llama_vocab_get_add_sep(llama_model_get_vocab(model)) ? TRUE : FALSE);
}

// Special-token ids not covered by llama_vocab_info(). Absent tokens come back
// as NA rather than -1.
extern "C" SEXP r_llama_vocab_mask(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    llama_token tok = llama_vocab_mask(llama_model_get_vocab(model));
    return Rf_ScalarInteger(tok < 0 ? NA_INTEGER : (int) tok);
}

extern "C" SEXP r_llama_vocab_fim_pad(SEXP r_model) {
    llama_model * model = llamar_model_arg(r_model);
    llama_token tok = llama_vocab_fim_pad(llama_model_get_vocab(model));
    return Rf_ScalarInteger(tok < 0 ? NA_INTEGER : (int) tok);
}

// --- hardware / build limits (no model or context needed) ------------------

extern "C" SEXP r_llama_max_parallel_sequences(void) {
    return Rf_ScalarInteger((int) llama_max_parallel_sequences());
}

extern "C" SEXP r_llama_max_tensor_buft_overrides(void) {
    return Rf_ScalarInteger((int) llama_max_tensor_buft_overrides());
}

// ============================================================
// Context Config
// ============================================================

extern "C" SEXP r_llama_set_n_threads(SEXP r_ctx, SEXP r_n_threads, SEXP r_n_threads_batch) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    int32_t n_threads       = INTEGER(r_n_threads)[0];
    int32_t n_threads_batch = INTEGER(r_n_threads_batch)[0];
    llama_set_n_threads(ctx, n_threads, n_threads_batch);
    return R_NilValue;
}

extern "C" SEXP r_llama_set_causal_attn(SEXP r_ctx, SEXP r_causal) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    bool causal = LOGICAL(r_causal)[0] != 0;
    llama_set_causal_attn(ctx, causal);
    return R_NilValue;
}

extern "C" SEXP r_llama_get_model(SEXP r_ctx) {
    // The model R object is stored as the "prot" of the context external pointer.
    // Return it directly — same R externalptr that was passed to llama_new_context().
    (void) llamar_ctx_arg(r_ctx);   // validate the handle, discard the address
    return R_ExternalPtrProtected(r_ctx);
}

extern "C" SEXP r_llama_set_warmup(SEXP r_ctx, SEXP r_warmup) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    llama_set_warmup(ctx, LOGICAL(r_warmup)[0] != 0);
    return R_NilValue;
}

// Global abort callback state (one slot — sufficient for single-context use)
static SEXP s_abort_callback = R_NilValue;

static bool r_abort_callback(void * data) {
    (void) data;
    if (s_abort_callback == R_NilValue) return false;
    SEXP call   = PROTECT(Rf_lang1(s_abort_callback));
    int  error  = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &error);
    UNPROTECT(1);
    if (error) return true;  // abort on R error
    if (TYPEOF(result) == LGLSXP && LENGTH(result) >= 1)
        return LOGICAL(result)[0] != 0;
    return false;
}

extern "C" SEXP r_llama_set_abort_callback(SEXP r_ctx, SEXP r_fn) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    // Validate before mutating any state, so an error leaves the old callback
    // intact rather than half-replaced.
    if (r_fn != R_NilValue && !Rf_isFunction(r_fn)) {
        Rf_error("llamaR: abort_callback must be a function or NULL");
    }

    // Release the previously preserved callback (if any) before replacing it,
    // otherwise it leaks on the precious list and stays alive forever.
    if (s_abort_callback != R_NilValue) {
        R_ReleaseObject(s_abort_callback);
        s_abort_callback = R_NilValue;
    }

    if (r_fn == R_NilValue) {
        llama_set_abort_callback(ctx, NULL, NULL);
    } else {
        s_abort_callback = r_fn;
        R_PreserveObject(s_abort_callback);
        llama_set_abort_callback(ctx, r_abort_callback, NULL);
    }
    return R_NilValue;
}

extern "C" SEXP r_llama_n_ctx(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger((int) llama_n_ctx(ctx));
}

extern "C" SEXP r_llama_n_ctx_seq(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger((int) llama_n_ctx_seq(ctx));
}

extern "C" SEXP r_llama_n_batch(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger((int) llama_n_batch(ctx));
}

extern "C" SEXP r_llama_n_ubatch(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger((int) llama_n_ubatch(ctx));
}

extern "C" SEXP r_llama_n_seq_max(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger((int) llama_n_seq_max(ctx));
}

// llama.cpp's own name for a flash-attention type ("auto" / "enabled" /
// "disabled"), for the enum values llama_new_context() accepts.
extern "C" SEXP r_llama_flash_attn_type_name(SEXP r_type) {
    const int t = INTEGER(r_type)[0];
    if (t == NA_INTEGER) Rf_error("llamaR: flash_attn type must not be NA");
    if (t < LLAMA_FLASH_ATTN_TYPE_AUTO || t > LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        Rf_error("llamaR: unknown flash_attn type %d", t);
    }
    const char * name = llama_flash_attn_type_name((enum llama_flash_attn_type) t);
    return Rf_mkString(name ? name : "");
}

// What a context actually resolved flash attention to. Asking for "auto" leaves
// the decision to llama.cpp, and llama.h exposes no getter for the outcome, so
// this reads the resolved value out of the context's cparams.
//
// NB: cparams.auto_fa is not "the choice was automatic" but "the choice is
// still pending": llama_context resolves auto during construction and clears
// the flag right after (llama-context.cpp), so on a live context it is always
// false and there is nothing to report from it.
extern "C" SEXP r_llama_context_flash_attn(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const llama_cparams & cp = ctx->get_cparams();

    SEXP out   = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));

    SET_STRING_ELT(names, 0, Rf_mkChar("enabled"));
    SET_STRING_ELT(names, 1, Rf_mkChar("type_name"));

    SET_VECTOR_ELT(out, 0, Rf_ScalarLogical(cp.flash_attn));
    SET_VECTOR_ELT(out, 1, Rf_mkString(llama_flash_attn_type_name(
        cp.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                      : LLAMA_FLASH_ATTN_TYPE_DISABLED)));

    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2);
    return out;
}

extern "C" SEXP r_llama_n_threads(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger(llama_n_threads(ctx));
}

extern "C" SEXP r_llama_n_threads_batch(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarInteger(llama_n_threads_batch(ctx));
}

extern "C" SEXP r_llama_pooling_type(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    int pt = (int) llama_pooling_type(ctx);
    const char * name;
    switch (pt) {
        case -1: name = "unspecified"; break;
        case  0: name = "none";        break;
        case  1: name = "mean";        break;
        case  2: name = "cls";         break;
        case  3: name = "last";        break;
        case  4: name = "rank";        break;
        default: name = "unknown";     break;
    }
    return Rf_mkString(name);
}

// ============================================================
// Memory / KV Cache
// ============================================================

extern "C" SEXP r_llama_memory_clear(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_rm(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_seq_id seq_id = INTEGER(r_seq_id)[0];
    llama_pos p0 = INTEGER(r_p0)[0];
    llama_pos p1 = INTEGER(r_p1)[0];

    bool ok = llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
    return Rf_ScalarLogical(ok ? TRUE : FALSE);
}

extern "C" SEXP r_llama_memory_seq_cp(SEXP r_ctx, SEXP r_seq_src, SEXP r_seq_dst, SEXP r_p0, SEXP r_p1) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_memory_seq_cp(llama_get_memory(ctx),
                        INTEGER(r_seq_src)[0], INTEGER(r_seq_dst)[0],
                        INTEGER(r_p0)[0], INTEGER(r_p1)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_keep(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_memory_seq_keep(llama_get_memory(ctx), INTEGER(r_seq_id)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_add(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1, SEXP r_delta) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_memory_seq_add(llama_get_memory(ctx),
                         INTEGER(r_seq_id)[0],
                         INTEGER(r_p0)[0], INTEGER(r_p1)[0],
                         INTEGER(r_delta)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_div(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1, SEXP r_d) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_memory_seq_div(llama_get_memory(ctx),
                         INTEGER(r_seq_id)[0],
                         INTEGER(r_p0)[0], INTEGER(r_p1)[0],
                         INTEGER(r_d)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_pos_range(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_seq_id seq_id = INTEGER(r_seq_id)[0];
    llama_memory_t mem = llama_get_memory(ctx);

    SEXP result = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(result)[0] = llama_memory_seq_pos_min(mem, seq_id);
    INTEGER(result)[1] = llama_memory_seq_pos_max(mem, seq_id);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("min"));
    SET_STRING_ELT(names, 1, Rf_mkChar("max"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_memory_can_shift(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarLogical(llama_memory_can_shift(llama_get_memory(ctx)) ? TRUE : FALSE);
}

// ============================================================
// State Save / Load
// ============================================================

extern "C" SEXP r_llama_state_save(SEXP r_ctx, SEXP r_path) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const char * path = CHAR(STRING_ELT(r_path, 0));
    bool ok = llama_state_save_file(ctx, path, NULL, 0);
    if (!ok) Rf_error("llamaR: failed to save state to '%s'", path);
    return Rf_ScalarLogical(TRUE);
}

extern "C" SEXP r_llama_state_load(SEXP r_ctx, SEXP r_path) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const char * path = CHAR(STRING_ELT(r_path, 0));
    size_t n_token_count = 0;
    bool ok = llama_state_load_file(ctx, path, NULL, 0, &n_token_count);
    if (!ok) Rf_error("llamaR: failed to load state from '%s'", path);
    return Rf_ScalarLogical(TRUE);
}

// --- state as raw bytes, whole context ------------------------------------

// The state is handed to R as a raw vector. llama_state_get_size() may
// overestimate, so the buffer is trimmed to the number of bytes actually
// written before being returned.
extern "C" SEXP r_llama_state_get_data(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    size_t size = llama_state_get_size(ctx);
    if (size == 0) Rf_error("llamaR: context reports a zero-sized state");

    SEXP buf = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) size));
    size_t written = llama_state_get_data(ctx, RAW(buf), size);
    if (written == 0) {
        UNPROTECT(1);
        Rf_error("llamaR: failed to copy context state");
    }

    if (written < size) {
        SEXP trimmed = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) written));
        memcpy(RAW(trimmed), RAW(buf), written);
        UNPROTECT(2);
        return trimmed;
    }

    UNPROTECT(1);
    return buf;
}

extern "C" SEXP r_llama_state_set_data(SEXP r_ctx, SEXP r_data) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    if (TYPEOF(r_data) != RAWSXP) Rf_error("llamaR: state data must be a raw vector");

    size_t size = (size_t) Rf_xlength(r_data);
    size_t read = llama_state_set_data(ctx, RAW(r_data), size);
    if (read == 0) {
        Rf_error("llamaR: failed to restore context state "
                 "(the data may be truncated or from an incompatible model)");
    }
    return Rf_ScalarReal((double) read);
}

// --- per-sequence state ----------------------------------------------------

// The _ext entry points take a flags word; passing 0 makes them behave exactly
// like the plain ones, so a single implementation covers both.
extern "C" SEXP r_llama_state_seq_get_size(SEXP r_ctx, SEXP r_seq_id, SEXP r_flags) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    llama_state_seq_flags flags = (llama_state_seq_flags) INTEGER(r_flags)[0];

    return Rf_ScalarReal((double) llama_state_seq_get_size_ext(ctx, seq_id, flags));
}

extern "C" SEXP r_llama_state_seq_get_data(SEXP r_ctx, SEXP r_seq_id, SEXP r_flags) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    llama_state_seq_flags flags = (llama_state_seq_flags) INTEGER(r_flags)[0];

    size_t size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
    if (size == 0) Rf_error("llamaR: sequence %d reports a zero-sized state", (int) seq_id);

    SEXP buf = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) size));
    size_t written = llama_state_seq_get_data_ext(ctx, RAW(buf), size, seq_id, flags);
    if (written == 0) {
        UNPROTECT(1);
        Rf_error("llamaR: failed to copy the state of sequence %d", (int) seq_id);
    }

    if (written < size) {
        SEXP trimmed = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) written));
        memcpy(RAW(trimmed), RAW(buf), written);
        UNPROTECT(2);
        return trimmed;
    }

    UNPROTECT(1);
    return buf;
}

extern "C" SEXP r_llama_state_seq_set_data(SEXP r_ctx, SEXP r_data, SEXP r_seq_id, SEXP r_flags) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    if (TYPEOF(r_data) != RAWSXP) Rf_error("llamaR: state data must be a raw vector");

    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    llama_state_seq_flags flags = (llama_state_seq_flags) INTEGER(r_flags)[0];

    size_t read = llama_state_seq_set_data_ext(ctx, RAW(r_data),
                                               (size_t) Rf_xlength(r_data),
                                               seq_id, flags);
    if (read == 0) {
        Rf_error("llamaR: failed to restore the state of sequence %d "
                 "(the data may be truncated or from an incompatible model)", (int) seq_id);
    }
    return Rf_ScalarReal((double) read);
}

// Sequence state on disk. The token list travels with the state so a reloaded
// sequence knows which prompt produced it.
extern "C" SEXP r_llama_state_seq_save_file(SEXP r_ctx, SEXP r_path, SEXP r_seq_id, SEXP r_tokens) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const char * path = CHAR(STRING_ELT(r_path, 0));
    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];

    std::vector<llama_token> tokens;
    if (!Rf_isNull(r_tokens)) {
        R_xlen_t n = Rf_xlength(r_tokens);
        tokens.reserve((size_t) n);
        for (R_xlen_t i = 0; i < n; i++) {
            tokens.push_back((llama_token) INTEGER(r_tokens)[i]);
        }
    }

    size_t written = llama_state_seq_save_file(ctx, path, seq_id,
                                               tokens.empty() ? NULL : tokens.data(),
                                               tokens.size());
    if (written == 0) {
        llamar_error("llamaR: failed to save the state of sequence %d to '%s'", (int) seq_id, path);
    }
    return Rf_ScalarReal((double) written);
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_state_seq_load_file(SEXP r_ctx, SEXP r_path, SEXP r_seq_id,
                                            SEXP r_n_token_capacity) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const char * path = CHAR(STRING_ELT(r_path, 0));
    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    size_t capacity = (size_t) INTEGER(r_n_token_capacity)[0];

    std::vector<llama_token> tokens(capacity);
    size_t n_token_count = 0;

    size_t read = llama_state_seq_load_file(ctx, path, seq_id,
                                            capacity ? tokens.data() : NULL,
                                            capacity, &n_token_count);
    if (read == 0) {
        llamar_error("llamaR: failed to load the state of sequence %d from '%s' "
                     "(the file may be missing, truncated, or from an incompatible model)",
                     (int) seq_id, path);
    }

    SEXP r_toks = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t) n_token_count));
    for (size_t i = 0; i < n_token_count; i++) {
        INTEGER(r_toks)[i] = (int) tokens[i];
    }

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal((double) read));
    SET_VECTOR_ELT(result, 1, r_toks);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("n_bytes"));
    SET_STRING_ELT(names, 1, Rf_mkChar("tokens"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(3);
    return result;
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_state_get_size(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    return Rf_ScalarReal((double) llama_state_get_size(ctx));
}

extern "C" SEXP r_llama_synchronize(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    llama_synchronize(ctx);
    return R_NilValue;
}

// ============================================================
// Logits
// ============================================================

extern "C" SEXP r_llama_get_logits(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    float * logits = llama_get_logits(ctx);
    if (!logits) Rf_error("llamaR: logits are NULL (no decode has been performed)");

    SEXP result = PROTECT(Rf_allocVector(REALSXP, n_vocab));
    for (int i = 0; i < n_vocab; i++) {
        REAL(result)[i] = (double) logits[i];
    }
    UNPROTECT(1);
    return result;
}

extern "C" SEXP r_llama_get_logits_ith(SEXP r_ctx, SEXP r_i) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    int32_t i = INTEGER(r_i)[0];
    float * logits = llama_get_logits_ith(ctx, i);
    if (!logits) Rf_error("llamaR: logits_ith is NULL for i=%d", i);

    SEXP result = PROTECT(Rf_allocVector(REALSXP, n_vocab));
    for (int k = 0; k < n_vocab; k++) {
        REAL(result)[k] = (double) logits[k];
    }
    UNPROTECT(1);
    return result;
}

// ============================================================
// Performance
// ============================================================

extern "C" SEXP r_llama_perf_context(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    struct llama_perf_context_data perf = llama_perf_context(ctx);

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 6));
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(perf.t_load_ms));
    SET_VECTOR_ELT(result, 1, Rf_ScalarReal(perf.t_p_eval_ms));
    SET_VECTOR_ELT(result, 2, Rf_ScalarReal(perf.t_eval_ms));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(perf.n_p_eval));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(perf.n_eval));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(perf.n_reused));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 6));
    SET_STRING_ELT(names, 0, Rf_mkChar("t_load_ms"));
    SET_STRING_ELT(names, 1, Rf_mkChar("t_p_eval_ms"));
    SET_STRING_ELT(names, 2, Rf_mkChar("t_eval_ms"));
    SET_STRING_ELT(names, 3, Rf_mkChar("n_p_eval"));
    SET_STRING_ELT(names, 4, Rf_mkChar("n_eval"));
    SET_STRING_ELT(names, 5, Rf_mkChar("n_reused"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_perf_context_reset(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    llama_perf_context_reset(ctx);
    return R_NilValue;
}

extern "C" SEXP r_llama_perf_context_print(SEXP r_ctx) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);
    llama_perf_context_print(ctx);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_breakdown_print(SEXP r_ctx) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);
    // [llamaR] upstream master removed llama_memory_breakdown_print(); it now
    // exposes llama_get_memory_breakdown() returning a per-buffer-type map, and
    // leaves printing to the caller. We aggregate and print via Rprintf.
    const llama_memory_breakdown bd = llama_get_memory_breakdown(ctx);
    size_t tot_model = 0, tot_ctx = 0, tot_compute = 0;
    for (const auto & kv : bd) {
        const ggml_backend_buffer_type_t buft = kv.first;
        const llama_memory_breakdown_data & d = kv.second;
        Rprintf("%-20s: model %8.2f MiB, context %8.2f MiB, compute %8.2f MiB\n",
                ggml_backend_buft_name(buft),
                d.model   / (1024.0 * 1024.0),
                d.context / (1024.0 * 1024.0),
                d.compute / (1024.0 * 1024.0));
        tot_model   += d.model;
        tot_ctx     += d.context;
        tot_compute += d.compute;
    }
    Rprintf("%-20s: model %8.2f MiB, context %8.2f MiB, compute %8.2f MiB\n",
            "total",
            tot_model   / (1024.0 * 1024.0),
            tot_ctx     / (1024.0 * 1024.0),
            tot_compute / (1024.0 * 1024.0));
    return R_NilValue;
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_system_info(void) {
    ensure_backend_init();
    return Rf_mkString(llama_print_system_info());
}

// ============================================================
// Hardware Support
// ============================================================

extern "C" SEXP r_llama_supports_mmap(void) {
    return Rf_ScalarLogical(llama_supports_mmap() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_supports_mlock(void) {
    return Rf_ScalarLogical(llama_supports_mlock() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_supports_rpc(void) {
    return Rf_ScalarLogical(llama_supports_rpc() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_max_devices(void) {
    return Rf_ScalarInteger((int) llama_max_devices());
}

// ============================================================
// Chat: builtin templates
// ============================================================

extern "C" SEXP r_llama_chat_builtin_templates(void) {
    LLAMAR_ENTRYPOINT_BEGIN
    // First call to get count
    int32_t count = llama_chat_builtin_templates(NULL, 0);
    if (count <= 0) {
        return Rf_allocVector(STRSXP, 0);
    }

    std::vector<const char *> names(count);
    llama_chat_builtin_templates(names.data(), count);

    SEXP result = PROTECT(Rf_allocVector(STRSXP, count));
    for (int32_t i = 0; i < count; i++) {
        SET_STRING_ELT(result, i, Rf_mkChar(names[i] ? names[i] : ""));
    }
    UNPROTECT(1);
    return result;
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Batch: init / free
// ============================================================

extern "C" SEXP r_llama_batch_init(SEXP r_n_tokens, SEXP r_embd, SEXP r_n_seq_max) {
    int32_t n_tokens  = INTEGER(r_n_tokens)[0];
    int32_t embd      = INTEGER(r_embd)[0];
    int32_t n_seq_max = INTEGER(r_n_seq_max)[0];

    struct llama_batch * batch = new llama_batch;
    *batch = llama_batch_init(n_tokens, embd, n_seq_max);

    SEXP tag = PROTECT(Rf_mkString("llama_batch"));
    SEXP result = PROTECT(R_MakeExternalPtr(batch, tag, R_NilValue));
    R_RegisterCFinalizer(result, [](SEXP x) {
        llama_batch * b = (llama_batch *) R_ExternalPtrAddr(x);
        if (b) {
            llama_batch_free(*b);
            delete b;
            R_SetExternalPtrAddr(x, NULL);
        }
    });
    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_batch_free(SEXP r_batch) {
    llama_batch * b = (llama_batch *) llamar_ptr_addr_or_null(r_batch, "batch");
    if (b) {
        llama_batch_free(*b);
        delete b;
        R_SetExternalPtrAddr(r_batch, NULL);
    }
    return R_NilValue;
}

// ============================================================
// Encode (encoder-decoder models)
// ============================================================

extern "C" SEXP r_llama_encode(SEXP r_ctx, SEXP r_tokens) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    int n_tokens = LENGTH(r_tokens);
    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = INTEGER(r_tokens)[i];
    }

    struct llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    int32_t ret = llama_encode(ctx, batch);
    if (ret < 0) llamar_error("llamaR: llama_encode failed (code %d)", ret);

    return Rf_ScalarInteger(ret);
    LLAMAR_ENTRYPOINT_END
}

// ============================================================
// Token to piece
// ============================================================

extern "C" SEXP r_llama_token_to_piece(SEXP r_ctx, SEXP r_token, SEXP r_special) {
    llama_context * ctx = llamar_ctx_arg(r_ctx);

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_token token = INTEGER(r_token)[0];
    bool special = LOGICAL(r_special)[0] != 0;

    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, special);
    if (n < 0) Rf_error("llamaR: llama_token_to_piece failed (buffer too small)");
    buf[n] = '\0';

    return Rf_mkString(buf);
}

// ============================================================
// Sampler chain API: build a chain by hand, inspect it, take it apart
// ============================================================
//
// Ownership is the whole difficulty here, because llama.cpp moves it around:
//   - llama_sampler_chain_add    : the chain takes ownership of the sampler
//   - llama_sampler_chain_remove : ownership comes back to the caller
//   - llama_sampler_chain_get    : borrows; freeing the result is a bug
//
// Each R handle therefore carries an `owned` flag, and the finalizer frees the
// sampler only when it still owns it. That alone would still let R hold a
// handle to a sampler its chain has already freed, so a handle added to a chain
// also records that chain plus the chain's generation counter at the time. The
// chain bumps its generation when it is freed, which makes every handle still
// pointing at it fail an O(1) check and raise an R error instead of touching
// freed memory. chain_remove severs the link entirely (parent = NULL,
// generation = 0), so a returned-to-R sampler cannot be invalidated later by a
// counter that happens to wrap around.

// A chain's liveness, kept in its own allocation so that handles to the
// samplers inside it can outlive the chain handle itself. Children hold a
// shared_ptr to this, never a pointer to the chain handle, which would dangle
// the moment the chain's finalizer ran.
struct llamar_chain_life {
    bool     alive      = true;
    uint64_t generation = 1;
};

struct llamar_sampler_handle {
    llama_sampler * smpl   = NULL;
    bool            owned  = true;   // false once a chain has taken it over
    bool            is_chain = false;

    // For a chain: its own liveness record. For a sampler owned by a chain: the
    // liveness record of that chain, plus the generation seen when it joined.
    std::shared_ptr<llamar_chain_life> life;
    std::shared_ptr<llamar_chain_life> parent_life;
    uint64_t                           parent_generation = 0;
};

static void llamar_sampler_handle_finalizer(SEXP x) {
    llamar_sampler_handle * h = (llamar_sampler_handle *) R_ExternalPtrAddr(x);
    if (!h) return;
    // Only free what we still own; a chain that took this sampler over will
    // free it itself. Freeing a chain invalidates every handle inside it.
    if (h->owned && h->smpl) {
        if (h->is_chain && h->life) {
            h->life->alive = false;
            h->life->generation++;
        }
        llama_sampler_free(h->smpl);
    }
    delete h;
    R_SetExternalPtrAddr(x, NULL);
}

static SEXP llamar_new_sampler_handle(llama_sampler * smpl, bool owned, bool is_chain) {
    llamar_sampler_handle * h = new llamar_sampler_handle();
    h->smpl     = smpl;
    h->owned    = owned;
    h->is_chain = is_chain;
    if (is_chain) h->life = std::make_shared<llamar_chain_life>();

    SEXP tag = Rf_install(is_chain ? "llama_sampler_chain" : "llama_sampler");
    SEXP ptr = PROTECT(R_MakeExternalPtr(h, tag, R_NilValue));
    R_RegisterCFinalizerEx(ptr, llamar_sampler_handle_finalizer, TRUE);

    SEXP cls = PROTECT(Rf_mkString(is_chain ? "llama_sampler_chain" : "llama_sampler"));
    Rf_setAttrib(ptr, R_ClassSymbol, cls);
    UNPROTECT(2);
    return ptr;
}

// Resolve an R handle to a live sampler, refusing anything freed or stale.
//
// `thrw` picks how a bad handle is reported: throwing (for callers inside an
// LLAMAR_ENTRYPOINT_BEGIN/END pair that own C++ objects) or Rf_error's longjmp
// (for the plain entry points). Same reasoning as the two flavours in
// r_llama_ptr.h.
static llamar_sampler_handle * llamar_sampler_handle_get_(SEXP r_smpl, bool thrw) {
    if (TYPEOF(r_smpl) != EXTPTRSXP) {
        if (thrw) llamar_error("llamaR: expected a sampler handle");
        Rf_error("llamaR: expected a sampler handle");
    }
    llamar_sampler_handle * h = (llamar_sampler_handle *) R_ExternalPtrAddr(r_smpl);
    if (!h || !h->smpl) {
        if (thrw) llamar_error("llamaR: sampler has already been freed");
        Rf_error("llamaR: sampler has already been freed");
    }
    // A sampler owned by a chain dies with that chain; the generation snapshot
    // catches exactly that case.
    if (h->parent_life) {
        if (!h->parent_life->alive ||
            h->parent_life->generation != h->parent_generation) {
            if (thrw) llamar_error("llamaR: sampler belonged to a chain that has been freed");
            Rf_error("llamaR: sampler belonged to a chain that has been freed");
        }
    }
    return h;
}

static llamar_sampler_handle * llamar_sampler_handle_get(SEXP r_smpl) {
    return llamar_sampler_handle_get_(r_smpl, /* thrw = */ false);
}

static llamar_sampler_handle * llamar_sampler_handle_get_throw(SEXP r_smpl) {
    return llamar_sampler_handle_get_(r_smpl, /* thrw = */ true);
}

static llamar_sampler_handle * llamar_chain_handle_get(SEXP r_chain) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_chain);
    if (!h->is_chain) Rf_error("llamaR: expected a sampler chain");
    return h;
}

static llamar_sampler_handle * llamar_chain_handle_get_throw(SEXP r_chain) {
    llamar_sampler_handle * h = llamar_sampler_handle_get_throw(r_chain);
    if (!h->is_chain) llamar_error("llamaR: expected a sampler chain");
    return h;
}

// Definition for the forward declaration used by the generation entry points.
static llama_sampler * llamar_handle_chain_ptr(llamar_sampler_handle * h) {
    return h->smpl;
}

extern "C" SEXP r_llama_sampler_chain_new(SEXP r_no_perf) {
    auto params = llama_sampler_chain_default_params();
    if (!Rf_isNull(r_no_perf)) {
        int b = Rf_asLogical(r_no_perf);
        if (b != NA_LOGICAL) params.no_perf = (b == TRUE);
    }
    llama_sampler * chain = llama_sampler_chain_init(params);
    if (!chain) Rf_error("llamaR: failed to create sampler chain");
    return llamar_new_sampler_handle(chain, true, true);
}

// Build one standalone sampler by name. This is the practical replacement for
// binding llama_sampler_init(), which exists to implement new samplers in C and
// would need an R callback in the decode loop to be useful from R.
extern "C" SEXP r_llama_sampler_new(SEXP r_kind, SEXP r_args, SEXP r_model) {
    LLAMAR_ENTRYPOINT_BEGIN
    const char * kind = CHAR(STRING_ELT(r_kind, 0));

    const llama_model * model = NULL;
    const llama_vocab * vocab = NULL;
    if (!Rf_isNull(r_model)) {
        model = llamar_model_arg_throw(r_model);
        vocab = llama_model_get_vocab(model);
    }

    // Defaults come from the same struct the declarative path uses, so a
    // sampler built here matches one built by llama_sampler_params().
    const llamar_sampler_params p = llamar_sampler_params_from_sexp(r_args);
    const size_t min_keep = (size_t) (p.min_keep > 0 ? p.min_keep : 1);

    llama_sampler * s = NULL;

    if      (strcmp(kind, "greedy")      == 0) s = llama_sampler_init_greedy();
    else if (strcmp(kind, "dist")        == 0) s = llama_sampler_init_dist(p.seed);
    else if (strcmp(kind, "top_k")       == 0) s = llama_sampler_init_top_k(p.top_k);
    else if (strcmp(kind, "top_p")       == 0) s = llama_sampler_init_top_p(p.top_p, min_keep);
    else if (strcmp(kind, "min_p")       == 0) s = llama_sampler_init_min_p(p.min_p, min_keep);
    else if (strcmp(kind, "typical")     == 0) s = llama_sampler_init_typical(p.typical_p, min_keep);
    else if (strcmp(kind, "temp")        == 0) s = llama_sampler_init_temp(p.temp);
    else if (strcmp(kind, "temp_ext")    == 0) s = llama_sampler_init_temp_ext(
                                                       p.temp, p.dynatemp_range, p.dynatemp_exponent);
    else if (strcmp(kind, "xtc")         == 0) s = llama_sampler_init_xtc(
                                                       p.xtc_probability, p.xtc_threshold, min_keep, p.seed);
    else if (strcmp(kind, "top_n_sigma") == 0) s = llama_sampler_init_top_n_sigma(p.top_n_sigma);
    else if (strcmp(kind, "penalties")   == 0) s = llama_sampler_init_penalties(
                                                       p.repeat_last_n, p.repeat_penalty,
                                                       p.freq_penalty, p.pres_penalty);
    else if (strcmp(kind, "adaptive_p")  == 0) s = llama_sampler_init_adaptive_p(
                                                       p.adaptive_target, p.adaptive_decay, p.seed);
    else if (strcmp(kind, "mirostat_v2") == 0) s = llama_sampler_init_mirostat_v2(
                                                       p.seed, p.mirostat_tau, p.mirostat_eta);
    else if (strcmp(kind, "mirostat")    == 0) {
        if (!vocab) llamar_error("llamaR: sampler '%s' requires a model", kind);
        s = llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), p.seed,
                                        p.mirostat_tau, p.mirostat_eta, 100);
    }
    else if (strcmp(kind, "infill")      == 0) {
        if (!vocab) llamar_error("llamaR: sampler '%s' requires a model", kind);
        s = llama_sampler_init_infill(vocab);
    }
    else if (strcmp(kind, "logit_bias")  == 0) {
        if (!vocab) llamar_error("llamaR: sampler '%s' requires a model", kind);
        if (p.logit_bias.empty()) llamar_error("llamaR: sampler 'logit_bias' needs a logit_bias argument");
        s = llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab),
                                          (int32_t) p.logit_bias.size(),
                                          p.logit_bias.data());
    }
    else if (strcmp(kind, "dry")         == 0) {
        if (!model) llamar_error("llamaR: sampler '%s' requires a model", kind);
        std::vector<const char *> breakers;
        breakers.reserve(p.dry_sequence_breakers.size());
        for (const auto & b : p.dry_sequence_breakers) breakers.push_back(b.c_str());
        s = llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model),
                                   p.dry_multiplier, p.dry_base,
                                   p.dry_allowed_length, p.dry_penalty_last_n,
                                   breakers.data(), breakers.size());
    }
    else {
        llamar_error("llamaR: unknown sampler kind '%s'", kind);
    }

    if (!s) llamar_error("llamaR: failed to create sampler '%s'", kind);
    return llamar_new_sampler_handle(s, true, false);
    LLAMAR_ENTRYPOINT_END
}

// Build a full chain from a llama_sampler_params() list, so the declarative
// path can be inspected and adjusted through the chain API.
extern "C" SEXP r_llama_sampler_chain_from_params(SEXP r_ctx, SEXP r_params,
                                                  SEXP r_grammar,
                                                  SEXP r_trigger_patterns,
                                                  SEXP r_trigger_tokens) {
    LLAMAR_ENTRYPOINT_BEGIN
    llama_context * ctx = llamar_ctx_arg_throw(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const llamar_sampler_params p = llamar_sampler_params_from_sexp(r_params);
    const char * grammar = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(sparams);
    if (!chain) llamar_error("llamaR: failed to create sampler chain");

    llamar_build_sampler_chain(chain, model, vocab, p, p.seed, grammar,
                               r_trigger_patterns, r_trigger_tokens);

    return llamar_new_sampler_handle(chain, true, true);
    LLAMAR_ENTRYPOINT_END
}

extern "C" SEXP r_llama_sampler_chain_add(SEXP r_chain, SEXP r_smpl) {
    llamar_sampler_handle * ch = llamar_chain_handle_get(r_chain);
    llamar_sampler_handle * h  = llamar_sampler_handle_get(r_smpl);

    if (!h->owned) {
        Rf_error("llamaR: sampler is already owned by a chain");
    }
    if (h == ch) {
        Rf_error("llamaR: cannot add a chain to itself");
    }

    llama_sampler_chain_add(ch->smpl, h->smpl);

    // Ownership moved to the chain; record which chain, so the handle can be
    // invalidated when that chain is freed.
    h->owned             = false;
    h->parent_life       = ch->life;
    h->parent_generation = ch->life->generation;

    return R_NilValue;
}

extern "C" SEXP r_llama_sampler_chain_n(SEXP r_chain) {
    llamar_sampler_handle * ch = llamar_chain_handle_get(r_chain);
    return Rf_ScalarInteger(llama_sampler_chain_n(ch->smpl));
}

// Borrowed view of a chain member: the returned handle never owns the sampler
// and is invalidated together with its chain.
extern "C" SEXP r_llama_sampler_chain_get(SEXP r_chain, SEXP r_i) {
    llamar_sampler_handle * ch = llamar_chain_handle_get(r_chain);
    const int i = INTEGER(r_i)[0];

    llama_sampler * s = llama_sampler_chain_get(ch->smpl, i);
    if (!s) Rf_error("llamaR: no sampler at index %d", i);

    SEXP ptr = PROTECT(llamar_new_sampler_handle(s, false, false));
    llamar_sampler_handle * h = (llamar_sampler_handle *) R_ExternalPtrAddr(ptr);
    h->parent_life       = ch->life;
    h->parent_generation = ch->life->generation;
    UNPROTECT(1);
    return ptr;
}

// Detach a sampler from its chain, handing ownership back to R.
extern "C" SEXP r_llama_sampler_chain_remove(SEXP r_chain, SEXP r_i) {
    llamar_sampler_handle * ch = llamar_chain_handle_get(r_chain);
    const int i = INTEGER(r_i)[0];

    llama_sampler * s = llama_sampler_chain_remove(ch->smpl, i);
    if (!s) Rf_error("llamaR: no sampler at index %d", i);

    // Any handle the caller still holds for this sampler (from chain_add or
    // chain_get) points at the same object the new handle now owns, so bump the
    // chain's generation to retire those older handles. Without this, two
    // handles would own-or-borrow one sampler and the first free would leave
    // the other dangling.
    ch->life->generation++;

    // The chain no longer owns it, so R does. The fresh handle carries no
    // parent_life at all: the link is severed outright rather than left holding
    // a generation number that a later wrap-around could falsely match.
    return llamar_new_sampler_handle(s, true, false);
}

extern "C" SEXP r_llama_sampler_name(SEXP r_smpl) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_smpl);
    const char * name = llama_sampler_name(h->smpl);
    return Rf_mkString(name ? name : "");
}

extern "C" SEXP r_llama_sampler_reset(SEXP r_smpl) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_smpl);
    llama_sampler_reset(h->smpl);
    return R_NilValue;
}

extern "C" SEXP r_llama_sampler_clone(SEXP r_smpl) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_smpl);
    llama_sampler * copy = llama_sampler_clone(h->smpl);
    if (!copy) Rf_error("llamaR: this sampler cannot be cloned");
    // A clone is always freshly owned, even when cloned from a chain member.
    return llamar_new_sampler_handle(copy, true, h->is_chain);
}

extern "C" SEXP r_llama_sampler_accept(SEXP r_smpl, SEXP r_token) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_smpl);
    llama_sampler_accept(h->smpl, (llama_token) INTEGER(r_token)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_sampler_get_seed(SEXP r_smpl) {
    llamar_sampler_handle * h = llamar_sampler_handle_get(r_smpl);
    const uint32_t seed = llama_sampler_get_seed(h->smpl);
    // LLAMA_DEFAULT_SEED means "no seed of its own"; it does not fit in an R
    // integer, so report it as NA rather than silently wrapping.
    if (seed == LLAMA_DEFAULT_SEED || seed > (uint32_t) INT_MAX) {
        return Rf_ScalarInteger(NA_INTEGER);
    }
    return Rf_ScalarInteger((int) seed);
}

// Free eagerly instead of waiting for the GC. Freeing a chain invalidates every
// handle to a sampler inside it, which the generation check then reports.
extern "C" SEXP r_llama_sampler_free(SEXP r_smpl) {
    if (TYPEOF(r_smpl) != EXTPTRSXP) Rf_error("llamaR: expected a sampler handle");
    llamar_sampler_handle * h = (llamar_sampler_handle *) R_ExternalPtrAddr(r_smpl);
    if (!h || !h->smpl) return R_NilValue;   // already freed: nothing to do

    if (!h->owned) {
        Rf_error("llamaR: this sampler is owned by a chain; free the chain instead");
    }
    if (h->is_chain && h->life) {
        h->life->alive = false;
        h->life->generation++;
    }
    llama_sampler_free(h->smpl);
    h->smpl = NULL;
    return R_NilValue;
}

// ============================================================
// Registration
// ============================================================

// Defined in r_chat_interface.cpp (kept separate so the heavy C++ chat/template
// headers don't pull into this translation unit).
extern "C" SEXP r_llama_chat_build(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP r_llama_chat_parse(SEXP, SEXP, SEXP, SEXP);

// Multimodal entry points (defined in r_mtmd_interface.cpp)
extern "C" SEXP r_mtmd_init(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP r_mtmd_support_vision(SEXP);
extern "C" SEXP r_mtmd_support_audio(SEXP);
extern "C" SEXP r_mtmd_marker(void);
extern "C" SEXP r_mtmd_set_verbosity(SEXP);
extern "C" SEXP r_mtmd_bitmap_from_file(SEXP, SEXP);
extern "C" SEXP r_mtmd_eval(SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    // Version & hardware
    {"r_llama_version",               (DL_FUNC) &r_llama_version,               0},
    {"r_llama_supports_gpu",          (DL_FUNC) &r_llama_supports_gpu,          0},
    {"r_llama_supports_mmap",         (DL_FUNC) &r_llama_supports_mmap,         0},
    {"r_llama_supports_mlock",        (DL_FUNC) &r_llama_supports_mlock,        0},
    {"r_llama_max_devices",           (DL_FUNC) &r_llama_max_devices,           0},
    {"r_llama_system_info",           (DL_FUNC) &r_llama_system_info,           0},
    // Verbosity
    {"r_llama_set_verbosity",         (DL_FUNC) &r_llama_set_verbosity,         1},
    {"r_llama_get_verbosity",         (DL_FUNC) &r_llama_get_verbosity,         0},
    // Model
    {"r_llama_time_us",               (DL_FUNC) &r_llama_time_us,               0},
    {"r_llama_numa_init",             (DL_FUNC) &r_llama_numa_init,             1},
    {"r_llama_backend_devices",       (DL_FUNC) &r_llama_backend_devices,       0},
    {"r_llama_load_model",            (DL_FUNC) &r_llama_load_model,            6},
    {"r_llama_load_model_from_splits",(DL_FUNC) &r_llama_load_model_from_splits,6},
    {"r_llama_split_path",            (DL_FUNC) &r_llama_split_path,            3},
    {"r_llama_split_prefix",          (DL_FUNC) &r_llama_split_prefix,          3},
    {"r_llama_free_model",            (DL_FUNC) &r_llama_free_model,            1},
    {"r_llama_model_info",            (DL_FUNC) &r_llama_model_info,            1},
    {"r_llama_model_size",            (DL_FUNC) &r_llama_model_size,            1},
    {"r_llama_model_n_params",        (DL_FUNC) &r_llama_model_n_params,        1},
    {"r_llama_model_has_encoder",     (DL_FUNC) &r_llama_model_has_encoder,     1},
    {"r_llama_model_has_decoder",     (DL_FUNC) &r_llama_model_has_decoder,     1},
    {"r_llama_model_is_recurrent",    (DL_FUNC) &r_llama_model_is_recurrent,    1},
    {"r_llama_model_is_hybrid",       (DL_FUNC) &r_llama_model_is_hybrid,       1},
    {"r_llama_model_is_diffusion",    (DL_FUNC) &r_llama_model_is_diffusion,    1},
    {"r_llama_model_n_embd_inp",      (DL_FUNC) &r_llama_model_n_embd_inp,      1},
    {"r_llama_model_n_embd_out",      (DL_FUNC) &r_llama_model_n_embd_out,      1},
    {"r_llama_model_n_swa",           (DL_FUNC) &r_llama_model_n_swa,           1},
    {"r_llama_model_rope_type",       (DL_FUNC) &r_llama_model_rope_type,       1},
    {"r_llama_model_rope_freq_scale_train",
                                      (DL_FUNC) &r_llama_model_rope_freq_scale_train, 1},
    {"r_llama_model_n_cls_out",       (DL_FUNC) &r_llama_model_n_cls_out,       1},
    {"r_llama_model_cls_labels",      (DL_FUNC) &r_llama_model_cls_labels,      1},
    {"r_llama_model_decoder_start_token",
                                      (DL_FUNC) &r_llama_model_decoder_start_token, 1},
    {"r_llama_model_sampling_meta_keys",
                                      (DL_FUNC) &r_llama_model_sampling_meta_keys, 0},
    {"r_llama_model_meta",            (DL_FUNC) &r_llama_model_meta,            1},
    {"r_llama_model_meta_val",        (DL_FUNC) &r_llama_model_meta_val,        2},
    // Vocabulary
    {"r_llama_vocab_info",            (DL_FUNC) &r_llama_vocab_info,            1},
    {"r_llama_vocab_type",            (DL_FUNC) &r_llama_vocab_type,            1},
    {"r_llama_vocab_is_eog",          (DL_FUNC) &r_llama_vocab_is_eog,          2},
    {"r_llama_vocab_is_control",      (DL_FUNC) &r_llama_vocab_is_control,      2},
    {"r_llama_vocab_get_text",        (DL_FUNC) &r_llama_vocab_get_text,        2},
    {"r_llama_vocab_get_score",       (DL_FUNC) &r_llama_vocab_get_score,       2},
    {"r_llama_vocab_get_attr",        (DL_FUNC) &r_llama_vocab_get_attr,        2},
    {"r_llama_vocab_get_add_bos",     (DL_FUNC) &r_llama_vocab_get_add_bos,     1},
    {"r_llama_vocab_get_add_eos",     (DL_FUNC) &r_llama_vocab_get_add_eos,     1},
    {"r_llama_vocab_get_add_sep",     (DL_FUNC) &r_llama_vocab_get_add_sep,     1},
    {"r_llama_vocab_mask",            (DL_FUNC) &r_llama_vocab_mask,            1},
    {"r_llama_vocab_fim_pad",         (DL_FUNC) &r_llama_vocab_fim_pad,         1},
    {"r_llama_max_parallel_sequences",
                                      (DL_FUNC) &r_llama_max_parallel_sequences, 0},
    {"r_llama_max_tensor_buft_overrides",
                                      (DL_FUNC) &r_llama_max_tensor_buft_overrides, 0},
    // Context
    {"r_llama_new_context",           (DL_FUNC) &r_llama_new_context,           9},
    {"r_llama_free_context",          (DL_FUNC) &r_llama_free_context,          1},
    {"r_llama_get_model",             (DL_FUNC) &r_llama_get_model,             1},
    {"r_llama_set_warmup",            (DL_FUNC) &r_llama_set_warmup,            2},
    {"r_llama_set_abort_callback",    (DL_FUNC) &r_llama_set_abort_callback,    2},
    {"r_llama_n_ctx",                 (DL_FUNC) &r_llama_n_ctx,                 1},
    {"r_llama_n_ctx_seq",             (DL_FUNC) &r_llama_n_ctx_seq,             1},
    {"r_llama_n_batch",               (DL_FUNC) &r_llama_n_batch,               1},
    {"r_llama_n_ubatch",              (DL_FUNC) &r_llama_n_ubatch,              1},
    {"r_llama_n_seq_max",             (DL_FUNC) &r_llama_n_seq_max,             1},
    {"r_llama_flash_attn_type_name",  (DL_FUNC) &r_llama_flash_attn_type_name,  1},
    {"r_llama_context_flash_attn",    (DL_FUNC) &r_llama_context_flash_attn,    1},
    {"r_llama_n_threads",             (DL_FUNC) &r_llama_n_threads,             1},
    {"r_llama_n_threads_batch",       (DL_FUNC) &r_llama_n_threads_batch,       1},
    {"r_llama_pooling_type",          (DL_FUNC) &r_llama_pooling_type,          1},
    {"r_llama_set_n_threads",         (DL_FUNC) &r_llama_set_n_threads,         3},
    {"r_llama_set_causal_attn",       (DL_FUNC) &r_llama_set_causal_attn,       2},
    // Tokenize / Detokenize / Token piece
    {"r_llama_tokenize",              (DL_FUNC) &r_llama_tokenize,              4},
    {"r_llama_detokenize",            (DL_FUNC) &r_llama_detokenize,            2},
    {"r_llama_token_to_piece",        (DL_FUNC) &r_llama_token_to_piece,        3},
    // Batch
    {"r_llama_batch_init",            (DL_FUNC) &r_llama_batch_init,            3},
    {"r_llama_batch_free",            (DL_FUNC) &r_llama_batch_free,            1},
    // Encode
    {"r_llama_encode",                (DL_FUNC) &r_llama_encode,                2},
    // Generate
    {"r_llama_generate",              (DL_FUNC) &r_llama_generate,               9},
    {"r_llama_gen_begin",             (DL_FUNC) &r_llama_gen_begin,              8},
    {"r_llama_gen_begin_at",          (DL_FUNC) &r_llama_gen_begin_at,           8},
    {"r_llama_gen_next",              (DL_FUNC) &r_llama_gen_next,              1},
    {"r_llama_gen_end",               (DL_FUNC) &r_llama_gen_end,               1},
    {"r_llama_generate_batch",        (DL_FUNC) &r_llama_generate_batch,         7},
    // Embeddings & Logits
    {"r_llama_embeddings",            (DL_FUNC) &r_llama_embeddings,            2},
    {"r_llama_embed_batch",           (DL_FUNC) &r_llama_embed_batch,           2},
    {"r_llama_get_embeddings",        (DL_FUNC) &r_llama_get_embeddings,        2},
    {"r_llama_get_embeddings_ith",    (DL_FUNC) &r_llama_get_embeddings_ith,    2},
    {"r_llama_get_embeddings_seq",    (DL_FUNC) &r_llama_get_embeddings_seq,    2},
    {"r_llama_get_logits",            (DL_FUNC) &r_llama_get_logits,            1},
    {"r_llama_get_logits_ith",        (DL_FUNC) &r_llama_get_logits_ith,        2},
    // Memory / KV Cache
    {"r_llama_memory_clear",          (DL_FUNC) &r_llama_memory_clear,          1},
    {"r_llama_memory_seq_rm",         (DL_FUNC) &r_llama_memory_seq_rm,         4},
    {"r_llama_memory_seq_cp",         (DL_FUNC) &r_llama_memory_seq_cp,         5},
    {"r_llama_memory_seq_keep",       (DL_FUNC) &r_llama_memory_seq_keep,       2},
    {"r_llama_memory_seq_add",        (DL_FUNC) &r_llama_memory_seq_add,        5},
    {"r_llama_memory_seq_div",        (DL_FUNC) &r_llama_memory_seq_div,        5},
    {"r_llama_memory_seq_pos_range",  (DL_FUNC) &r_llama_memory_seq_pos_range,  2},
    {"r_llama_memory_can_shift",      (DL_FUNC) &r_llama_memory_can_shift,      1},
    // State
    {"r_llama_state_save",            (DL_FUNC) &r_llama_state_save,            2},
    {"r_llama_state_load",            (DL_FUNC) &r_llama_state_load,            2},
    {"r_llama_state_get_data",        (DL_FUNC) &r_llama_state_get_data,        1},
    {"r_llama_state_set_data",        (DL_FUNC) &r_llama_state_set_data,        2},
    {"r_llama_state_seq_get_size",    (DL_FUNC) &r_llama_state_seq_get_size,    3},
    {"r_llama_state_seq_get_data",    (DL_FUNC) &r_llama_state_seq_get_data,    3},
    {"r_llama_state_seq_set_data",    (DL_FUNC) &r_llama_state_seq_set_data,    4},
    {"r_llama_state_seq_save_file",   (DL_FUNC) &r_llama_state_seq_save_file,   4},
    {"r_llama_state_seq_load_file",   (DL_FUNC) &r_llama_state_seq_load_file,   4},
    {"r_llama_state_get_size",        (DL_FUNC) &r_llama_state_get_size,        1},
    {"r_llama_synchronize",           (DL_FUNC) &r_llama_synchronize,           1},
    // Chat templates
    {"r_llama_chat_template",         (DL_FUNC) &r_llama_chat_template,         2},
    {"r_llama_chat_apply_template",   (DL_FUNC) &r_llama_chat_apply_template,   3},
    {"r_llama_chat_builtin_templates",(DL_FUNC) &r_llama_chat_builtin_templates,0},
    // Tool-aware chat templates + parsing (r_chat_interface.cpp)
    {"r_llama_chat_build",            (DL_FUNC) &r_llama_chat_build,            7},
    {"r_llama_chat_parse",            (DL_FUNC) &r_llama_chat_parse,            4},
    // LoRA
    {"r_llama_lora_load",             (DL_FUNC) &r_llama_lora_load,             2},
    {"r_llama_lora_apply",            (DL_FUNC) &r_llama_lora_apply,            3},
    {"r_llama_lora_remove",           (DL_FUNC) &r_llama_lora_remove,           2},
    {"r_llama_lora_clear",            (DL_FUNC) &r_llama_lora_clear,            1},
    {"r_llama_lora_meta",             (DL_FUNC) &r_llama_lora_meta,             1},
    {"r_llama_lora_meta_val",         (DL_FUNC) &r_llama_lora_meta_val,         2},
    {"r_llama_lora_alora_invocation_tokens",
                                      (DL_FUNC) &r_llama_lora_alora_invocation_tokens, 1},
    {"r_llama_apply_adapter_cvec",    (DL_FUNC) &r_llama_apply_adapter_cvec,    5},
    // Sampler chain API
    {"r_llama_sampler_chain_new",     (DL_FUNC) &r_llama_sampler_chain_new,     1},
    {"r_llama_sampler_new",           (DL_FUNC) &r_llama_sampler_new,           3},
    {"r_llama_sampler_chain_from_params",
                                      (DL_FUNC) &r_llama_sampler_chain_from_params, 5},
    {"r_llama_sampler_chain_add",     (DL_FUNC) &r_llama_sampler_chain_add,     2},
    {"r_llama_sampler_chain_n",       (DL_FUNC) &r_llama_sampler_chain_n,       1},
    {"r_llama_sampler_chain_get",     (DL_FUNC) &r_llama_sampler_chain_get,     2},
    {"r_llama_sampler_chain_remove",  (DL_FUNC) &r_llama_sampler_chain_remove,  2},
    {"r_llama_sampler_name",          (DL_FUNC) &r_llama_sampler_name,          1},
    {"r_llama_sampler_reset",         (DL_FUNC) &r_llama_sampler_reset,         1},
    {"r_llama_sampler_clone",         (DL_FUNC) &r_llama_sampler_clone,         1},
    {"r_llama_sampler_accept",        (DL_FUNC) &r_llama_sampler_accept,        2},
    {"r_llama_sampler_get_seed",      (DL_FUNC) &r_llama_sampler_get_seed,      1},
    {"r_llama_sampler_free",          (DL_FUNC) &r_llama_sampler_free,          1},
    // Performance & Debug
    {"r_llama_perf_context",          (DL_FUNC) &r_llama_perf_context,          1},
    {"r_llama_perf_sampler",          (DL_FUNC) &r_llama_perf_sampler,          1},
    {"r_llama_perf_sampler_print",    (DL_FUNC) &r_llama_perf_sampler_print,    1},
    {"r_llama_perf_sampler_reset",    (DL_FUNC) &r_llama_perf_sampler_reset,    1},
    {"r_llama_perf_context_reset",    (DL_FUNC) &r_llama_perf_context_reset,    1},
    {"r_llama_perf_context_print",    (DL_FUNC) &r_llama_perf_context_print,    1},
    {"r_llama_memory_breakdown_print",(DL_FUNC) &r_llama_memory_breakdown_print,1},
    // Hardware
    {"r_llama_supports_rpc",          (DL_FUNC) &r_llama_supports_rpc,          0},
    // Multimodal (mtmd / clip) — defined in r_mtmd_interface.cpp
    {"r_mtmd_init",                   (DL_FUNC) &r_mtmd_init,                   4},
    {"r_mtmd_support_vision",         (DL_FUNC) &r_mtmd_support_vision,         1},
    {"r_mtmd_support_audio",          (DL_FUNC) &r_mtmd_support_audio,          1},
    {"r_mtmd_marker",                 (DL_FUNC) &r_mtmd_marker,                 0},
    {"r_mtmd_set_verbosity",          (DL_FUNC) &r_mtmd_set_verbosity,          1},
    {"r_mtmd_bitmap_from_file",       (DL_FUNC) &r_mtmd_bitmap_from_file,       2},
    {"r_mtmd_eval",                   (DL_FUNC) &r_mtmd_eval,                   5},
    {NULL, NULL, 0}
};

extern "C" void R_init_llamaR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
