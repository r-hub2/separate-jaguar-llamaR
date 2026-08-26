# ============================================================
# Model / vocabulary / build introspection bindings.
#   llama_model_info() extra fields, llama_model_cls_labels(),
#   llama_model_decoder_start_token(), llama_model_sampling_meta(),
#   llama_vocab_get_attr(), llama_vocab_get_add_*(),
#   llama_vocab_mask() / _fim_pad(),
#   llama_max_parallel_sequences() / _tensor_buft_overrides()
# The build-limit queries need no model; the rest guard with
# skip_if_no_model() and so stay light enough for CRAN.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

# --- build limits: no model required --------------------------------------

test_that("build limits are positive integers", {
    n_seq <- llama_max_parallel_sequences()
    expect_type(n_seq, "integer")
    expect_length(n_seq, 1L)
    expect_gt(n_seq, 0L)

    n_ovr <- llama_max_tensor_buft_overrides()
    expect_type(n_ovr, "integer")
    expect_length(n_ovr, 1L)
    expect_gt(n_ovr, 0L)
})

test_that("a context cannot ask for more sequences than the build allows", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    # n_seq_max is bounded by the compile-time ceiling, whatever the model.
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L, n_seq_max = 2L)
    on.exit(llama_free_context(ctx), add = TRUE, after = FALSE)

    expect_lte(llama_n_seq_max(ctx), llama_max_parallel_sequences())
})

# --- model metadata --------------------------------------------------------

test_that("llama_model_info reports the architecture flags", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    info <- llama_model_info(model)

    for (f in c("is_recurrent", "is_hybrid", "is_diffusion")) {
        expect_type(info[[f]], "logical")
        expect_length(info[[f]], 1L)
        expect_false(is.na(info[[f]]))
    }

    # A plain transformer is none of these.
    expect_false(info$is_recurrent)
    expect_false(info$is_hybrid)
    expect_false(info$is_diffusion)
})

test_that("llama_model_info reports embedding widths and the SWA span", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    info <- llama_model_info(model)

    for (f in c("n_embd_inp", "n_embd_out", "n_swa", "n_cls_out")) {
        expect_type(info[[f]], "integer")
        expect_length(info[[f]], 1L)
    }

    expect_gt(info$n_embd_inp, 0L)
    expect_gt(info$n_embd_out, 0L)
    # On a model with no projection widening, all three agree.
    expect_equal(info$n_embd_inp, info$n_embd)
    expect_equal(info$n_embd_out, info$n_embd)
    # Full-context attention reports a zero window.
    expect_gte(info$n_swa, 0L)
})

test_that("llama_model_info reports the RoPE configuration", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    info <- llama_model_info(model)

    expect_type(info$rope_type, "character")
    expect_length(info$rope_type, 1L)
    expect_true(info$rope_type %in%
                c("none", "norm", "neox", "mrope", "imrope", "vision", "unknown"))

    expect_type(info$rope_freq_scale_train, "double")
    expect_gt(info$rope_freq_scale_train, 0)
})

test_that("llama_model_cls_labels returns NULL for a generative model", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    # Only classifiers and rerankers carry output labels.
    expect_null(llama_model_cls_labels(model))
})

test_that("llama_model_decoder_start_token is NA on a decoder-only model", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    tok <- llama_model_decoder_start_token(model)
    expect_type(tok, "integer")
    expect_length(tok, 1L)
    expect_true(is.na(tok))
})

test_that("llama_model_sampling_meta returns a named list", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    meta <- llama_model_sampling_meta(model)
    expect_type(meta, "list")
    # Most GGUFs record no sampling hints; whatever is present must be named
    # and stored as a string.
    if (length(meta) > 0) {
        expect_true(all(nzchar(names(meta))))
        for (v in meta) expect_type(v, "character")
    }
})

# --- vocabulary ------------------------------------------------------------

test_that("llama_vocab_get_attr agrees with llama_vocab_is_control", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    known <- c("unknown", "unused", "normal", "control", "user_defined",
               "byte", "normalized", "lstrip", "rstrip", "single_word")

    for (tok in c(0L, 1L, 2L, 3L, 259L)) {
        attrs <- llama_vocab_get_attr(model, tok)
        expect_type(attrs, "character")
        expect_true(all(attrs %in% known))
        # The dedicated predicate and the flag list must tell the same story.
        expect_equal("control" %in% attrs, llama_vocab_is_control(model, tok))
    }
})

test_that("llama_vocab_get_attr flags BOS and EOS as control tokens", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    v <- llama_vocab_info(model)
    for (tok in c(v[["bos"]], v[["eos"]])) {
        if (tok >= 0) expect_true("control" %in% llama_vocab_get_attr(model, tok))
    }
})

test_that("tokenizer BOS/EOS/SEP defaults are logical scalars", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    for (f in list(llama_vocab_get_add_bos, llama_vocab_get_add_eos,
                   llama_vocab_get_add_sep)) {
        val <- f(model)
        expect_type(val, "logical")
        expect_length(val, 1L)
        expect_false(is.na(val))
    }
})

test_that("add_bos matches what tokenization actually does", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    v    <- llama_vocab_info(model)
    toks <- llama_tokenize(ctx, "hello", add_special = TRUE)

    # If the tokenizer says it prepends BOS, the token stream must start with it.
    if (llama_vocab_get_add_bos(model) && v[["bos"]] >= 0) {
        expect_equal(toks[1], v[["bos"]])
    }
})

test_that("mask and fim_pad are token ids or NA", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    for (f in list(llama_vocab_mask, llama_vocab_fim_pad)) {
        tok <- f(model)
        expect_type(tok, "integer")
        expect_length(tok, 1L)
        # Absent tokens come back as NA, never as a negative id.
        if (!is.na(tok)) expect_gte(tok, 0L)
    }
})

test_that("vocabulary queries reject an invalid model pointer", {
    bad <- structure(list(), class = "externalptr")
    expect_error(llama_vocab_get_add_bos(bad))
    expect_error(llama_vocab_get_attr(bad, 0L))
})
