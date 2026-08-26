#' Set logging verbosity level
#'
#' Controls how much diagnostic output is printed during model loading and inference.
#'
#' @param level Integer verbosity level:
#'   - 0: Silent (no output)
#'   - 1: Errors only (default)
#'   - 2: Normal (warnings and info)
#'   - 3: Verbose (all debug messages)
#' @return No return value, called for side effects. Sets the global
#'   verbosity level used by the underlying 'llama.cpp' library.
#' @export
#' @examples
#' # Suppress all output
#' llama_set_verbosity(0)
#'
#' # Show only errors
#' llama_set_verbosity(1)
#'
#' # Verbose output for debugging
#' llama_set_verbosity(3)
llama_set_verbosity <- function(level) {
    stopifnot(is.numeric(level), length(level) == 1)
    .Call("r_llama_set_verbosity", as.integer(level))
    invisible(NULL)
}

#' Get current verbosity level
#'
#' @return An integer scalar indicating the current verbosity level
#'   (0 = silent, 1 = errors only, 2 = normal, 3 = verbose).
#' @export
#' @examples
#' # Save current level, suppress output, then restore
#' old <- llama_get_verbosity()
#' llama_set_verbosity(0)
#' # ... noisy operations ...
#' llama_set_verbosity(old)
llama_get_verbosity <- function() {
    .Call("r_llama_get_verbosity")
}

#' Check whether GPU offloading is available
#'
#' Returns `TRUE` if at least one GPU backend (e.g. Vulkan) was detected at
#' runtime. Use the result to decide whether to pass `n_gpu_layers != 0`
#' to [llama_load_model].
#'
#' @return A logical scalar: \code{TRUE} if at least one GPU backend
#'   (e.g. Vulkan) is available, \code{FALSE} otherwise.
#' @export
#' @examples
#' if (llama_supports_gpu()) {
#'   message("GPU available, will use Vulkan backend")
#' } else {
#'   message("GPU not available, using CPU only")
#' }
llama_supports_gpu <- function() {
    .Call("r_llama_supports_gpu")
}

#' Get current time in microseconds
#'
#' @return A numeric scalar with the current time in microseconds.
#' @export
#' @examples
#' # Measure elapsed time for an operation
#' t0 <- llama_time_us()
#' Sys.sleep(0.01)
#' elapsed_ms <- (llama_time_us() - t0) / 1000
#' cat("Elapsed:", round(elapsed_ms, 1), "ms\n")
llama_time_us <- function() {
    .Call("r_llama_time_us")
}

#' Initialize NUMA optimization
#'
#' Call once for better performance on NUMA systems.
#'
#' @param strategy NUMA strategy: \code{"disabled"} (default), \code{"distribute"},
#'   \code{"isolate"}, \code{"numactl"}, or \code{"mirror"}.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # On multi-socket servers, distribute memory across NUMA nodes
#' # for better memory bandwidth during inference
#' llama_numa_init("distribute")
#'
#' # Call before loading any models — affects all subsequent allocations
#' model <- llama_load_model("model.gguf", n_gpu_layers = 0L)
#' }
llama_numa_init <- function(strategy = "disabled") {
    strategies <- c(disabled = 0L, distribute = 1L, isolate = 2L,
                    numactl = 3L, mirror = 4L)
    if (!strategy %in% names(strategies))
        stop("llamaR: invalid NUMA strategy '", strategy,
             "'. Valid: ", paste(names(strategies), collapse = ", "))
    .Call("r_llama_numa_init", strategies[[strategy]])
    invisible(NULL)
}

#' List available backend devices
#'
#' Returns a data.frame of all compute devices (CPU, GPU, etc.) detected
#' by the ggml backend. Use device names from this list in the \code{devices}
#' parameter of \code{\link{llama_load_model}}.
#'
#' @return A data.frame with columns \code{name}, \code{description}, and
#'   \code{type} (one of \code{"cpu"}, \code{"gpu"}, \code{"igpu"}, \code{"accel"}).
#' @export
#' @examples
#' # List available compute devices and pick GPU names for llama_load_model()
#' devs <- llama_backend_devices()
#' print(devs)
#' gpu_names <- devs$name[devs$type == "GPU"]
llama_backend_devices <- function() {
    .Call("r_llama_backend_devices")
}

#' Load a GGUF model file
#'
#' @param path Path to the .gguf model file
#' @param n_gpu_layers Number of layers to offload to GPU
#'   (\code{-1L} = all, \code{0L} = CPU only). Default \code{-1L} offloads
#'   everything to the GPU when one is detected; if no GPU backend is
#'   available, falls back to CPU with a warning.
#' @param devices Character vector of device names or types to use for offloading.
#'   \code{NULL} (default) uses all available devices. Use \code{"cpu"} for CPU-only,
#'   \code{"gpu"} for first GPU, or specific device names from
#'   \code{\link{llama_backend_devices}}. Multiple devices enable multi-GPU split.
#' @param split_mode Multi-GPU split strategy: \code{"none"} (single GPU),
#'   \code{"layer"} (split layers across GPUs, default), or \code{"row"}
#'   (tensor-parallel across GPUs).
#' @param use_mmap Logical; map model file into memory (default \code{TRUE}).
#' @param use_mlock Logical; force the OS to keep model pages resident
#'   (default \code{FALSE}).
#' @return An external pointer (class \code{externalptr}) wrapping the loaded
#'   model. This handle is required by \code{\link{llama_new_context}},
#'   \code{\link{llama_model_info}}, and other model-level functions.
#'   Freed automatically by the garbage collector or manually via
#'   \code{\link{llama_free_model}}.
#' @export
#' @examples
#' \dontrun{
#' # Default: full GPU offload (falls back to CPU if no GPU)
#' model <- llama_load_model("model.gguf")
#'
#' # Force CPU-only
#' model <- llama_load_model("model.gguf", n_gpu_layers = 0L)
#'
#' # Explicit CPU-only backend
#' model <- llama_load_model("model.gguf", devices = "cpu")
#'
#' # Specific GPU device (see llama_backend_devices())
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L, devices = "Vulkan0")
#'
#' # Multi-GPU: use two devices
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L,
#'                           devices = c("Vulkan0", "Vulkan1"))
#' }
llama_load_model <- function(path, n_gpu_layers = -1L, devices = NULL,
                             split_mode = "layer",
                             use_mmap = TRUE, use_mlock = FALSE) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: model file does not exist: ", path)

    args <- llamar_model_load_args(n_gpu_layers, devices, split_mode)
    .Call("r_llama_load_model", path, args$n_gpu_layers, devices,
          args$split_mode, as.logical(use_mmap), as.logical(use_mlock))
}

# Validate and normalize the loading arguments shared by llama_load_model() and
# llama_load_model_from_splits().
llamar_model_load_args <- function(n_gpu_layers, devices, split_mode) {
    if (!is.null(devices)) stopifnot(is.character(devices))

    n_gpu_layers <- as.integer(n_gpu_layers)
    if (n_gpu_layers != 0L && !llama_supports_gpu()) {
        warning("llamaR: no GPU backend detected, falling back to CPU")
        n_gpu_layers <- 0L
    }

    split_mode_int <- switch(split_mode,
        "none"  = 0L,
        "layer" = 1L,
        "row"   = 2L,
        stop("split_mode must be 'none', 'layer', or 'row'"))

    list(n_gpu_layers = n_gpu_layers, split_mode = split_mode_int)
}

#' Load a model split across several GGUF files
#'
#' [llama_load_model] already handles splits named with llama.cpp's own pattern
#' (\code{<name>-00001-of-00003.gguf}): pointing it at the first chunk loads all
#' of them. Use this function when the files do not follow that pattern and must
#' be listed explicitly.
#'
#' @param paths Character vector of paths to the split files, \strong{in order}.
#'   The order is not inferred from the names, so a wrong order yields a broken
#'   model rather than an error.
#' @inheritParams llama_load_model
#' @return An external pointer wrapping the loaded model, exactly as
#'   [llama_load_model] returns.
#' @seealso [llama_load_model], [llama_split_path], [llama_split_prefix]
#' @export
#' @examples
#' \dontrun{
#' # Files with a custom naming scheme
#' model <- llama_load_model_from_splits(c("part-a.gguf", "part-b.gguf"))
#'
#' # Standard naming: build the paths, then load them
#' paths <- vapply(1:3, function(i) llama_split_path("model", i, 3), character(1))
#' model <- llama_load_model_from_splits(paths)
#'
#' # ...though for the standard pattern this is enough:
#' model <- llama_load_model("model-00001-of-00003.gguf")
#' }
llama_load_model_from_splits <- function(paths, n_gpu_layers = -1L, devices = NULL,
                                         split_mode = "layer",
                                         use_mmap = TRUE, use_mlock = FALSE) {
    stopifnot(is.character(paths), length(paths) >= 1)
    missing_files <- paths[!file.exists(paths)]
    if (length(missing_files)) {
        stop("llamaR: split file does not exist: ",
             paste(missing_files, collapse = ", "))
    }

    args <- llamar_model_load_args(n_gpu_layers, devices, split_mode)
    .Call("r_llama_load_model_from_splits", paths, args$n_gpu_layers, devices,
          args$split_mode, as.logical(use_mmap), as.logical(use_mlock))
}

#' Build the path of one chunk of a split GGUF
#'
#' Applies llama.cpp's split naming pattern,
#' \code{<prefix>-<split_no>-of-<split_count>.gguf}, with both numbers padded to
#' five digits.
#'
#' @param prefix Path prefix, without the split suffix or the \code{.gguf}
#'   extension (e.g. \code{"/models/ggml-model-q4_0"}).
#' @param split_no Which chunk, counting from 1 --- the same number that appears
#'   in the file name. (llama.cpp's C function counts from 0 here; the R
#'   interface counts from 1 throughout.)
#' @param split_count How many chunks in total.
#' @return A character scalar with the full path.
#' @seealso [llama_split_prefix], [llama_load_model_from_splits]
#' @export
#' @examples
#' llama_split_path("/models/ggml-model-q4_0", 2, 4)
#' # "/models/ggml-model-q4_0-00002-of-00004.gguf"
llama_split_path <- function(prefix, split_no, split_count) {
    stopifnot(is.character(prefix), length(prefix) == 1)
    .Call("r_llama_split_path", prefix,
          as.integer(split_no), as.integer(split_count))
}

#' Recover the prefix from a split GGUF path
#'
#' The inverse of [llama_split_path]. The path is only accepted when it really
#' is chunk \code{split_no} of \code{split_count}; any other combination returns
#' \code{NA}, which makes this usable as a test of whether a path matches a
#' given split.
#'
#' @param path Path to one chunk of a split GGUF.
#' @param split_no Which chunk the path is expected to be, counting from 1 ---
#'   the number as it appears in the file name.
#' @param split_count How many chunks the path is expected to be part of.
#' @return A character scalar with the prefix, or \code{NA_character_} when the
#'   path does not match \code{split_no} / \code{split_count}.
#' @seealso [llama_split_path], [llama_load_model_from_splits]
#' @export
#' @examples
#' llama_split_prefix("/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4)
#' # "/models/ggml-model-q4_0"
#'
#' # Mismatched numbers: NA
#' llama_split_prefix("/models/ggml-model-q4_0-00002-of-00004.gguf", 3, 4)
llama_split_prefix <- function(path, split_no, split_count) {
    stopifnot(is.character(path), length(path) == 1)
    .Call("r_llama_split_prefix", path,
          as.integer(split_no), as.integer(split_count))
}

#' Free a loaded model
#'
#' @param model Model handle returned by [llama_load_model]
#' @return No return value, called for side effects. Releases the memory
#'   associated with the model.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' # ... use model ...
#' llama_free_model(model)
#' }
llama_free_model <- function(model) {
    .Call("r_llama_free_model", model)
    invisible(NULL)
}

#' Get model metadata
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named list with fields:
#'   - `n_ctx_train`: context size the model was trained with
#'   - `n_embd`: embedding dimension
#'   - `n_vocab`: vocabulary size
#'   - `n_layer`: number of layers
#'   - `n_head`: number of attention heads
#'   - `n_head_kv`: number of key-value attention heads (GQA)
#'   - `desc`: human-readable model description string
#'   - `size`: model size in bytes
#'   - `n_params`: number of parameters
#'   - `has_encoder`: whether the model has an encoder
#'   - `has_decoder`: whether the model has a decoder
#'   - `is_recurrent`: whether the model is recurrent (e.g. Mamba)
#'   - `is_hybrid`: whether the model mixes attention and recurrent layers
#'     (e.g. Jamba, Qwen3.5)
#'   - `is_diffusion`: whether the model is a diffusion LLM
#'   - `n_embd_inp` / `n_embd_out`: input and output embedding widths, which
#'     differ from `n_embd` on models whose projections are wider than the
#'     residual stream
#'   - `n_swa`: sliding-window attention span, `0` when attention is full-context
#'   - `rope_type`: one of `"none"`, `"norm"`, `"neox"`, `"mrope"`, `"imrope"`,
#'     `"vision"`
#'   - `rope_freq_scale_train`: RoPE frequency scaling used during training
#'   - `n_cls_out`: number of classifier outputs (`0` for ordinary LLMs)
#' @seealso [llama_model_cls_labels], [llama_model_sampling_meta],
#'   [llama_vocab_info]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' info <- llama_model_info(model)
#' cat("Model:", info$desc, "\n")
#' cat("Layers:", info$n_layer, "\n")
#' cat("Context:", info$n_ctx_train, "\n")
#' cat("Size:", info$size / 1e9, "GB\n")
#' }
llama_model_info <- function(model) {
    info <- .Call("r_llama_model_info", model)
    info$size         <- .Call("r_llama_model_size", model)
    info$n_params     <- .Call("r_llama_model_n_params", model)
    info$has_encoder  <- .Call("r_llama_model_has_encoder", model)
    info$has_decoder  <- .Call("r_llama_model_has_decoder", model)
    info$is_recurrent <- .Call("r_llama_model_is_recurrent", model)
    info$is_hybrid    <- .Call("r_llama_model_is_hybrid", model)
    info$is_diffusion <- .Call("r_llama_model_is_diffusion", model)
    info$n_embd_inp   <- .Call("r_llama_model_n_embd_inp", model)
    info$n_embd_out   <- .Call("r_llama_model_n_embd_out", model)
    info$n_swa        <- .Call("r_llama_model_n_swa", model)
    info$rope_type    <- .Call("r_llama_model_rope_type", model)
    info$rope_freq_scale_train <- .Call("r_llama_model_rope_freq_scale_train", model)
    info$n_cls_out    <- .Call("r_llama_model_n_cls_out", model)
    info
}

#' Classifier output labels
#'
#' For classifier and reranker models, returns the label of each output. Plain
#' generative models have no classifier head and return `NULL`.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A character vector of length `llama_model_info(model)$n_cls_out`, or
#'   `NULL` when the model is not a classifier or provides no labels. Individual
#'   entries are `NA` when that output is unlabelled.
#' @seealso [llama_model_info]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("reranker.gguf")
#' llama_model_cls_labels(model)
#' }
llama_model_cls_labels <- function(model) {
    .Call("r_llama_model_cls_labels", model)
}

#' Token that starts decoding in encoder-decoder models
#'
#' @param model Model handle returned by [llama_load_model]
#' @return An integer token ID, or `NA_integer_` when the model does not define
#'   one (which is the case for all decoder-only models).
#' @seealso [llama_model_info]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("t5.gguf")
#' llama_model_decoder_start_token(model)
#' }
llama_model_decoder_start_token <- function(model) {
    .Call("r_llama_model_decoder_start_token", model)
}

#' Sampling parameters recommended by the model author
#'
#' Some GGUF files record the sampling settings their author recommends. This
#' reads those keys out of the model metadata and returns the values that are
#' present, so they can be passed on to [llama_generate].
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named list of the recommended settings the model actually defines
#'   (names such as `temp`, `top_k`, `top_p`, `min_p`), or an empty list when
#'   the GGUF carries none. Values are returned as strings, exactly as stored.
#' @seealso [llama_model_meta], [llama_model_meta_val], [llama_generate]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' str(llama_model_sampling_meta(model))
#' }
llama_model_sampling_meta <- function(model) {
    keys <- .Call("r_llama_model_sampling_meta_keys")
    out  <- lapply(keys, function(k) if (is.na(k)) NULL else llama_model_meta_val(model, k))
    names(out) <- names(keys)
    out[!vapply(out, is.null, logical(1))]
}

#' Create an inference context
#'
#' @param model Model handle returned by [llama_load_model]
#' @param n_ctx Context window size (number of tokens). 0 means use the model's trained value.
#' @param n_threads Number of CPU threads for single-token decode. \code{NULL}
#'   (default) picks \code{2L} when a GPU backend is available, otherwise \code{4L}.
#' @param n_threads_batch Number of CPU threads for batch (prompt) processing.
#'   \code{NULL} (default) inherits from \code{n_threads}.
#' @param n_batch Logical maximum batch size submitted to a single decode call
#'   (tokens). Default \code{2048L} matches llama.cpp.
#' @param n_ubatch Physical micro-batch size used inside decode. Larger values
#'   improve prefill throughput on GPU at the cost of memory. Default \code{512L}.
#' @param n_seq_max Maximum number of parallel sequences the context can hold
#'   simultaneously (KV cache is partitioned across them). Default \code{1L}
#'   for single-prompt use; raise to \code{N} when using
#'   \code{\link{llama_generate_batch}} with \code{N} prompts. Increasing this
#'   does not by itself enlarge the context — also size \code{n_ctx} accordingly.
#' @param flash_attn One of \code{"auto"} (let llama.cpp decide, default),
#'   \code{"on"} (force enable Flash Attention), or \code{"off"} (disable).
#' @param embedding Logical; if \code{TRUE}, create context in embedding mode.
#'   This enables embedding output and disables causal attention, suitable for
#'   embedding models (e.g. nomic-embed, bge). When \code{TRUE},
#'   \code{\link{llama_embed_batch}} uses efficient pooled batch decode.
#' @return An external pointer (class \code{externalptr}) wrapping the inference
#'   context. This handle is required by generation, tokenization, and embedding
#'   functions. Freed automatically by the garbage collector or manually via
#'   \code{\link{llama_free_context}}.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model, n_ctx = 4096L, n_threads = 8L)
#' # ... use context for generation ...
#' llama_free_context(ctx)
#' llama_free_model(model)
#'
#' # Tune for GPU prefill throughput
#' ctx <- llama_new_context(model, n_ctx = 4096L,
#'                          n_ubatch = 2048L, flash_attn = "on")
#'
#' # Embedding mode
#' emb_ctx <- llama_new_context(model, n_ctx = 512L, embedding = TRUE)
#' mat <- llama_embed_batch(emb_ctx, c("hello", "world"))
#' }
llama_new_context <- function(model, n_ctx = 2048L,
                              n_threads = NULL, n_threads_batch = NULL,
                              n_batch = 2048L, n_ubatch = 512L,
                              n_seq_max = 1L,
                              flash_attn = "auto",
                              embedding = FALSE) {
    if (is.null(n_threads)) {
        n_threads <- if (llama_supports_gpu()) 2L else 4L
    }
    if (is.null(n_threads_batch)) {
        n_threads_batch <- n_threads
    }

    flash_attn_int <- switch(flash_attn,
        "auto" = -1L,
        "on"   =  1L,
        "off"  =  0L,
        stop("flash_attn must be 'auto', 'on', or 'off'"))

    .Call("r_llama_new_context", model,
          as.integer(n_ctx),
          as.integer(n_threads), as.integer(n_threads_batch),
          as.integer(n_batch), as.integer(n_ubatch),
          as.integer(n_seq_max),
          flash_attn_int,
          as.logical(embedding))
}

#' Free an inference context
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects. Releases the memory
#'   associated with the inference context.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' # ... use context ...
#' llama_free_context(ctx)
#' }
llama_free_context <- function(ctx) {
    .Call("r_llama_free_context", ctx)
    invisible(NULL)
}

#' Tokenize text into token IDs
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param text Character string to tokenize
#' @param add_special Whether to add special tokens (BOS/EOS) as configured by the model
#' @param parse_special Whether to parse control/special tokens (e.g. Mistral's
#'   \code{[INST]}, ChatML's \code{<|im_start|>}) as single tokens rather than
#'   as their literal characters. Use \code{TRUE} for a prompt produced by
#'   [llama_chat_apply_template]; the default \code{FALSE} treats such markup as
#'   plain text.
#' @return An integer vector of token IDs as used by the model's vocabulary.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' tokens <- llama_tokenize(ctx, "Hello, world!")
#' print(tokens)
#' # [1] 1 15043 29892 3186 29991
#'
#' # Without special tokens
#' tokens <- llama_tokenize(ctx, "Hello", add_special = FALSE)
#'
#' # Parse a templated prompt's role markers as control tokens
#' prompt <- llama_chat_apply_template(list(list(role = "user", content = "hi")))
#' tokens <- llama_tokenize(ctx, prompt, parse_special = TRUE)
#' }
llama_tokenize <- function(ctx, text, add_special = TRUE, parse_special = FALSE) {
    stopifnot(is.character(text), length(text) == 1)
    .Call("r_llama_tokenize", ctx, text,
          as.logical(add_special), as.logical(parse_special))
}

#' Detokenize token IDs back to text
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param tokens Integer vector of token IDs (as returned by [llama_tokenize])
#' @return A character scalar containing the decoded text.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' # Round-trip: text -> tokens -> text
#' original <- "Hello, world!"
#' tokens <- llama_tokenize(ctx, original, add_special = FALSE)
#' restored <- llama_detokenize(ctx, tokens)
#' identical(original, restored)  # TRUE
#' }
llama_detokenize <- function(ctx, tokens) {
    stopifnot(is.integer(tokens))
    .Call("r_llama_detokenize", ctx, tokens)
}

#' Describe a sampler chain
#'
#' Bundles every sampling parameter into one list, which the generation
#' functions ([llama_generate], [llama_gen_begin], [llama_gen_begin_at],
#' [llama_generate_batch]) accept as their \code{sampler} argument. Use it to
#' reach the samplers that have no dedicated argument on those functions --- DRY,
#' XTC, dynamic temperature, top-n-sigma, logit bias, infill and adaptive-p.
#'
#' The chain is assembled in the same order llama.cpp itself uses:
#' grammar, logit bias, penalties, DRY, top-n-sigma, top-k, typical-p, top-p,
#' min-p, XTC, infill, temperature, and finally a token-selecting sampler
#' (greedy when \code{temp <= 0}, adaptive-p when enabled, otherwise
#' distribution sampling). Setting \code{mirostat} to 1 or 2 replaces the whole
#' truncation section with temperature + Mirostat, as upstream does.
#'
#' @param temp Sampling temperature. 0 or less = greedy decoding.
#' @param top_k Top-K filtering (0 = disabled)
#' @param top_p Top-P (nucleus) filtering (1.0 = disabled)
#' @param min_p Min-P filtering threshold (0.0 = disabled)
#' @param typical_p Locally typical sampling threshold (1.0 = disabled)
#' @param seed Random seed for sampling
#' @param min_keep Minimum number of candidates the truncation samplers must
#'   leave in place (1 = upstream default)
#' @param repeat_penalty Repetition penalty (1.0 = disabled)
#' @param repeat_last_n Number of last tokens to penalize (0 = disabled,
#'   -1 = context size)
#' @param frequency_penalty Frequency penalty (0.0 = disabled)
#' @param presence_penalty Presence penalty (0.0 = disabled)
#' @param mirostat Mirostat sampling mode (0 = disabled, 1 = Mirostat,
#'   2 = Mirostat 2.0)
#' @param mirostat_tau Mirostat target entropy (tau parameter)
#' @param mirostat_eta Mirostat learning rate (eta parameter)
#' @param dynatemp_range Dynamic-temperature range (0.0 = plain temperature).
#'   The temperature varies within \code{temp +/- dynatemp_range} according to
#'   the entropy of the distribution.
#' @param dynatemp_exponent Controls how entropy maps to temperature in dynamic
#'   temperature sampling.
#' @param xtc_probability Probability of applying XTC ("exclude top choices")
#'   at each step (0.0 = disabled)
#' @param xtc_threshold XTC probability threshold; above 0.5 disables XTC
#' @param top_n_sigma Top-n-sigma filtering, in standard deviations of the
#'   logit distribution (negative = disabled)
#' @param dry_multiplier DRY repetition-penalty multiplier (0.0 = disabled)
#' @param dry_base DRY penalty base; the penalty grows as
#'   \code{dry_multiplier * dry_base ^ (repeat length - dry_allowed_length)}
#' @param dry_allowed_length Repetitions longer than this are penalized
#' @param dry_penalty_last_n How many recent tokens DRY scans for repetitions
#'   (0 = disabled, -1 = context size)
#' @param dry_sequence_breakers Character vector of strings that reset DRY's
#'   repetition tracking
#' @param adaptive_target Adaptive-p target probability (negative = disabled).
#'   When enabled it selects the token instead of distribution sampling.
#' @param adaptive_decay EMA decay used by adaptive-p; the history spans about
#'   \code{1 / (1 - adaptive_decay)} tokens
#' @param infill If TRUE, add the fill-in-the-middle sampler (for FIM models)
#' @param logit_bias Per-token logit adjustment, as a list with integer
#'   \code{token} and numeric \code{bias} of equal length, e.g.
#'   \code{list(token = c(15L, 22L), bias = c(-5, 2))}. \code{NULL} = none.
#' @return A named list of sampler parameters, to be passed as the
#'   \code{sampler} argument of the generation functions.
#' @seealso [llama_generate], [llama_gen_begin], [llama_generate_batch]
#' @export
#' @examples
#' \dontrun{
#' # DRY repetition penalty plus XTC
#' sp <- llama_sampler_params(temp = 0.9, dry_multiplier = 0.8,
#'                            xtc_probability = 0.5)
#' llama_generate(ctx, "Tell me a story", sampler = sp)
#'
#' # Dynamic temperature
#' sp <- llama_sampler_params(temp = 1.0, dynatemp_range = 0.5)
#'
#' # Suppress a specific token
#' sp <- llama_sampler_params(logit_bias = list(token = 1234L, bias = -100))
#' }
llama_sampler_params <- function(temp = 0.8, top_k = 50L, top_p = 0.9,
                                 min_p = 0.0, typical_p = 1.0, seed = 42L,
                                 min_keep = 1L,
                                 repeat_penalty = 1.0, repeat_last_n = 64L,
                                 frequency_penalty = 0.0, presence_penalty = 0.0,
                                 mirostat = 0L, mirostat_tau = 5.0,
                                 mirostat_eta = 0.1,
                                 dynatemp_range = 0.0, dynatemp_exponent = 1.0,
                                 xtc_probability = 0.0, xtc_threshold = 0.1,
                                 top_n_sigma = -1.0,
                                 dry_multiplier = 0.0, dry_base = 1.75,
                                 dry_allowed_length = 2L,
                                 dry_penalty_last_n = -1L,
                                 dry_sequence_breakers = c("\n", ":", "\"", "*"),
                                 adaptive_target = -1.0, adaptive_decay = 0.9,
                                 infill = FALSE, logit_bias = NULL) {
    if (!is.null(logit_bias)) {
        if (!is.list(logit_bias) ||
            is.null(logit_bias$token) || is.null(logit_bias$bias)) {
            stop("logit_bias must be a list with 'token' and 'bias' elements")
        }
        if (length(logit_bias$token) != length(logit_bias$bias)) {
            stop("logit_bias 'token' and 'bias' must have the same length")
        }
        logit_bias <- list(token = as.integer(logit_bias$token),
                           bias  = as.double(logit_bias$bias))
    }
    list(temp                  = as.double(temp),
         top_k                 = as.integer(top_k),
         top_p                 = as.double(top_p),
         min_p                 = as.double(min_p),
         typical_p             = as.double(typical_p),
         seed                  = as.integer(seed),
         min_keep              = as.integer(min_keep),
         repeat_penalty        = as.double(repeat_penalty),
         repeat_last_n         = as.integer(repeat_last_n),
         frequency_penalty     = as.double(frequency_penalty),
         presence_penalty      = as.double(presence_penalty),
         mirostat              = as.integer(mirostat),
         mirostat_tau          = as.double(mirostat_tau),
         mirostat_eta          = as.double(mirostat_eta),
         dynatemp_range        = as.double(dynatemp_range),
         dynatemp_exponent     = as.double(dynatemp_exponent),
         xtc_probability       = as.double(xtc_probability),
         xtc_threshold         = as.double(xtc_threshold),
         top_n_sigma           = as.double(top_n_sigma),
         dry_multiplier        = as.double(dry_multiplier),
         dry_base              = as.double(dry_base),
         dry_allowed_length    = as.integer(dry_allowed_length),
         dry_penalty_last_n    = as.integer(dry_penalty_last_n),
         dry_sequence_breakers = as.character(dry_sequence_breakers),
         adaptive_target       = as.double(adaptive_target),
         adaptive_decay        = as.double(adaptive_decay),
         infill                = as.logical(infill),
         logit_bias            = logit_bias)
}

# Resolve the sampler argument of a generation function. When `sampler` is
# supplied it wins outright; otherwise the flat per-call arguments (kept for
# backward compatibility) are folded into a parameter list. `flat` holds the
# caller's flat arguments, already named as llama_sampler_params() expects.
#
# A chain handle from the chain API passes straight through: the C side copies
# it for the generation rather than using it in place.
llamar_resolve_sampler <- function(sampler, flat) {
    if (!is.null(sampler)) {
        if (inherits(sampler, "llama_sampler_chain")) {
            return(sampler)
        }
        if (inherits(sampler, "llama_sampler")) {
            stop("sampler must be a chain, not a single sampler; ",
                 "add it to a chain with llama_sampler_chain_add()")
        }
        if (!is.list(sampler)) {
            stop("sampler must be a parameter list from llama_sampler_params() ",
                 "or a chain from llama_sampler_chain_new()")
        }
        return(sampler)
    }
    do.call(llama_sampler_params, flat)
}

# ============================================================
# Sampler chain API
# ============================================================

#' Create an empty sampler chain
#'
#' Builds a chain by hand, one sampler at a time, as the imperative counterpart
#' to the declarative [llama_sampler_params]. Add samplers with
#' [llama_sampler_chain_add], inspect them with [llama_sampler_chain_get] and
#' [llama_sampler_chain_n], and take them back out with
#' [llama_sampler_chain_remove].
#'
#' \strong{Ownership.} A chain takes ownership of every sampler added to it and
#' frees them when the chain itself is freed. A handle whose sampler has been
#' taken over this way cannot be freed on its own, and using it after its chain
#' is gone raises an error rather than crashing. [llama_sampler_chain_remove]
#' hands ownership back to R, which also retires any older handle to that same
#' sampler.
#'
#' Chains are freed by the garbage collector, so [llama_sampler_free] is
#' optional.
#'
#' @param no_perf If TRUE, skip the chain's own performance measurements.
#'   \code{NULL} (default) keeps llama.cpp's default.
#' @return An external pointer of class \code{"llama_sampler_chain"}.
#' @seealso [llama_sampler_new], [llama_sampler_chain_add],
#'   [llama_sampler_chain_from_params]
#' @export
#' @examples
#' \dontrun{
#' chain <- llama_sampler_chain_new()
#' llama_sampler_chain_add(chain, llama_sampler_new("top_k", top_k = 40L))
#' llama_sampler_chain_add(chain, llama_sampler_new("temp", temp = 0.7))
#' llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 42L))
#' llama_sampler_chain_n(chain)  # 3
#' }
llama_sampler_chain_new <- function(no_perf = NULL) {
    .Call("r_llama_sampler_chain_new",
          if (is.null(no_perf)) NULL else as.logical(no_perf))
}

#' Create a single sampler
#'
#' Builds one standalone sampler, to be added to a chain with
#' [llama_sampler_chain_add]. Parameters are taken from the named arguments in
#' \code{...}, using the same names and defaults as [llama_sampler_params], so
#' a sampler built here behaves exactly like its counterpart in the declarative
#' chain.
#'
#' Recognized kinds:
#' \itemize{
#'   \item Selection: \code{"greedy"}, \code{"dist"}, \code{"adaptive_p"},
#'     \code{"mirostat"}, \code{"mirostat_v2"}
#'   \item Truncation: \code{"top_k"}, \code{"top_p"}, \code{"min_p"},
#'     \code{"typical"}, \code{"top_n_sigma"}, \code{"xtc"}
#'   \item Temperature: \code{"temp"}, \code{"temp_ext"}
#'   \item Penalties: \code{"penalties"}, \code{"dry"}
#'   \item Other: \code{"logit_bias"}, \code{"infill"}
#' }
#'
#' \code{"mirostat"}, \code{"dry"}, \code{"logit_bias"} and \code{"infill"} read
#' the model's vocabulary, so they require \code{model}.
#'
#' @param kind Character scalar naming the sampler; see the list above.
#' @param ... Named sampler parameters, as accepted by [llama_sampler_params].
#' @param model Model handle from [llama_load_model], required by the kinds
#'   noted above and ignored by the rest.
#' @return An external pointer of class \code{"llama_sampler"}.
#' @seealso [llama_sampler_chain_new], [llama_sampler_chain_add]
#' @export
#' @examples
#' \dontrun{
#' s <- llama_sampler_new("top_k", top_k = 40L)
#' llama_sampler_name(s)  # "top-k"
#'
#' # kinds that need the vocabulary
#' d <- llama_sampler_new("dry", dry_multiplier = 0.8, model = model)
#' }
llama_sampler_new <- function(kind, ..., model = NULL) {
    stopifnot(is.character(kind), length(kind) == 1)
    args <- list(...)
    if (length(args) && is.null(names(args))) {
        stop("sampler parameters in `...` must be named")
    }
    .Call("r_llama_sampler_new", kind,
          do.call(llama_sampler_params, args), model)
}

#' Build a sampler chain from a parameter list
#'
#' Assembles the same chain the generation functions build internally from a
#' [llama_sampler_params] list, but hands it back for inspection or adjustment.
#' Use it to see which samplers a given parameter list actually produces, and in
#' what order.
#'
#' @param ctx Context handle returned by [llama_new_context]. The chain reads
#'   the model's vocabulary, so it is tied to this context's model.
#' @param params A parameter list from [llama_sampler_params].
#' @param grammar GBNF grammar string, or \code{NULL}.
#' @param trigger_patterns,trigger_tokens Lazy-grammar triggers; see
#'   [llama_generate].
#' @return An external pointer of class \code{"llama_sampler_chain"}.
#' @seealso [llama_sampler_params], [llama_sampler_chain_new]
#' @export
#' @examples
#' \dontrun{
#' sp <- llama_sampler_params(temp = 0.8, dry_multiplier = 0.8)
#' chain <- llama_sampler_chain_from_params(ctx, sp)
#' vapply(seq_len(llama_sampler_chain_n(chain)) - 1L,
#'        function(i) llama_sampler_name(llama_sampler_chain_get(chain, i)),
#'        character(1))
#' }
llama_sampler_chain_from_params <- function(ctx, params, grammar = NULL,
                                            trigger_patterns = NULL,
                                            trigger_tokens = NULL) {
    stopifnot(is.list(params))
    .Call("r_llama_sampler_chain_from_params", ctx, params, grammar,
          if (is.null(trigger_patterns)) NULL else as.character(trigger_patterns),
          if (is.null(trigger_tokens)) NULL else as.integer(trigger_tokens))
}

#' Add a sampler to a chain
#'
#' The chain takes ownership of \code{sampler}: it is freed together with the
#' chain, cannot be freed on its own, and cannot be added to a second chain.
#' The handle stays usable for inspection while the chain lives.
#'
#' @param chain A chain from [llama_sampler_chain_new] or
#'   [llama_sampler_chain_from_params].
#' @param sampler A sampler from [llama_sampler_new], not yet owned by any chain.
#' @return \code{chain}, invisibly, so calls can be piped.
#' @seealso [llama_sampler_chain_remove], [llama_sampler_chain_n]
#' @export
llama_sampler_chain_add <- function(chain, sampler) {
    .Call("r_llama_sampler_chain_add", chain, sampler)
    invisible(chain)
}

#' Number of samplers in a chain
#'
#' @param chain A sampler chain.
#' @return Integer count of samplers currently in the chain.
#' @seealso [llama_sampler_chain_get]
#' @export
llama_sampler_chain_n <- function(chain) {
    .Call("r_llama_sampler_chain_n", chain)
}

#' Get a sampler out of a chain
#'
#' Returns a \emph{borrowed} handle: the chain keeps ownership, so the returned
#' sampler must not be freed and stops being usable once the chain is gone.
#' Use [llama_sampler_chain_remove] to take a sampler out for keeps.
#'
#' @param chain A sampler chain.
#' @param i Zero-based index of the sampler, matching llama.cpp's own indexing.
#'   \code{-1} returns the chain itself, as a borrowed handle --- useful only to
#'   confirm that \code{chain} really is a chain.
#' @return An external pointer of class \code{"llama_sampler"}.
#' @seealso [llama_sampler_chain_n], [llama_sampler_chain_remove]
#' @export
llama_sampler_chain_get <- function(chain, i) {
    .Call("r_llama_sampler_chain_get", chain, as.integer(i))
}

#' Remove a sampler from a chain
#'
#' Detaches the sampler at index \code{i} and hands ownership back to R, so the
#' returned handle is freed on its own (by the garbage collector, or by
#' [llama_sampler_free]). Any handle obtained earlier for that same sampler ---
#' from [llama_sampler_chain_add] or [llama_sampler_chain_get] --- is retired by
#' this call and raises an error if used afterwards.
#'
#' @param chain A sampler chain.
#' @param i Zero-based index of the sampler to remove.
#' @return An external pointer of class \code{"llama_sampler"}, now owned by R.
#' @seealso [llama_sampler_chain_add], [llama_sampler_chain_get]
#' @export
llama_sampler_chain_remove <- function(chain, i) {
    .Call("r_llama_sampler_chain_remove", chain, as.integer(i))
}

#' Name of a sampler
#'
#' @param sampler A sampler or sampler chain.
#' @return A character scalar with llama.cpp's name for the sampler, e.g.
#'   \code{"top-k"}, or \code{""} when it has none.
#' @export
llama_sampler_name <- function(sampler) {
    .Call("r_llama_sampler_name", sampler)
}

#' Reset a sampler's internal state
#'
#' Clears whatever state a sampler carries between tokens --- Mirostat's
#' \code{mu}, adaptive-p's moving average, the penalty samplers' token history.
#' Applied to a chain, it resets every sampler in it.
#'
#' @param sampler A sampler or sampler chain.
#' @return \code{NULL}, invisibly.
#' @export
llama_sampler_reset <- function(sampler) {
    .Call("r_llama_sampler_reset", sampler)
    invisible(NULL)
}

#' Copy a sampler
#'
#' Returns an independent copy, including any accumulated state, owned by R.
#' Cloning a chain clones every sampler in it. Not every sampler supports
#' cloning; those that do not raise an error.
#'
#' @param sampler A sampler or sampler chain.
#' @return A new external pointer of the same class as \code{sampler}.
#' @export
llama_sampler_clone <- function(sampler) {
    .Call("r_llama_sampler_clone", sampler)
}

#' Feed a token to a sampler
#'
#' Advances the samplers that track generation history --- the penalty samplers,
#' DRY, grammar --- without sampling anything. Only needed when driving a
#' sampler by hand; the generation functions do this themselves.
#'
#' @param sampler A sampler or sampler chain.
#' @param token Integer token ID.
#' @return \code{NULL}, invisibly.
#' @export
llama_sampler_accept <- function(sampler, token) {
    .Call("r_llama_sampler_accept", sampler, as.integer(token))
    invisible(NULL)
}

#' Seed used by a sampler
#'
#' @param sampler A sampler or sampler chain. For a chain, the seed of the first
#'   seeded sampler in it.
#' @return The seed as an integer, or \code{NA_integer_} when the sampler has no
#'   seed of its own (or the seed does not fit in an R integer).
#' @export
llama_sampler_get_seed <- function(sampler) {
    .Call("r_llama_sampler_get_seed", sampler)
}

#' Free a sampler or chain
#'
#' Releases the sampler immediately instead of waiting for the garbage
#' collector, which frees it anyway. Freeing a chain also frees every sampler
#' inside it, and handles to those samplers raise an error afterwards rather
#' than reaching freed memory. A sampler a chain has taken over cannot be freed
#' on its own --- free the chain instead.
#'
#' Freeing an already-freed handle does nothing.
#'
#' @param sampler A sampler or sampler chain.
#' @return \code{NULL}, invisibly.
#' @export
llama_sampler_free <- function(sampler) {
    .Call("r_llama_sampler_free", sampler)
    invisible(NULL)
}

#' Generate text from a prompt
#'
#' Tokenizes the prompt, runs the full autoregressive decode loop with sampling,
#' and returns the generated text (excluding the original prompt).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param prompt Character string prompt
#' @param max_new_tokens Maximum number of tokens to generate
#' @param temp Sampling temperature. 0 = greedy decoding.
#' @param top_k Top-K filtering (0 = disabled)
#' @param top_p Top-P (nucleus) filtering (1.0 = disabled)
#' @param seed Random seed for sampling
#' @param min_p Min-P filtering threshold (0.0 = disabled)
#' @param typical_p Locally typical sampling threshold (1.0 = disabled)
#' @param repeat_penalty Repetition penalty (1.0 = disabled)
#' @param repeat_last_n Number of last tokens to penalize (0 = disabled, -1 = context size)
#' @param frequency_penalty Frequency penalty (0.0 = disabled)
#' @param presence_penalty Presence penalty (0.0 = disabled)
#' @param mirostat Mirostat sampling mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
#' @param mirostat_tau Mirostat target entropy (tau parameter)
#' @param mirostat_eta Mirostat learning rate (eta parameter)
#' @param grammar GBNF grammar string for constrained generation (NULL = disabled)
#' @param trigger_patterns Character vector of regular expressions that lazily
#'   activate a grammar (e.g. the prefix a model emits before a tool call). Only
#'   meaningful alongside a lazy \code{grammar}; \code{NULL} (default) applies the
#'   grammar from the first token. Supplied by \code{\link{llama_chat_build}}.
#' @param trigger_tokens Integer vector of token IDs that lazily activate the
#'   grammar, the token-level counterpart to \code{trigger_patterns}. \code{NULL}
#'   (default) means none.
#' @param sampler Either a sampler-parameter list from
#'   \code{\link{llama_sampler_params}}, or a chain built by hand with
#'   \code{\link{llama_sampler_chain_new}}. When supplied it takes precedence
#'   over the individual sampling arguments above, and a parameter list is the
#'   only way to reach the samplers that have no argument here (DRY, XTC,
#'   dynamic temperature, top-n-sigma, logit bias, infill, adaptive-p).
#'   \code{NULL} (default) builds the chain from the individual arguments.
#'
#'   A supplied chain is \strong{copied} for the generation, so the caller's
#'   chain is left untouched and its lifetime is its own concern.
#' @param sampler_reset Only meaningful when \code{sampler} is a chain. If
#'   \code{TRUE} (default), the copy starts with its accumulated state cleared
#'   --- Mirostat's \code{mu}, adaptive-p's moving average, the penalty
#'   samplers' token history --- so repeated calls with one chain behave
#'   identically, as they do on the parameter-list path. Set \code{FALSE} to
#'   carry that state into the generation and deliberately continue where an
#'   earlier one left off.
#' @param with_timings If TRUE, attach a named numeric vector of per-stage
#'   timings (in ms) as attribute "timings" of the returned text. Stages:
#'   tokenize, build_sampler, kv_clear, prefill_dispatch, prefill_sync,
#'   gpu_sync (cumulative across decode-loop iterations), sample (cumulative),
#'   decode_dispatch (cumulative), detokenize, plus n_iterations and t_total_ms.
#'   Adds llama_synchronize calls inside the loop, so it is intended for
#'   profiling and may slightly slow generation.
#' @return A character scalar containing the generated text (excluding the
#'   original prompt).
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L)
#' ctx <- llama_new_context(model, n_ctx = 2048L)
#'
#' # Basic generation
#' result <- llama_generate(ctx, "Once upon a time")
#' cat(result)
#'
#' # Greedy decoding (deterministic)
#' result <- llama_generate(ctx, "The answer is", temp = 0)
#'
#' # More creative output
#' result <- llama_generate(ctx, "Write a poem about R:",
#'                          max_new_tokens = 100L,
#'                          temp = 1.0, top_p = 0.95)
#'
#' # With repetition penalty
#' result <- llama_generate(ctx, "List items:",
#'                          repeat_penalty = 1.1, repeat_last_n = 64L)
#'
#' # JSON output with grammar
#' result <- llama_generate(ctx, "Output JSON:",
#'                          grammar = 'root ::= "{" "}" ')
#' }
llama_generate <- function(ctx, prompt, max_new_tokens = 256L,
                           temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L,
                           min_p = 0.0, typical_p = 1.0,
                           repeat_penalty = 1.0, repeat_last_n = 64L,
                           frequency_penalty = 0.0, presence_penalty = 0.0,
                           mirostat = 0L, mirostat_tau = 5.0, mirostat_eta = 0.1,
                           grammar = NULL, with_timings = FALSE,
                           trigger_patterns = NULL, trigger_tokens = NULL,
                           sampler = NULL, sampler_reset = TRUE) {
    stopifnot(is.character(prompt), length(prompt) == 1)
    sp <- llamar_resolve_sampler(sampler, list(
        temp = temp, top_k = top_k, top_p = top_p, seed = seed,
        min_p = min_p, typical_p = typical_p,
        repeat_penalty = repeat_penalty, repeat_last_n = repeat_last_n,
        frequency_penalty = frequency_penalty, presence_penalty = presence_penalty,
        mirostat = mirostat, mirostat_tau = mirostat_tau,
        mirostat_eta = mirostat_eta))
    .Call("r_llama_generate", ctx, prompt,
          as.integer(max_new_tokens), sp,
          grammar, as.logical(with_timings),
          if (is.null(trigger_patterns)) NULL else as.character(trigger_patterns),
          if (is.null(trigger_tokens)) NULL else as.integer(trigger_tokens),
          as.logical(sampler_reset))
}

#' Begin a streaming (token-by-token) generation
#'
#' Sets up sampling and prefills the prompt, returning an opaque state handle
#' that is pulled one chunk at a time with [llama_gen_next]. This is the
#' streaming counterpart to [llama_generate]: same sampler chain and the same
#' output for a given seed, but text arrives incrementally so it can be pushed
#' into an SSE stream as it is produced.
#'
#' Typical loop:
#' \preformatted{
#' st <- llama_gen_begin(ctx, prompt)
#' repeat {
#'   chunk <- llama_gen_next(st)
#'   if (is.null(chunk)) break
#'   cat(chunk)
#' }
#' cat(llama_gen_end(st))  # flush any held-back trailing bytes
#' }
#'
#' Only one streaming generation may be active per context at a time: each
#' call to \code{llama_gen_begin} clears the context KV cache.
#'
#' @inheritParams llama_generate
#' @return An external pointer holding the generation state. Pass it to
#'   [llama_gen_next] and [llama_gen_end]. The underlying sampler is freed
#'   automatically by the garbage collector.
#' @seealso [llama_gen_next], [llama_gen_end], [llama_generate]
#' @export
llama_gen_begin <- function(ctx, prompt, max_new_tokens = 256L,
                            temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L,
                            min_p = 0.0, typical_p = 1.0,
                            repeat_penalty = 1.0, repeat_last_n = 64L,
                            frequency_penalty = 0.0, presence_penalty = 0.0,
                            mirostat = 0L, mirostat_tau = 5.0, mirostat_eta = 0.1,
                            grammar = NULL,
                            trigger_patterns = NULL, trigger_tokens = NULL,
                            sampler = NULL, sampler_reset = TRUE) {
    stopifnot(is.character(prompt), length(prompt) == 1)
    sp <- llamar_resolve_sampler(sampler, list(
        temp = temp, top_k = top_k, top_p = top_p, seed = seed,
        min_p = min_p, typical_p = typical_p,
        repeat_penalty = repeat_penalty, repeat_last_n = repeat_last_n,
        frequency_penalty = frequency_penalty, presence_penalty = presence_penalty,
        mirostat = mirostat, mirostat_tau = mirostat_tau,
        mirostat_eta = mirostat_eta))
    .Call("r_llama_gen_begin", ctx, prompt,
          as.integer(max_new_tokens), sp, grammar,
          if (is.null(trigger_patterns)) NULL else as.character(trigger_patterns),
          if (is.null(trigger_tokens)) NULL else as.integer(trigger_tokens),
          as.logical(sampler_reset))
}

#' Begin streaming generation from an already-prefilled context
#'
#' Like [llama_gen_begin], but does \strong{not} tokenize a prompt or clear the
#' KV cache. Use it to continue generation after [llama_image_eval] (or any
#' other code that has already decoded tokens into the context), so the
#' multimodal prefill is preserved. Sampling continues from the context's last
#' logits; pull tokens with [llama_gen_next] and flush with [llama_gen_end] as
#' usual.
#'
#' @param ctx A llama context whose KV cache has already been populated (e.g. by
#'   [llama_image_eval]).
#' @param n_past Starting KV position, as returned by [llama_image_eval]. Kept
#'   for clarity/symmetry; the context already tracks its own position.
#' @inheritParams llama_gen_begin
#' @return An external pointer holding the generation state (see
#'   [llama_gen_begin]).
#' @seealso [llama_image_eval], [llama_gen_next], [llama_gen_end]
#' @export
llama_gen_begin_at <- function(ctx, n_past, max_new_tokens = 256L,
                               temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L,
                               min_p = 0.0, typical_p = 1.0,
                               repeat_penalty = 1.0, repeat_last_n = 64L,
                               frequency_penalty = 0.0, presence_penalty = 0.0,
                               mirostat = 0L, mirostat_tau = 5.0, mirostat_eta = 0.1,
                               grammar = NULL,
                               trigger_patterns = NULL, trigger_tokens = NULL,
                               sampler = NULL, sampler_reset = TRUE) {
    stopifnot(inherits(ctx, "externalptr"))
    sp <- llamar_resolve_sampler(sampler, list(
        temp = temp, top_k = top_k, top_p = top_p, seed = seed,
        min_p = min_p, typical_p = typical_p,
        repeat_penalty = repeat_penalty, repeat_last_n = repeat_last_n,
        frequency_penalty = frequency_penalty, presence_penalty = presence_penalty,
        mirostat = mirostat, mirostat_tau = mirostat_tau,
        mirostat_eta = mirostat_eta))
    .Call("r_llama_gen_begin_at", ctx, as.integer(n_past),
          as.integer(max_new_tokens), sp, grammar,
          if (is.null(trigger_patterns)) NULL else as.character(trigger_patterns),
          if (is.null(trigger_tokens)) NULL else as.integer(trigger_tokens),
          as.logical(sampler_reset))
}

#' Pull the next chunk of a streaming generation
#'
#' Advances a generation started with [llama_gen_begin] by one token and
#' returns the next chunk of decoded text. A possibly-incomplete trailing
#' UTF-8 character is held back until enough bytes arrive, so every returned
#' chunk is valid UTF-8 (the chunk may be \code{""} when the only new byte is
#' part of an unfinished character).
#'
#' @param state Generation state handle from [llama_gen_begin].
#' @return A length-1 UTF-8 character vector with the next chunk, or
#'   \code{NULL} when generation has finished (end-of-generation token reached
#'   or \code{max_new_tokens} exhausted). After \code{NULL}, call
#'   [llama_gen_end] to flush any remaining bytes.
#' @seealso [llama_gen_begin], [llama_gen_end]
#' @export
llama_gen_next <- function(state) {
    .Call("r_llama_gen_next", state)
}

#' Finish a streaming generation
#'
#' Marks the generation done and returns any bytes still held in the internal
#' UTF-8 carry buffer (the tail of an unfinished character, if generation
#' stopped mid-character). Concatenating every [llama_gen_next] chunk followed
#' by the \code{llama_gen_end} result reproduces the full [llama_generate]
#' output for the same seed and parameters. Safe to call more than once.
#'
#' @param state Generation state handle from [llama_gen_begin].
#' @return A length-1 UTF-8 character vector with any remaining buffered text
#'   (often \code{""}).
#' @seealso [llama_gen_begin], [llama_gen_next]
#' @export
llama_gen_end <- function(state) {
    .Call("r_llama_gen_end", state)
}

#' Generate completions for multiple prompts in parallel
#'
#' Runs continuous batching: all prompts share the same decode loop, so each
#' iteration dispatches one matmul over all still-running sequences. This
#' converts decode from memory-bound vector ops into compute-bound matrix ops
#' on the GPU and lifts throughput compared to calling \code{\link{llama_generate}}
#' in a loop.
#'
#' The context must be created with \code{n_seq_max >= length(prompts)} and
#' \code{n_ctx} large enough to hold every prompt plus its generated tokens
#' simultaneously. As a rule of thumb:
#' \code{n_ctx >= sum(prompt_lengths) + length(prompts) * max_new_tokens}.
#'
#' Each sequence gets its own sampler chain seeded with \code{seed + seq_index},
#' so identical prompts still produce diverse outputs at \code{temp > 0}
#' (useful for self-consistency sampling). Sampler hyperparameters are shared
#' across sequences in this version.
#'
#' Stop conditions per sequence: end-of-generation token (model-defined) or
#' \code{max_new_tokens} reached. \code{with_timings} is not supported here —
#' use \code{\link{llama_generate}} for that.
#'
#' @param ctx Context handle returned by [llama_new_context], created with
#'   sufficient \code{n_seq_max} and \code{n_ctx} (see Details).
#' @param prompts Character vector of prompts, one per parallel sequence.
#' @param max_new_tokens,temp,top_k,top_p,seed,min_p,typical_p,repeat_penalty,repeat_last_n,frequency_penalty,presence_penalty,mirostat,mirostat_tau,mirostat_eta,grammar,trigger_patterns,trigger_tokens,sampler
#'   Sampling parameters; see \code{\link{llama_generate}}. Shared across
#'   sequences. \code{seed} is offset per sequence (\code{seed + s}), so each
#'   sequence samples independently. For that reason \code{sampler} accepts only
#'   a parameter list here, not a chain: every sequence needs its own differently
#'   seeded chain, which one supplied chain cannot provide. Passing a chain is
#'   an error rather than being silently ignored.
#' @return A list of length \code{length(prompts)}, in the same order as the
#'   input. Each element is a list with fields:
#'   \itemize{
#'     \item \code{text}: character scalar with the generated text
#'     \item \code{n_tokens}: integer count of tokens generated
#'     \item \code{finished_reason}: \code{"eos"} or \code{"max_tokens"}
#'   }
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L)
#' # 4 parallel sequences, up to 256 new tokens each
#' ctx <- llama_new_context(model, n_ctx = 4096L, n_seq_max = 4L,
#'                          flash_attn = "on")
#'
#' # Batch classification
#' prompts <- c("Classify: 'great movie' as positive/negative.",
#'              "Classify: 'awful service' as positive/negative.",
#'              "Classify: 'just okay' as positive/negative.",
#'              "Classify: 'loved every minute' as positive/negative.")
#' out <- llama_generate_batch(ctx, prompts, max_new_tokens = 16L, temp = 0)
#' vapply(out, `[[`, character(1), "text")
#'
#' # Self-consistency sampling: same prompt repeated
#' samples <- llama_generate_batch(ctx, rep("2 + 2 =", 4L),
#'                                 max_new_tokens = 8L, temp = 0.7)
#' }
llama_generate_batch <- function(ctx, prompts, max_new_tokens = 256L,
                                 temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L,
                                 min_p = 0.0, typical_p = 1.0,
                                 repeat_penalty = 1.0, repeat_last_n = 64L,
                                 frequency_penalty = 0.0, presence_penalty = 0.0,
                                 mirostat = 0L, mirostat_tau = 5.0,
                                 mirostat_eta = 0.1, grammar = NULL,
                                 trigger_patterns = NULL, trigger_tokens = NULL,
                                 sampler = NULL) {
    stopifnot(is.character(prompts), length(prompts) >= 1)
    sp <- llamar_resolve_sampler(sampler, list(
        temp = temp, top_k = top_k, top_p = top_p, seed = seed,
        min_p = min_p, typical_p = typical_p,
        repeat_penalty = repeat_penalty, repeat_last_n = repeat_last_n,
        frequency_penalty = frequency_penalty, presence_penalty = presence_penalty,
        mirostat = mirostat, mirostat_tau = mirostat_tau,
        mirostat_eta = mirostat_eta))
    .Call("r_llama_generate_batch", ctx, prompts,
          as.integer(max_new_tokens), sp, grammar,
          if (is.null(trigger_patterns)) NULL else as.character(trigger_patterns),
          if (is.null(trigger_tokens)) NULL else as.integer(trigger_tokens))
}

#' Extract embeddings for a text
#'
#' Runs the model in embeddings mode and returns the hidden-state vector
#' of the last token. Note: meaningful only for models that support embeddings.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param text Character string to embed
#' @return A numeric vector of length \code{n_embd} (the model's embedding
#'   dimension) containing the hidden-state representation of the input text.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' emb1 <- llama_embeddings(ctx, "Hello world")
#' emb2 <- llama_embeddings(ctx, "Hi there")
#'
#' # Cosine similarity
#' similarity <- sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
#' cat("Similarity:", similarity, "\n")
#' }
llama_embeddings <- function(ctx, text) {
    stopifnot(is.character(text), length(text) == 1)
    .Call("r_llama_embeddings", ctx, text)
}

#' Batch embeddings for multiple texts
#'
#' Computes embeddings for a character vector of texts in a single decode pass
#' using per-sequence pooling. This is more efficient than calling
#' \code{\link{llama_embeddings}} in a loop when embedding many texts.
#'
#' @details Requires a model that supports pooled embeddings (e.g. embedding
#'   models like nomic-embed, bge, etc.). The context must have enough capacity
#'   for the total number of tokens across all texts. Causal attention is
#'   automatically disabled during computation.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param texts Character vector of texts to embed
#' @return A numeric matrix with \code{nrow = length(texts)} and
#'   \code{ncol = n_embd}.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("embedding-model.gguf")
#' ctx <- llama_new_context(model, n_ctx = 2048L)
#' llama_set_causal_attn(ctx, FALSE)
#'
#' mat <- llama_embed_batch(ctx, c("hello world", "foo bar", "test"))
#' # mat is a 3 x n_embd matrix
#' }
llama_embed_batch <- function(ctx, texts) {
    stopifnot(is.character(texts))
    .Call("r_llama_embed_batch", ctx, texts)
}

#' Get embeddings for the i-th token in the batch
#'
#' Returns the embedding vector for a specific token position after a decode
#' call with embeddings enabled. Negative indices count from the end
#' (-1 = last token).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param i Integer index of the token (0-based, or negative for reverse indexing)
#' @return A numeric vector of length \code{n_embd}.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' llama_generate(ctx, "Hello world", max_new_tokens = 1L)
#'
#' # Get the embedding of the last decoded token
#' emb <- llama_get_embeddings_ith(ctx, -1L)
#' cat("Embedding dim:", length(emb), "\n")
#' }
llama_get_embeddings_ith <- function(ctx, i) {
    .Call("r_llama_get_embeddings_ith", ctx, as.integer(i))
}

#' Get pooled embeddings for a sequence
#'
#' Returns the pooled embedding vector for a given sequence ID after a batch
#' decode. Only works when the model supports pooling (embedding models).
#'
#' @param ctx Context handle returned by [llama_new_context] with
#'   \code{embedding = TRUE}
#' @param seq_id Integer sequence ID (0-based)
#' @return A numeric vector of length \code{n_embd}.
#' @export
#' @examples
#' \dontrun{
#' # Get pooled embedding for sequence 0 (requires embedding context)
#' model <- llama_load_model("nomic-embed.gguf")
#' ctx <- llama_new_context(model, embedding = TRUE)
#' mat <- llama_embed_batch(ctx, "Hello world")
#' emb <- llama_get_embeddings_seq(ctx, 0L)
#' cat("Pooled embedding dim:", length(emb), "\n")
#' }
llama_get_embeddings_seq <- function(ctx, seq_id) {
    .Call("r_llama_get_embeddings_seq", ctx, as.integer(seq_id))
}

#' Get all output token embeddings as a matrix
#'
#' Returns a matrix of shape \code{n_outputs × n_embd} containing the raw
#' embedding vectors for all tokens whose \code{logits} flag was set in the batch.
#' Only works when \code{pooling_type == "none"} (generative models or embedding
#' contexts without pooling). For pooled embeddings use [llama_get_embeddings_seq].
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param n_outputs Number of outputs requested in the last decode call
#'   (i.e. how many tokens had \code{logits = TRUE} in the batch).
#' @return A numeric matrix with \code{n_outputs} rows and \code{n_embd} columns.
#' @export
llama_get_embeddings <- function(ctx, n_outputs) {
    .Call("r_llama_get_embeddings", ctx, as.integer(n_outputs))
}

# ============================================================
# Chat templates
# ============================================================

#' Get model's built-in chat template
#'
#' Returns the chat template string embedded in the model file, if any.
#' Common templates include ChatML, Llama, Mistral, etc.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param name Optional template name (NULL for default)
#' @return A character scalar with the chat template string, or \code{NULL} if
#'   the model does not contain a built-in template.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("llama-3.2-instruct.gguf")
#' tmpl <- llama_chat_template(model)
#' cat(tmpl)
#' }
llama_chat_template <- function(model, name = NULL) {
    .Call("r_llama_chat_template", model, name)
}

#' Apply chat template to messages
#'
#' Formats a conversation using the specified chat template.
#' This is essential for instruct/chat models to work correctly.
#'
#' @param messages List of messages, each with `role` and `content` elements.
#'   Roles are typically "system", "user", "assistant".
#' @param template Template string (from [llama_chat_template]) or NULL to use default
#' @param add_generation_prompt Whether to add the assistant prompt prefix at the end
#' @return A character scalar containing the formatted prompt string, ready
#'   to be passed to \code{\link{llama_generate}}.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("llama-3.2-instruct.gguf")
#' tmpl <- llama_chat_template(model)
#'
#' messages <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "What is R?")
#' )
#'
#' prompt <- llama_chat_apply_template(messages, template = tmpl)
#' cat(prompt)
#'
#' ctx <- llama_new_context(model)
#' response <- llama_generate(ctx, prompt)
#' }
llama_chat_apply_template <- function(messages, template = NULL, add_generation_prompt = TRUE) {
    stopifnot(is.list(messages))
    .Call("r_llama_chat_apply_template", template, messages, as.logical(add_generation_prompt))
}

#' Build a tool-aware chat prompt and its parsing grammar
#'
#' Applies the model's built-in chat template (via llama.cpp's Jinja path,
#' i.e. \code{common_chat_templates_apply}) to a conversation that may include
#' tool definitions, returning both the formatted \code{prompt} and the
#' \code{grammar} that constrains the model to emit tool calls in the format
#' that template expects. Unlike \code{\link{llama_chat_apply_template}} (which
#' uses the low-level template path and is text-only), this understands
#' \code{tools} and is what \code{\link{llama_serve_anthropic}} uses to support
#' tool calling.
#'
#' Pair the returned \code{format} with \code{\link{llama_chat_parse}} to turn
#' the model's raw output back into content and structured tool calls.
#'
#' @param model A model handle from \code{\link{llama_load_model}}.
#' @param messages List of messages in OpenAI chat shape: each a list with
#'   \code{role} and \code{content} (content may be a string or a list of
#'   content parts). Assistant tool calls and tool results are supported via
#'   the usual \code{tool_calls} / \code{tool_call_id} fields.
#' @param tools Optional list of tool definitions in OpenAI shape (each a list
#'   with \code{type = "function"} and a \code{function} entry holding
#'   \code{name}, \code{description}, \code{parameters}). \code{NULL} for a
#'   plain chat prompt.
#' @param tool_choice One of \code{"auto"}, \code{"required"}, \code{"none"},
#'   or \code{NULL} to leave it at the template default.
#' @param json_schema Optional JSON-schema string to constrain free-form output
#'   (structured output). Mutually meaningful with \code{tools = NULL}.
#' @param add_generation_prompt Whether to append the assistant prompt prefix.
#' @param enable_thinking Whether to allow reasoning blocks for models that
#'   support them.
#' @return A list with elements \code{prompt}, \code{grammar} (possibly empty),
#'   \code{format} (integer format id for \code{\link{llama_chat_parse}}),
#'   \code{grammar_lazy}, \code{additional_stops}, and \code{preserved_tokens}.
#' @seealso [llama_chat_parse], [llama_serve_anthropic]
#' @export
llama_chat_build <- function(model, messages, tools = NULL, tool_choice = NULL,
                             json_schema = NULL, add_generation_prompt = TRUE,
                             enable_thinking = TRUE) {
    stopifnot(is.list(messages))
    messages_json <- jsonlite::toJSON(messages, auto_unbox = TRUE, null = "null")
    tools_json <- if (is.null(tools) || length(tools) == 0) NULL
                  else jsonlite::toJSON(tools, auto_unbox = TRUE, null = "null")
    .Call("r_llama_chat_build",
          model,
          as.character(messages_json),
          if (is.null(tools_json)) NULL else as.character(tools_json),
          if (is.null(tool_choice)) NULL else as.character(tool_choice),
          if (is.null(json_schema)) NULL else as.character(json_schema),
          as.logical(add_generation_prompt),
          as.logical(enable_thinking))
}

#' Parse raw model output into content and tool calls
#'
#' Inverse of the formatting done by \code{\link{llama_chat_build}}: takes the
#' model's raw generated text and the \code{format} id returned by
#' \code{llama_chat_build()}, and extracts assistant \code{content}, any
#' \code{reasoning_content}, and structured tool calls (name, arguments JSON,
#' id) using llama.cpp's per-format parsers (\code{common_chat_parse}).
#'
#' @param input Raw generated text from the model.
#' @param format Integer format id as returned in
#'   \code{llama_chat_build()$format}.
#' @param is_partial Set \code{TRUE} when \code{input} is an incomplete stream
#'   prefix (enables partial/streaming-tolerant parsing).
#' @param parser Serialized PEG parser arena as returned in
#'   \code{llama_chat_build()$parser}. Required for PEG-based formats
#'   (\code{PEG_SIMPLE}/\code{PEG_NATIVE}/\code{PEG_CONSTRUCTED}, e.g. Mistral /
#'   Ministral); ignored otherwise. Pass \code{llama_chat_build()$parser}
#'   straight through.
#' @return A list with \code{content}, \code{reasoning_content}, and
#'   \code{tool_calls} — the latter a data frame with columns \code{name},
#'   \code{arguments} (a JSON string), and \code{id} (zero rows if none).
#' @seealso [llama_chat_build]
#' @export
llama_chat_parse <- function(input, format, is_partial = FALSE, parser = NULL) {
    res <- .Call("r_llama_chat_parse",
                 as.character(input), as.integer(format), as.logical(is_partial),
                 if (is.null(parser)) NULL else as.character(parser))
    list(
        content           = res$content,
        reasoning_content = res$reasoning_content,
        tool_calls = data.frame(
            name      = res$tool_names,
            arguments = res$tool_arguments,
            id        = res$tool_ids,
            stringsAsFactors = FALSE
        )
    )
}

# ============================================================
# LoRA adapters
# ============================================================

#' Load a LoRA adapter
#'
#' Loads a LoRA (Low-Rank Adaptation) adapter file that can be applied
#' to modify the model's behavior without changing the base weights.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param path Path to the LoRA adapter file (.gguf or .bin)
#' @return An external pointer (class \code{externalptr}) wrapping the loaded
#'   LoRA (Low-Rank Adaptation) adapter. Pass this handle to
#'   \code{\link{llama_lora_apply}} to activate the adapter.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("base-model.gguf")
#' lora <- llama_lora_load(model, "fine-tuned-adapter.gguf")
#'
#' ctx <- llama_new_context(model)
#' llama_lora_apply(ctx, lora, scale = 1.0)
#'
#' # Now generation uses the LoRA-modified model
#' result <- llama_generate(ctx, "Hello")
#' }
llama_lora_load <- function(model, path) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: LoRA file does not exist: ", path)
    .Call("r_llama_lora_load", model, path)
}

#' Apply a LoRA adapter to context
#'
#' Activates a loaded LoRA adapter for the given context.
#' Multiple LoRA adapters can be applied simultaneously.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param lora LoRA adapter handle from [llama_lora_load]
#' @param scale Scaling factor for the adapter (1.0 = full effect, 0.5 = half effect)
#' @return No return value, called for side effects. Activates the LoRA adapter
#'   for the given context.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("base-model.gguf")
#' lora <- llama_lora_load(model, "adapter.gguf")
#' ctx <- llama_new_context(model)
#'
#' # Apply with full strength
#' llama_lora_apply(ctx, lora, scale = 1.0)
#'
#' # Or apply with reduced effect
#' llama_lora_apply(ctx, lora, scale = 0.5)
#' }
llama_lora_apply <- function(ctx, lora, scale = 1.0) {
    .Call("r_llama_lora_apply", ctx, lora, as.double(scale))
    invisible(NULL)
}

#' Remove a LoRA adapter from context
#'
#' Deactivates a specific LoRA adapter from the context.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param lora LoRA adapter handle to remove
#' @return An integer scalar: 0 on success, -1 if the adapter was not applied
#'   to this context.
#' @export
#' @examples
#' \dontrun{
#' # Remove a specific adapter while keeping others active
#' llama_lora_remove(ctx, lora)
#' result <- llama_generate(ctx, "Without adapter: ", max_new_tokens = 20L)
#' }
llama_lora_remove <- function(ctx, lora) {
    .Call("r_llama_lora_remove", ctx, lora)
}

#' Remove all LoRA adapters from context
#'
#' Deactivates all LoRA adapters from the context, returning to base model behavior.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects. Removes all active LoRA
#'   adapters from the context.
#' @export
#' @examples
#' \dontrun{
#' # Apply multiple LoRAs
#' llama_lora_apply(ctx, lora1)
#' llama_lora_apply(ctx, lora2)
#'
#' # Remove all at once
#' llama_lora_clear(ctx)
#' }
llama_lora_clear <- function(ctx) {
    .Call("r_llama_lora_clear", ctx)
    invisible(NULL)
}

#' Read the metadata of a LoRA adapter
#'
#' `llama_lora_meta()` returns every GGUF metadata entry stored in the adapter;
#' `llama_lora_meta_val()` looks up a single key. This is the adapter-level
#' counterpart of [llama_model_meta] / [llama_model_meta_val].
#'
#' @param lora LoRA adapter handle returned by [llama_lora_load]
#' @param key A single string naming the metadata key to look up.
#' @return `llama_lora_meta()`: a named character vector, possibly empty.
#'   `llama_lora_meta_val()`: a character string, or `NULL` when the key is
#'   absent.
#' @name llama_lora_meta
#' @seealso [llama_lora_load], [llama_model_meta]
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' lora  <- llama_lora_load(model, "adapter.gguf")
#' llama_lora_meta(lora)
#' llama_lora_meta_val(lora, "general.name")
#' }
NULL

#' @rdname llama_lora_meta
#' @export
llama_lora_meta <- function(lora) {
    .Call("r_llama_lora_meta", lora)
}

#' @rdname llama_lora_meta
#' @export
llama_lora_meta_val <- function(lora, key) {
    stopifnot(is.character(key), length(key) == 1)
    .Call("r_llama_lora_meta_val", lora, key)
}

#' Invocation tokens of an activated LoRA (aLoRA)
#'
#' An activated LoRA carries a token sequence that switches it on: the adapter
#' only takes effect once those tokens appear in the context. Ordinary LoRAs
#' apply unconditionally and define no invocation tokens.
#'
#' @param lora LoRA adapter handle returned by [llama_lora_load]
#' @return An integer vector of token IDs, or `NULL` when the adapter is an
#'   ordinary LoRA.
#' @seealso [llama_lora_load], [llama_detokenize]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' lora  <- llama_lora_load(model, "alora.gguf")
#' toks  <- llama_lora_alora_invocation_tokens(lora)
#' if (!is.null(toks)) llama_detokenize(model, toks)
#' }
llama_lora_alora_invocation_tokens <- function(lora) {
    .Call("r_llama_lora_alora_invocation_tokens", lora)
}

#' Apply a control vector to a context
#'
#' A control vector steers generation by adding a fixed direction to the
#' residual stream of a layer range. Unlike a LoRA it is applied directly to the
#' context and needs no adapter file.
#'
#' `data` holds the direction for each layer laid end to end: `n_embd` values
#' for layer `il_start`, then `n_embd` for the next layer, and so on. Pass
#' `data = NULL` to remove a control vector already applied to the context.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param data Numeric vector of length `n_embd * (il_end - il_start + 1)`, or
#'   `NULL` to clear the current control vector.
#' @param n_embd Embedding dimension of the model, i.e.
#'   `llama_model_info(model)$n_embd`.
#' @param il_start,il_end First and last layer index the vector applies to.
#' @return `NULL`, invisibly. Errors when the dimensions do not match the model.
#' @seealso [llama_lora_apply], [llama_model_info]
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx   <- llama_new_context(model)
#' n_embd <- llama_model_info(model)$n_embd
#'
#' # A steering direction for layers 10..20
#' n_layers <- 20 - 10 + 1
#' vec <- rnorm(n_embd * n_layers) * 0.1
#' llama_apply_control_vector(ctx, vec, n_embd, il_start = 10L, il_end = 20L)
#'
#' llama_apply_control_vector(ctx, NULL, n_embd, 10L, 20L)   # clear it again
#' }
llama_apply_control_vector <- function(ctx, data, n_embd, il_start, il_end) {
    n_embd   <- as.integer(n_embd)
    il_start <- as.integer(il_start)
    il_end   <- as.integer(il_end)

    if (!is.null(data)) {
        expected <- n_embd * (il_end - il_start + 1L)
        if (length(data) != expected) {
            stop("`data` must have length n_embd * (il_end - il_start + 1) = ",
                 expected, ", got ", length(data), call. = FALSE)
        }
        data <- as.double(data)
    }

    .Call("r_llama_apply_adapter_cvec", ctx, data, n_embd, il_start, il_end)
    invisible(NULL)
}

# ============================================================
# Model metadata (individual access)
# ============================================================

#' Get all model metadata as a named character vector
#'
#' Returns all key-value metadata pairs stored in the GGUF model file.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named character vector where names are metadata keys and values
#'   are the corresponding metadata values.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' meta <- llama_model_meta(model)
#' print(meta)
#' }
llama_model_meta <- function(model) {
    .Call("r_llama_model_meta", model)
}

#' Get a single model metadata value by key
#'
#' @param model Model handle returned by [llama_load_model]
#' @param key Character string metadata key (e.g. "general.name", "general.architecture")
#' @return A character scalar with the metadata value, or \code{NULL} if the key
#'   does not exist.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' llama_model_meta_val(model, "general.name")
#' llama_model_meta_val(model, "general.architecture")
#' }
llama_model_meta_val <- function(model, key) {
    stopifnot(is.character(key), length(key) == 1)
    .Call("r_llama_model_meta_val", model, key)
}

# ============================================================
# Vocabulary info
# ============================================================

#' Get vocabulary special token IDs
#'
#' Returns the token IDs for special tokens (BOS, EOS, etc.) and
#' fill-in-middle (FIM) tokens used by the model's vocabulary.
#' A value of -1 indicates the token is not defined.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named integer vector with token IDs for: \code{bos}, \code{eos},
#'   \code{eot}, \code{sep}, \code{nl}, \code{pad}, \code{fim_pre},
#'   \code{fim_suf}, \code{fim_mid}, \code{fim_rep}, \code{fim_sep}.
#'   A value of -1 means the token is not defined by the model.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' vocab <- llama_vocab_info(model)
#' cat("BOS token:", vocab["bos"], "\n")
#' cat("EOS token:", vocab["eos"], "\n")
#' }
llama_vocab_info <- function(model) {
    .Call("r_llama_vocab_info", model)
}

#' Get vocabulary type
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A character string: one of `"spm"` (LLaMA/SentencePiece BPE),
#'   `"bpe"` (GPT-2 BPE), `"wpm"` (BERT WordPiece), `"ugm"` (T5 Unigram),
#'   `"rwkv"`, `"plamo2"`, or `"none"`.
#' @export
llama_vocab_type <- function(model) {
    .Call("r_llama_vocab_type", model)
}

#' Check if a token is an end-of-generation token
#'
#' Returns `TRUE` for EOS, EOT, and other tokens that signal end of output.
#' Useful for implementing custom generation loops.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param token Integer token ID (0-based)
#' @return A logical scalar.
#' @export
llama_vocab_is_eog <- function(model, token) {
    .Call("r_llama_vocab_is_eog", model, as.integer(token))
}

#' Check if a token is a control token
#'
#' @param model Model handle returned by [llama_load_model]
#' @param token Integer token ID (0-based)
#' @return A logical scalar.
#' @export
llama_vocab_is_control <- function(model, token) {
    .Call("r_llama_vocab_is_control", model, as.integer(token))
}

#' Get the text representation of a token
#'
#' Returns the raw text string stored in the vocabulary for a given token ID.
#' Unlike [llama_token_to_piece], this does not apply any special rendering —
#' it returns exactly what is stored in the GGUF vocabulary table.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param token Integer token ID (0-based)
#' @return A character string, or `NULL` if the token has no text entry.
#' @export
llama_vocab_get_text <- function(model, token) {
    .Call("r_llama_vocab_get_text", model, as.integer(token))
}

#' Get the score of a token
#'
#' Returns the log-probability score stored in the vocabulary (used by SPM/UGM tokenizers).
#'
#' @param model Model handle returned by [llama_load_model]
#' @param token Integer token ID (0-based)
#' @return A numeric scalar.
#' @export
llama_vocab_get_score <- function(model, token) {
    .Call("r_llama_vocab_get_score", model, as.integer(token))
}

#' Get the attribute flags of a token
#'
#' Token attributes are stored as a bit mask in the GGUF vocabulary. This
#' returns the set flags by name, which is easier to work with from R than the
#' raw integer.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param token Integer token ID (0-based)
#' @return A character vector of the flags that are set, drawn from
#'   `"unknown"`, `"unused"`, `"normal"`, `"control"`, `"user_defined"`,
#'   `"byte"`, `"normalized"`, `"lstrip"`, `"rstrip"`, `"single_word"`.
#'   A zero-length vector means the token has no attributes defined.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' llama_vocab_get_attr(model, 0L)   # e.g. "control"
#' }
llama_vocab_get_attr <- function(model, token) {
    .Call("r_llama_vocab_get_attr", model, as.integer(token))
}

#' Tokenizer BOS / EOS / SEP insertion defaults
#'
#' Report whether the model's tokenizer is configured to add a
#' beginning-of-sequence, end-of-sequence, or separator token automatically.
#' These are the defaults [llama_tokenize] follows when `add_special = TRUE`.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A logical scalar.
#' @name llama_vocab_add_special
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' llama_vocab_get_add_bos(model)
#' }
NULL

#' @rdname llama_vocab_add_special
#' @export
llama_vocab_get_add_bos <- function(model) {
    .Call("r_llama_vocab_get_add_bos", model)
}

#' @rdname llama_vocab_add_special
#' @export
llama_vocab_get_add_eos <- function(model) {
    .Call("r_llama_vocab_get_add_eos", model)
}

#' @rdname llama_vocab_add_special
#' @export
llama_vocab_get_add_sep <- function(model) {
    .Call("r_llama_vocab_get_add_sep", model)
}

#' Additional special token IDs
#'
#' `llama_vocab_mask()` returns the mask token (used by encoder models such as
#' BERT); `llama_vocab_fim_pad()` returns the fill-in-the-middle padding token
#' (used by code models). Both complement the ids already reported by
#' [llama_vocab_info].
#'
#' @param model Model handle returned by [llama_load_model]
#' @return An integer token ID, or `NA_integer_` when the vocabulary does not
#'   define the token.
#' @name llama_vocab_special_tokens
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' llama_vocab_fim_pad(model)
#' }
NULL

#' @rdname llama_vocab_special_tokens
#' @export
llama_vocab_mask <- function(model) {
    .Call("r_llama_vocab_mask", model)
}

#' @rdname llama_vocab_special_tokens
#' @export
llama_vocab_fim_pad <- function(model) {
    .Call("r_llama_vocab_fim_pad", model)
}

# ============================================================
# Context configuration
# ============================================================

#' Set the number of threads for a context
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param n_threads Number of threads for single-token generation
#' @param n_threads_batch Number of threads for batch processing (prompt encoding).
#'   Defaults to the same value as \code{n_threads}.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' llama_set_threads(ctx, n_threads = 8L)
#' }
llama_set_threads <- function(ctx, n_threads, n_threads_batch = n_threads) {
    .Call("r_llama_set_n_threads", ctx, as.integer(n_threads), as.integer(n_threads_batch))
    invisible(NULL)
}

#' Set causal attention mode
#'
#' When disabled, the model uses full (bidirectional) attention.
#' This is useful for embedding models.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param causal Logical; \code{TRUE} for causal (autoregressive) attention,
#'   \code{FALSE} for full bidirectional attention
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' llama_set_causal_attn(ctx, FALSE)  # for embeddings
#' }
llama_set_causal_attn <- function(ctx, causal) {
    .Call("r_llama_set_causal_attn", ctx, as.logical(causal))
    invisible(NULL)
}

#' Get context window size
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: the context window size (number of tokens).
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model, n_ctx = 4096L)
#' llama_n_ctx(ctx)  # 4096
#' }
llama_n_ctx <- function(ctx) {
    .Call("r_llama_n_ctx", ctx)
}

#' Get the model associated with a context
#'
#' Returns the model handle that was used to create this context.
#' The returned object is the same R external pointer that was passed to
#' [llama_new_context] — no new allocation occurs.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A model handle (external pointer), equivalent to the original
#'   handle returned by [llama_load_model].
#' @export
llama_get_model <- function(ctx) {
    .Call("r_llama_get_model", ctx)
}

#' Set warmup mode
#'
#' When `warmup = TRUE`, the context runs in warmup mode which pre-caches
#' model weights in GPU memory without producing meaningful outputs.
#' Call with `warmup = FALSE` to return to normal inference mode.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param warmup Logical; `TRUE` to enable warmup mode, `FALSE` to disable.
#' @return No return value, called for side effects.
#' @export
llama_set_warmup <- function(ctx, warmup) {
    .Call("r_llama_set_warmup", ctx, as.logical(warmup))
    invisible(NULL)
}

#' Set or clear the abort callback
#'
#' Registers an R function that is called periodically during generation.
#' If the function returns `TRUE`, the current decode operation is aborted.
#' Pass `NULL` to remove the callback.
#'
#' Note: only one callback is active globally — setting a new one replaces
#' the previous one across all contexts.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param fn A zero-argument R function returning a logical scalar,
#'   or `NULL` to clear.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # Abort after 2 seconds
#' deadline <- Sys.time() + 2
#' llama_set_abort_callback(ctx, function() Sys.time() > deadline)
#' result <- llama_generate(ctx, "Tell me a long story", max_new_tokens = 500L)
#' llama_set_abort_callback(ctx, NULL)
#' }
llama_set_abort_callback <- function(ctx, fn) {
    .Call("r_llama_set_abort_callback", ctx, fn)
    invisible(NULL)
}

#' Get per-sequence context window size
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: maximum context size per sequence.
#' @export
llama_n_ctx_seq <- function(ctx) {
    .Call("r_llama_n_ctx_seq", ctx)
}

#' Get logical batch size
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: the logical batch size (max tokens per `llama_decode` call).
#' @export
llama_n_batch <- function(ctx) {
    .Call("r_llama_n_batch", ctx)
}

#' Get physical micro-batch size
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: the physical micro-batch size.
#' @export
llama_n_ubatch <- function(ctx) {
    .Call("r_llama_n_ubatch", ctx)
}

#' Get maximum number of sequences
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: maximum number of concurrent sequences.
#' @export
llama_n_seq_max <- function(ctx) {
    .Call("r_llama_n_seq_max", ctx)
}

#' Name of a flash-attention type
#'
#' llama.cpp's own name for one of the values [llama_new_context] accepts for
#' \code{flash_attn}. Note that these are the library's names, not the R
#' argument's: \code{"on"} is called \code{"enabled"} and \code{"off"} is called
#' \code{"disabled"}.
#'
#' This translates a request, not an outcome --- to find out what a context
#' actually settled on after asking for \code{"auto"}, use
#' [llama_context_flash_attn].
#'
#' @param type One of \code{"auto"}, \code{"on"} or \code{"off"}, as passed to
#'   [llama_new_context].
#' @return A character scalar with llama.cpp's name for that type.
#' @seealso [llama_context_flash_attn], [llama_new_context]
#' @export
#' @examples
#' llama_flash_attn_type_name("auto")  # "auto"
#' llama_flash_attn_type_name("on")    # "enabled"
llama_flash_attn_type_name <- function(type) {
    stopifnot(is.character(type), length(type) == 1)
    type_int <- switch(type,
        "auto" = -1L,
        "on"   =  1L,
        "off"  =  0L,
        stop("type must be 'auto', 'on', or 'off'"))
    .Call("r_llama_flash_attn_type_name", type_int)
}

#' Flash attention a context actually uses
#'
#' Reports whether flash attention ended up enabled for a context, which is the
#' question [llama_new_context]'s \code{flash_attn = "auto"} leaves open: the
#' decision is llama.cpp's, and depends on the model and backend.
#'
#' Whether the request was \code{"auto"} cannot be recovered from a live
#' context: llama.cpp resolves it while the context is being built and does not
#' keep the original request.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A named list:
#'   \itemize{
#'     \item \code{enabled}: logical, whether flash attention is in use
#'     \item \code{type_name}: llama.cpp's name for the resolved state,
#'       \code{"enabled"} or \code{"disabled"}
#'   }
#' @seealso [llama_flash_attn_type_name], [llama_new_context]
#' @export
#' @examples
#' \dontrun{
#' ctx <- llama_new_context(model, flash_attn = "auto")
#' if (llama_context_flash_attn(ctx)$enabled) {
#'     message("llama.cpp enabled flash attention for this model")
#' }
#' }
llama_context_flash_attn <- function(ctx) {
    .Call("r_llama_context_flash_attn", ctx)
}

#' Get number of threads for single-token generation
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: current thread count for generation.
#' @export
llama_n_threads <- function(ctx) {
    .Call("r_llama_n_threads", ctx)
}

#' Get number of threads for batch processing
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: current thread count for prompt encoding.
#' @export
llama_n_threads_batch <- function(ctx) {
    .Call("r_llama_n_threads_batch", ctx)
}

#' Get pooling type
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A character string: one of `"none"`, `"mean"`, `"cls"`, `"last"`,
#'   `"rank"`, `"unspecified"`.
#' @export
llama_pooling_type <- function(ctx) {
    .Call("r_llama_pooling_type", ctx)
}

# ============================================================
# Memory / KV Cache
# ============================================================

#' Clear the KV cache
#'
#' Removes all tokens from the KV cache. Call this before starting
#' a new generation from scratch.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # Clear the KV cache to start a fresh conversation
#' llama_memory_clear(ctx)
#' result <- llama_generate(ctx, "New topic: ", max_new_tokens = 50L)
#' }
llama_memory_clear <- function(ctx) {
    .Call("r_llama_memory_clear", ctx)
    invisible(NULL)
}

#' Remove tokens from a sequence in the KV cache
#'
#' Removes cached tokens for the given sequence in the position range [p0, p1).
#' Use p0 = -1 and p1 = -1 to remove all tokens for the sequence.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID (integer)
#' @param p0 Start position (inclusive, -1 for beginning)
#' @param p1 End position (exclusive, -1 for end)
#' @return A logical scalar: \code{TRUE} if tokens were successfully removed.
#' @export
#' @examples
#' \dontrun{
#' # Remove all tokens from sequence 0
#' llama_memory_seq_rm(ctx, seq_id = 0L, p0 = -1L, p1 = -1L)
#' }
llama_memory_seq_rm <- function(ctx, seq_id, p0 = -1L, p1 = -1L) {
    .Call("r_llama_memory_seq_rm", ctx, as.integer(seq_id),
          as.integer(p0), as.integer(p1))
}

#' Copy a sequence in the KV cache
#'
#' Copies cached tokens from one sequence to another in the position range [p0, p1).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id_src Source sequence ID
#' @param seq_id_dst Destination sequence ID
#' @param p0 Start position (inclusive, -1 for beginning)
#' @param p1 End position (exclusive, -1 for end)
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # Copy sequence 0 to sequence 1
#' llama_memory_seq_cp(ctx, seq_id_src = 0L, seq_id_dst = 1L,
#'                     p0 = -1L, p1 = -1L)
#' }
llama_memory_seq_cp <- function(ctx, seq_id_src, seq_id_dst, p0 = -1L, p1 = -1L) {
    .Call("r_llama_memory_seq_cp", ctx, as.integer(seq_id_src),
          as.integer(seq_id_dst), as.integer(p0), as.integer(p1))
    invisible(NULL)
}

#' Keep only one sequence in the KV cache
#'
#' Removes all sequences except the specified one from the KV cache.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID to keep
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' llama_memory_seq_keep(ctx, seq_id = 0L)
#' }
llama_memory_seq_keep <- function(ctx, seq_id) {
    .Call("r_llama_memory_seq_keep", ctx, as.integer(seq_id))
    invisible(NULL)
}

#' Shift token positions in a sequence
#'
#' Adds a position delta to all tokens in the given sequence within [p0, p1).
#' This is useful for implementing context shifting (sliding window).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID
#' @param p0 Start position (inclusive)
#' @param p1 End position (exclusive)
#' @param delta Position shift amount (can be negative)
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # Shift positions left by 100 for context window management
#' llama_memory_seq_add(ctx, seq_id = 0L, p0 = 100L, p1 = -1L, delta = -100L)
#' }
llama_memory_seq_add <- function(ctx, seq_id, p0, p1, delta) {
    .Call("r_llama_memory_seq_add", ctx, as.integer(seq_id),
          as.integer(p0), as.integer(p1), as.integer(delta))
    invisible(NULL)
}

#' Integer-divide token positions in a sequence
#'
#' Divides all token positions in the range \code{[p0, p1)} for the given
#' sequence by \code{d}. Use \code{p0 = -1} and \code{p1 = -1} for the full range.
#' Useful for implementing sliding-window context compression.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID
#' @param p0 Start position (inclusive). Use -1 for beginning.
#' @param p1 End position (exclusive). Use -1 for end.
#' @param d Divisor (positive integer)
#' @return No return value, called for side effects.
#' @export
llama_memory_seq_div <- function(ctx, seq_id, p0, p1, d) {
    .Call("r_llama_memory_seq_div", ctx, as.integer(seq_id),
          as.integer(p0), as.integer(p1), as.integer(d))
    invisible(NULL)
}

#' Get position range for a sequence
#'
#' Returns the minimum and maximum token positions for a given sequence
#' in the KV cache.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID
#' @return A named integer vector with elements \code{min} and \code{max}.
#' @export
#' @examples
#' \dontrun{
#' range <- llama_memory_seq_pos_range(ctx, seq_id = 0L)
#' cat("Positions:", range["min"], "to", range["max"], "\n")
#' }
llama_memory_seq_pos_range <- function(ctx, seq_id) {
    .Call("r_llama_memory_seq_pos_range", ctx, as.integer(seq_id))
}

#' Check if the KV cache supports shifting
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A logical scalar: \code{TRUE} if the memory supports position shifting.
#' @export
#' @examples
#' \dontrun{
#' if (llama_memory_can_shift(ctx)) {
#'   message("Context shifting is supported")
#' }
#' }
llama_memory_can_shift <- function(ctx) {
    .Call("r_llama_memory_can_shift", ctx)
}

# ============================================================
# State save / load
# ============================================================

#' Save context state to file
#'
#' Saves the full context state (including KV cache) to a binary file.
#' This allows resuming generation later from the exact same state.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param path File path to save state to
#' @return A logical scalar: \code{TRUE} on success (errors on failure).
#' @export
#' @examples
#' \dontrun{
#' llama_state_save(ctx, "state.bin")
#' }
llama_state_save <- function(ctx, path) {
    stopifnot(is.character(path), length(path) == 1)
    .Call("r_llama_state_save", ctx, path)
}

#' Load context state from file
#'
#' Restores a previously saved context state (including KV cache).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param path File path to load state from
#' @return A logical scalar: \code{TRUE} on success (errors on failure).
#' @export
#' @examples
#' \dontrun{
#' llama_state_load(ctx, "state.bin")
#' # Continue generation from saved state
#' result <- llama_generate(ctx, "")
#' }
llama_state_load <- function(ctx, path) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: state file does not exist: ", path)
    .Call("r_llama_state_load", ctx, path)
}

#' Get the size of the serialized context state in bytes
#'
#' Returns the number of bytes required to serialize the current context state
#' (KV cache + sampling state). Use before allocating a buffer for raw state I/O.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A numeric scalar (size in bytes).
#' @export
llama_state_get_size <- function(ctx) {
    .Call("r_llama_state_get_size", ctx)
}

#' Copy the context state to and from raw bytes
#'
#' `llama_state_get_data()` serializes the whole context state (KV cache,
#' logits, embeddings) into a raw vector; `llama_state_set_data()` restores it.
#' This is the in-memory counterpart of [llama_state_save] /
#' [llama_state_load], useful for snapshotting a context without touching disk.
#'
#' The bytes are only meaningful to a context created from the same model with
#' the same parameters. Restoring a snapshot into a mismatched context errors.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param data A raw vector previously returned by `llama_state_get_data()`.
#' @return `llama_state_get_data()`: a raw vector.
#'   `llama_state_set_data()`: the number of bytes read, invisibly.
#' @name llama_state_data
#' @seealso [llama_state_save], [llama_state_get_size]
#' @examples
#' \dontrun{
#' ctx <- llama_new_context(model)
#' llama_generate(ctx, "The capital of France is", max_new_tokens = 8L)
#'
#' snapshot <- llama_state_get_data(ctx)   # remember where we are
#' llama_generate(ctx, "Something else", max_new_tokens = 8L)
#' llama_state_set_data(ctx, snapshot)     # and go back
#' }
NULL

#' @rdname llama_state_data
#' @export
llama_state_get_data <- function(ctx) {
    .Call("r_llama_state_get_data", ctx)
}

#' @rdname llama_state_data
#' @export
llama_state_set_data <- function(ctx, data) {
    stopifnot(is.raw(data))
    invisible(.Call("r_llama_state_set_data", ctx, data))
}

# Flag words accepted by the per-sequence state functions. Kept internal: the
# R API takes them as booleans rather than exposing a bit mask.
.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY <- 1L
.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE    <- 2L

.state_seq_flags <- function(partial_only = FALSE, on_device = FALSE) {
    flags <- 0L
    if (isTRUE(partial_only)) flags <- bitwOr(flags, .LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
    if (isTRUE(on_device))    flags <- bitwOr(flags, .LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)
    flags
}

#' Save and restore the state of a single sequence
#'
#' Where [llama_state_get_data] snapshots the entire context, these work on one
#' sequence at a time. That makes it possible to checkpoint a single
#' conversation in a multi-sequence context, or to cache the KV state of a long
#' shared prefix and restore it into a fresh sequence instead of recomputing it.
#'
#' `partial_only = TRUE` restricts the copy to partial state — the SWA window of
#' a sliding-window model, or the recurrent state of a Mamba-style model. Leave
#' it `FALSE` for ordinary transformers.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID (0-based), below `llama_n_seq_max(ctx)`.
#' @param data A raw vector previously returned by `llama_state_seq_get_data()`.
#' @param partial_only Copy only partial state (SWA window / recurrent state).
#' @param on_device Keep the copy in device memory where the backend supports it.
#' @return `llama_state_seq_get_size()`: size in bytes.
#'   `llama_state_seq_get_data()`: a raw vector.
#'   `llama_state_seq_set_data()`: bytes read, invisibly.
#' @name llama_state_seq
#' @seealso [llama_state_seq_save_file], [llama_state_data], [llama_memory_seq_cp]
#' @examples
#' \dontrun{
#' ctx <- llama_new_context(model, n_seq_max = 4L)
#'
#' # Cache the KV state of a shared prefix, then reuse it for another sequence
#' prefix <- llama_state_seq_get_data(ctx, seq_id = 0L)
#' llama_state_seq_set_data(ctx, prefix, seq_id = 1L)
#' }
NULL

#' @rdname llama_state_seq
#' @export
llama_state_seq_get_size <- function(ctx, seq_id, partial_only = FALSE, on_device = FALSE) {
    .Call("r_llama_state_seq_get_size", ctx, as.integer(seq_id),
          .state_seq_flags(partial_only, on_device))
}

#' @rdname llama_state_seq
#' @export
llama_state_seq_get_data <- function(ctx, seq_id, partial_only = FALSE, on_device = FALSE) {
    .Call("r_llama_state_seq_get_data", ctx, as.integer(seq_id),
          .state_seq_flags(partial_only, on_device))
}

#' @rdname llama_state_seq
#' @export
llama_state_seq_set_data <- function(ctx, data, seq_id, partial_only = FALSE, on_device = FALSE) {
    stopifnot(is.raw(data))
    invisible(.Call("r_llama_state_seq_set_data", ctx, data, as.integer(seq_id),
                    .state_seq_flags(partial_only, on_device)))
}

#' Save and load the state of a single sequence to a file
#'
#' The file-backed counterpart of [llama_state_seq_get_data]. The token list
#' that produced the state is stored alongside it, so a reloaded sequence can
#' report which prompt it came from.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param path Path to the session file.
#' @param seq_id Sequence ID (0-based).
#' @param tokens Integer vector of the tokens that produced this state, or
#'   `NULL` to store none.
#' @param n_token_capacity Maximum number of tokens to read back. Must be at
#'   least as large as the count that was saved.
#' @return `llama_state_seq_save_file()`: bytes written, invisibly.
#'   `llama_state_seq_load_file()`: a list with `n_bytes` and the `tokens` that
#'   were stored with the state.
#' @name llama_state_seq_file
#' @seealso [llama_state_seq], [llama_state_save]
#' @examples
#' \dontrun{
#' toks <- llama_tokenize(ctx, "A long shared prefix")
#' llama_state_seq_save_file(ctx, "prefix.bin", seq_id = 0L, tokens = toks)
#'
#' ctx2 <- llama_new_context(model)
#' res  <- llama_state_seq_load_file(ctx2, "prefix.bin", seq_id = 0L)
#' identical(res$tokens, toks)
#' }
NULL

#' @rdname llama_state_seq_file
#' @export
llama_state_seq_save_file <- function(ctx, path, seq_id, tokens = NULL) {
    stopifnot(is.character(path), length(path) == 1)
    if (!is.null(tokens)) tokens <- as.integer(tokens)
    invisible(.Call("r_llama_state_seq_save_file", ctx, path, as.integer(seq_id), tokens))
}

#' @rdname llama_state_seq_file
#' @export
llama_state_seq_load_file <- function(ctx, path, seq_id, n_token_capacity = 65536L) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: state file does not exist: ", path)
    .Call("r_llama_state_seq_load_file", ctx, path, as.integer(seq_id),
          as.integer(n_token_capacity))
}

#' Synchronize asynchronous computation
#'
#' Blocks until all pending GPU/async operations for this context are complete.
#' Normally not needed — `llama_decode` and `llama_generate` are synchronous —
#' but useful when using low-level batch APIs in async mode.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
llama_synchronize <- function(ctx) {
    .Call("r_llama_synchronize", ctx)
    invisible(NULL)
}

# ============================================================
# Logits
# ============================================================

#' Get logits from the last decode step
#'
#' Returns the raw logit vector (unnormalized log-probabilities) from the
#' last token position after a decode operation.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A numeric vector of length \code{n_vocab} containing the logits.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' result <- llama_generate(ctx, "The capital of France is", max_new_tokens = 1L)
#' logits <- llama_get_logits(ctx)
#' # Find top token
#' top_id <- which.max(logits)
#' }
llama_get_logits <- function(ctx) {
    .Call("r_llama_get_logits", ctx)
}

#' Get logits for a specific token position
#'
#' Returns the logit vector for token at index \code{i} in the last decoded batch.
#' Use \code{i = -1} to get the logits for the last token.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param i Integer index into the last batch (0-based). Use \code{-1} for the last token.
#' @return A numeric vector of length \code{n_vocab}.
#' @export
llama_get_logits_ith <- function(ctx, i) {
    .Call("r_llama_get_logits_ith", ctx, as.integer(i))
}

# ============================================================
# Performance
# ============================================================

#' Get performance statistics
#'
#' Returns timing and count statistics for the current context,
#' including prompt processing time, token generation time, and counts.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A named list with fields:
#'   - `t_load_ms`: model load time in milliseconds
#'   - `t_p_eval_ms`: prompt processing time in milliseconds
#'   - `t_eval_ms`: token generation time in milliseconds
#'   - `n_p_eval`: number of prompt tokens processed
#'   - `n_eval`: number of tokens generated
#'   - `n_reused`: number of reused compute graphs
#' @export
#' @examples
#' \dontrun{
#' result <- llama_generate(ctx, "Hello world")
#' perf <- llama_perf(ctx)
#' cat("Prompt speed:", perf$n_p_eval / (perf$t_p_eval_ms / 1000), "tok/s\n")
#' cat("Generation speed:", perf$n_eval / (perf$t_eval_ms / 1000), "tok/s\n")
#' }
llama_perf <- function(ctx) {
    .Call("r_llama_perf_context", ctx)
}

#' Reset performance counters
#'
#' Resets the timing and token count statistics for the context.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \dontrun{
#' # Reset counters before benchmarking a specific generation
#' llama_perf_reset(ctx)
#' result <- llama_generate(ctx, "Benchmark prompt", max_new_tokens = 100L)
#' perf <- llama_perf(ctx)
#' cat("Generation:", perf$n_eval / (perf$t_eval_ms / 1000), "tok/s\n")
#' }
llama_perf_reset <- function(ctx) {
    .Call("r_llama_perf_context_reset", ctx)
    invisible(NULL)
}

#' Print performance statistics to the console
#'
#' Prints a formatted summary of timing and throughput statistics for the context
#' (load time, prompt processing speed, generation speed). Output goes to the
#' R console via the llama.cpp logging callback.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
llama_perf_print <- function(ctx) {
    .Call("r_llama_perf_context_print", ctx)
    invisible(NULL)
}

#' Sampler performance statistics
#'
#' Timing for the sampling step alone, as opposed to the decode timings
#' reported by [llama_perf]. Together they show how generation time splits
#' between the model and the sampler chain — useful when an expensive sampler
#' such as a grammar is in play.
#'
#' These take a streaming generation state rather than a context, because the
#' sampler chain only outlives a single call on the streaming path: a one-shot
#' [llama_generate] builds its chain and frees it before returning.
#'
#' @param state Generation state returned by [llama_gen_begin] or
#'   [llama_gen_begin_at].
#' @return `llama_perf_sampler()`: a named list with `t_sample_ms` (sampling
#'   time in milliseconds) and `n_sample` (number of tokens sampled).
#'   The other two return `NULL` invisibly.
#' @name llama_perf_sampler
#' @seealso [llama_perf], [llama_gen_begin]
#' @examples
#' \dontrun{
#' st <- llama_gen_begin(ctx, "Hello", max_new_tokens = 64L)
#' repeat {
#'   chunk <- llama_gen_next(st)
#'   if (is.null(chunk)) break
#' }
#' p <- llama_perf_sampler(st)
#' cat("Sampling:", p$t_sample_ms, "ms for", p$n_sample, "tokens\n")
#' }
NULL

#' @rdname llama_perf_sampler
#' @export
llama_perf_sampler <- function(state) {
    .Call("r_llama_perf_sampler", state)
}

#' @rdname llama_perf_sampler
#' @export
llama_perf_sampler_print <- function(state) {
    .Call("r_llama_perf_sampler_print", state)
    invisible(NULL)
}

#' @rdname llama_perf_sampler
#' @export
llama_perf_sampler_reset <- function(state) {
    .Call("r_llama_perf_sampler_reset", state)
    invisible(NULL)
}

#' Print memory breakdown by device
#'
#' Prints a debug summary of how model weights are distributed across compute
#' devices (CPU, GPU layers). Useful for diagnosing memory allocation with
#' partial GPU offload.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
llama_memory_breakdown_print <- function(ctx) {
    .Call("r_llama_memory_breakdown_print", ctx)
    invisible(NULL)
}

# ============================================================
# System & Hardware info
# ============================================================

#' Get system information string
#'
#' Returns a string with information about the system capabilities
#' detected by llama.cpp (SIMD support, etc.).
#'
#' @return A character scalar with system capability information.
#' @export
#' @examples
#' cat(llama_system_info(), "\n")
llama_system_info <- function() {
    .Call("r_llama_system_info")
}

#' Check whether memory-mapped file I/O is supported
#'
#' @return A logical scalar: \code{TRUE} if mmap is supported.
#' @export
#' @examples
#' # Check memory-mapping support before loading large models
#' if (llama_supports_mmap()) {
#'   message("mmap available — large models will load faster")
#' }
llama_supports_mmap <- function() {
    .Call("r_llama_supports_mmap")
}

#' Check whether memory locking is supported
#'
#' @return A logical scalar: \code{TRUE} if mlock is supported.
#' @export
#' @examples
#' # Check if memory locking is available (prevents swapping model to disk)
#' if (llama_supports_mlock()) {
#'   message("mlock available — model weights can be pinned in RAM")
#' }
llama_supports_mlock <- function() {
    .Call("r_llama_supports_mlock")
}

#' Check whether RPC backend is available
#'
#' @return A logical scalar: `TRUE` if the RPC backend is compiled in.
#' @export
llama_supports_rpc <- function() {
    .Call("r_llama_supports_rpc")
}

#' Get maximum number of devices
#'
#' @return An integer scalar: the maximum number of compute devices available.
#' @export
#' @examples
#' # Query the maximum number of devices supported by the backend
#' n <- llama_max_devices()
#' cat("Max devices:", n, "\n")
llama_max_devices <- function() {
    .Call("r_llama_max_devices")
}

#' Get the maximum number of parallel sequences
#'
#' The compile-time ceiling on `n_seq_max` in [llama_new_context]. Requesting
#' more parallel sequences than this fails regardless of available memory.
#'
#' @return An integer scalar.
#' @seealso [llama_new_context], [llama_n_seq_max]
#' @export
#' @examples
#' # Upper bound on parallel sequences for this build
#' llama_max_parallel_sequences()
llama_max_parallel_sequences <- function() {
    .Call("r_llama_max_parallel_sequences")
}

#' Get the maximum number of tensor buffer-type overrides
#'
#' The size a tensor buffer-type override buffer must have. Reported for
#' completeness; llamaR does not currently expose per-tensor overrides.
#'
#' @return An integer scalar.
#' @export
#' @examples
#' llama_max_tensor_buft_overrides()
llama_max_tensor_buft_overrides <- function() {
    .Call("r_llama_max_tensor_buft_overrides")
}

# ============================================================
# Chat: builtin templates
# ============================================================

#' List built-in chat templates
#'
#' Returns a character vector of all chat template names supported by llama.cpp.
#'
#' @return A character vector of built-in template names.
#' @export
#' @examples
#' # See which chat template formats are supported out of the box
#' templates <- llama_chat_builtin_templates()
#' head(templates)
llama_chat_builtin_templates <- function() {
    .Call("r_llama_chat_builtin_templates")
}

#' Convert a single token ID to its text piece
#'
#' @param ctx A context pointer (llama_context).
#' @param token Integer token ID.
#' @param special Logical. If TRUE, render special tokens (e.g. \code{<bos>}).
#' @return A character string — the text piece for the token.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx   <- llama_new_context(model)
#'
#' # Inspect individual tokens from tokenizer output
#' tokens <- llama_tokenize(ctx, "Hello world")
#' pieces <- vapply(tokens, function(t) llama_token_to_piece(ctx, t), "")
#' cat(paste(pieces, collapse = "|"), "\n")
#' }
llama_token_to_piece <- function(ctx, token, special = FALSE) {
    .Call("r_llama_token_to_piece", ctx, as.integer(token), as.logical(special))
}

#' Encode tokens using the encoder (encoder-decoder models only)
#'
#' Runs the encoder pass for encoder-decoder architectures (e.g. T5, BART).
#' The encoder output is stored internally and used by subsequent decoder calls.
#'
#' @param ctx A context pointer (llama_context).
#' @param tokens Integer vector of token IDs to encode.
#' @return Integer return code (0 = success, negative = error).
#' @export
#' @examples
#' \dontrun{
#' model  <- llama_load_model("t5-model.gguf")
#' ctx    <- llama_new_context(model)
#' tokens <- llama_tokenize(ctx, "Hello world")
#' llama_encode(ctx, tokens)
#' }
llama_encode <- function(ctx, tokens) {
    .Call("r_llama_encode", ctx, as.integer(tokens))
}

#' Initialise a llama batch
#'
#' Allocates a \code{llama_batch} that can hold up to \code{n_tokens} tokens.
#' Use \code{llama_batch_free()} to release the memory when done.
#'
#' @param n_tokens Maximum number of tokens in the batch.
#' @param embd Embedding size; 0 means token-ID mode (normal inference).
#' @param n_seq_max Maximum number of sequences per token.
#' @return An external pointer to the allocated batch.
#' @export
#' @examples
#' \dontrun{
#' batch <- llama_batch_init(512L)
#' llama_batch_free(batch)
#' }
llama_batch_init <- function(n_tokens, embd = 0L, n_seq_max = 1L) {
    .Call("r_llama_batch_init", as.integer(n_tokens), as.integer(embd), as.integer(n_seq_max))
}

#' Free a llama batch allocated with \code{llama_batch_init()}
#'
#' @param batch An external pointer returned by \code{llama_batch_init()}.
#' @return \code{NULL} invisibly.
#' @export
#' @examples
#' \dontrun{
#' batch <- llama_batch_init(512L)
#' llama_batch_free(batch)
#' }
llama_batch_free <- function(batch) {
    invisible(.Call("r_llama_batch_free", batch))
}
