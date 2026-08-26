# ============================================================
# Context / sequence state serialization (HEAVY).
#   llama_state_get_data() / _set_data()      — whole-context snapshots
#   llama_state_seq_get_size/_get_data/_set_data
#   llama_state_seq_save_file() / _load_file()
# Every test needs a real context with a populated KV cache, so this
# file is listed in tests/testthat.R `heavy`. Run with NOT_CRAN=true.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

# A context whose KV cache holds a decoded prompt, so the state is non-trivial.
warm_ctx <- function(env, n_ctx = 256L, n_seq_max = 1L, prompt = "The capital of France is") {
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = n_ctx, n_threads = 2L,
                               n_seq_max = n_seq_max)
    withr::defer({ llama_free_context(ctx); llama_free_model(model) }, envir = env)
    llama_generate(ctx, prompt, max_new_tokens = 4L, temp = 0.0)
    list(model = model, ctx = ctx)
}

# --- whole-context state ---------------------------------------------------

test_that("llama_state_get_data returns raw bytes sized like state_get_size", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    blob <- llama_state_get_data(h$ctx)
    expect_true(is.raw(blob))
    expect_gt(length(blob), 0L)
    # get_size may over-estimate, so it is an upper bound, not an equality.
    expect_lte(length(blob), llama_state_get_size(h$ctx))
})

test_that("a context snapshot round-trips through set_data", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    blob <- llama_state_get_data(h$ctx)
    n <- llama_state_set_data(h$ctx, blob)

    expect_type(n, "double")
    expect_gt(n, 0)
    expect_lte(n, length(blob))
})

test_that("restoring a snapshot restores generation behaviour", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    # Snapshot, generate greedily, rewind, generate again: same continuation.
    blob  <- llama_state_get_data(h$ctx)
    first <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L, temp = 0.0)

    llama_state_set_data(h$ctx, blob)
    second <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L, temp = 0.0)

    expect_equal(second, first)
})

test_that("llama_state_set_data rejects non-raw input and junk bytes", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    expect_error(llama_state_set_data(h$ctx, "not raw"))
    expect_error(llama_state_set_data(h$ctx, as.raw(c(1, 2, 3))))
})

# --- per-sequence state ----------------------------------------------------

test_that("llama_state_seq_get_size reports a positive size for a used sequence", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    size <- llama_state_seq_get_size(h$ctx, seq_id = 0L)
    expect_type(size, "double")
    expect_gt(size, 0)
})

test_that("llama_state_seq_get_data returns raw bytes within the reported size", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    blob <- llama_state_seq_get_data(h$ctx, seq_id = 0L)
    expect_true(is.raw(blob))
    expect_gt(length(blob), 0L)
    expect_lte(length(blob), llama_state_seq_get_size(h$ctx, seq_id = 0L))
})

test_that("sequence state round-trips into the same sequence", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    blob <- llama_state_seq_get_data(h$ctx, seq_id = 0L)
    n <- llama_state_seq_set_data(h$ctx, blob, seq_id = 0L)

    expect_type(n, "double")
    expect_gt(n, 0)
})

test_that("sequence state can be copied into a different sequence", {
    skip_if_no_model()
    h <- warm_ctx(environment(), n_seq_max = 2L)

    # This is the prefix-cache use case: fill seq 0, then clone it into seq 1
    # rather than re-decoding the prompt.
    blob <- llama_state_seq_get_data(h$ctx, seq_id = 0L)
    expect_silent(llama_state_seq_set_data(h$ctx, blob, seq_id = 1L))

    # The destination now holds the same span of positions as the source.
    expect_equal(llama_memory_seq_pos_range(h$ctx, 1L),
                 llama_memory_seq_pos_range(h$ctx, 0L))
})

test_that("llama_state_seq_set_data rejects non-raw input", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    expect_error(llama_state_seq_set_data(h$ctx, "not raw", seq_id = 0L))
})

# --- file-backed sequence state -------------------------------------------

test_that("sequence state round-trips through a file, carrying its tokens", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    path <- withr::local_tempfile(fileext = ".bin")
    toks <- llama_tokenize(h$ctx, "The capital of France is")

    llama_state_seq_save_file(h$ctx, path, seq_id = 0L, tokens = toks)
    expect_true(file.exists(path))

    res <- llama_state_seq_load_file(h$ctx, path, seq_id = 0L)
    expect_type(res, "list")
    expect_true(all(c("n_bytes", "tokens") %in% names(res)))
    expect_gt(res$n_bytes, 0)
    # The token list travels with the state unchanged.
    expect_equal(res$tokens, toks)
})

test_that("sequence state saves with no token list", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    path <- withr::local_tempfile(fileext = ".bin")
    llama_state_seq_save_file(h$ctx, path, seq_id = 0L, tokens = NULL)

    res <- llama_state_seq_load_file(h$ctx, path, seq_id = 0L)
    expect_length(res$tokens, 0L)
})

test_that("llama_state_seq_load_file errors on a missing or invalid file", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    expect_error(llama_state_seq_load_file(h$ctx, "/no/such/state.bin", seq_id = 0L),
                 "does not exist")

    junk <- withr::local_tempfile(fileext = ".bin")
    writeBin(as.raw(rep(0L, 64)), junk)
    expect_error(llama_state_seq_load_file(h$ctx, junk, seq_id = 0L))
})

test_that("state file paths must be single strings", {
    skip_if_no_model()
    h <- warm_ctx(environment())

    expect_error(llama_state_seq_save_file(h$ctx, c("a", "b"), seq_id = 0L))
    expect_error(llama_state_seq_load_file(h$ctx, 42, seq_id = 0L))
})
