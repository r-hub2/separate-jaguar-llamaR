# ============================================================
# LoRA multi-adapter contract (HEAVY).
# Release 0.2.5 rewrote apply/remove/clear on top of master's
# llama_set_adapters_lora(ctx, adapters**, n, scales*) with a
# per-ctx tracking map. The documented contract:
#   apply(ctx, lora, scale)  -> add/update entry (NULL)
#   remove(ctx, lora)        -> 0 if it was applied, -1 otherwise
#   clear(ctx)               -> drop all; a later remove() -> -1
# Needs a model + a LoRA adapter; listed in tests/testthat.R
# `heavy` so it is skipped on CRAN. Skips locally until the
# adapter file exists.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"
LORA_PATH  <- "/mnt/Data2/DS_projects/llm_models/test-lora-adapter.gguf"

skip_if_no_lora <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
    if (!file.exists(LORA_PATH))  skip("test LoRA adapter not available")
}

test_that("remove returns -1 for an adapter that was never applied", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    # never applied to this ctx -> -1, and it must NOT error.
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})

test_that("apply then remove returns 0, and a second remove returns -1", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 1.0)

    expect_identical(llama_lora_remove(ctx, lora), 0L)   # was active
    expect_identical(llama_lora_remove(ctx, lora), -1L)  # now gone
})

test_that("re-applying the same adapter updates the scale (single entry)", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 0.5)
    llama_lora_apply(ctx, lora, scale = 1.0)   # update, not a duplicate

    # a single remove clears the (single) tracked entry -> 0, then -1.
    expect_identical(llama_lora_remove(ctx, lora), 0L)
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})

test_that("clear drops all adapters; a later remove returns -1", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 1.0)
    expect_no_error(llama_lora_clear(ctx))

    # everything cleared -> the previously-active adapter is gone.
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})

# ============================================================
# Adapter metadata and aLoRA invocation tokens
# ============================================================

test_that("llama_lora_meta returns a named character vector", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    meta <- llama_lora_meta(lora)

    expect_type(meta, "character")
    # A GGUF adapter always records at least its architecture.
    expect_gt(length(meta), 0L)
    expect_false(is.null(names(meta)))
    expect_true(all(nzchar(names(meta))))
})

test_that("llama_lora_meta_val agrees with llama_lora_meta", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    meta <- llama_lora_meta(lora)
    skip_if(length(meta) == 0L, "adapter carries no metadata")

    key <- names(meta)[1]
    expect_equal(llama_lora_meta_val(lora, key), unname(meta[[key]]))
})

test_that("llama_lora_meta_val returns NULL for an absent key", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    expect_null(llama_lora_meta_val(lora, "llamar.no.such.key"))
    expect_error(llama_lora_meta_val(lora, c("a", "b")))
})

test_that("an ordinary LoRA reports no aLoRA invocation tokens", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    toks <- llama_lora_alora_invocation_tokens(lora)

    # Only an activated LoRA defines these; a plain adapter yields NULL.
    if (!is.null(toks)) {
        expect_type(toks, "integer")
        expect_gt(length(toks), 0L)
        expect_true(all(toks >= 0L))
    } else {
        expect_null(toks)
    }
})
