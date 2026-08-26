# ============================================================
# Flash attention:
#   llama_flash_attn_type_name() — llama.cpp's name for a requested type
#   llama_context_flash_attn()   — what a context actually resolved to
# The name function is pure translation and needs no model; reading a
# context's resolved state does.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

ctx_with_fa <- function(env, flash_attn = "auto") {
    model <- llama_load_model(MODEL_PATH, n_gpu_layers = 0L)
    ctx   <- llama_new_context(model, n_ctx = 256L, n_threads = 2L,
                               flash_attn = flash_attn)
    withr::defer({ llama_free_context(ctx); llama_free_model(model) }, envir = env)
    ctx
}

# --- naming a requested type ------------------------------------------------

test_that("each accepted flash_attn value has a name", {
    expect_equal(llama_flash_attn_type_name("auto"), "auto")
    expect_equal(llama_flash_attn_type_name("on"),   "enabled")
    expect_equal(llama_flash_attn_type_name("off"),  "disabled")
})

test_that("the R names and llama.cpp's names are kept distinct", {
    # "on"/"off" are this package's argument values; llama.cpp calls them
    # "enabled"/"disabled", and the function reports llama.cpp's naming.
    expect_false(llama_flash_attn_type_name("on") == "on")
})

test_that("an unknown type is rejected", {
    expect_error(llama_flash_attn_type_name("enabled"),
                 "must be 'auto', 'on', or 'off'")
    expect_error(llama_flash_attn_type_name("yes"),
                 "must be 'auto', 'on', or 'off'")
})

test_that("a non-scalar type is rejected", {
    expect_error(llama_flash_attn_type_name(c("auto", "on")))
    expect_error(llama_flash_attn_type_name(1L))
})

# --- what a context resolved to ---------------------------------------------

test_that("llama_context_flash_attn reports the resolved state", {
    skip_if_no_model()
    ctx <- ctx_with_fa(environment(), "auto")

    fa <- llama_context_flash_attn(ctx)
    expect_type(fa, "list")
    expect_named(fa, c("enabled", "type_name"))
    expect_type(fa$enabled, "logical")
    expect_false(is.na(fa$enabled))
})

test_that("type_name agrees with the enabled flag", {
    skip_if_no_model()
    ctx <- ctx_with_fa(environment(), "auto")

    fa <- llama_context_flash_attn(ctx)
    expect_equal(fa$type_name, if (fa$enabled) "enabled" else "disabled")
})

test_that("'auto' resolves to a definite answer", {
    # The point of the getter: after "auto" the context holds a decision, not
    # a pending request. (Which way it went depends on model and backend.)
    skip_if_no_model()
    fa <- llama_context_flash_attn(ctx_with_fa(environment(), "auto"))
    expect_true(fa$enabled %in% c(TRUE, FALSE))
})

test_that("asking for 'off' disables flash attention outright", {
    skip_if_no_model()
    fa <- llama_context_flash_attn(ctx_with_fa(environment(), "off"))
    expect_false(fa$enabled)
    expect_equal(fa$type_name, "disabled")
})

test_that("asking for 'on' enables flash attention", {
    skip_if_no_model()
    fa <- llama_context_flash_attn(ctx_with_fa(environment(), "on"))
    expect_true(fa$enabled)
    expect_equal(fa$type_name, "enabled")
})

test_that("a non-context argument is an error, not a crash", {
    # R_ExternalPtrAddr does not type-check its argument, so this needs an
    # explicit EXTPTRSXP guard on the C side; without one it segfaults.
    expect_error(llama_context_flash_attn("not a context"),
                 "invalid context pointer")
    expect_error(llama_context_flash_attn(42), "invalid context pointer")
    expect_error(llama_context_flash_attn(NULL), "invalid context pointer")
})
