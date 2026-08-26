# ============================================================
# Control vectors and sampler timings (HEAVY).
#   llama_apply_control_vector()  — steering direction on a layer range
#   llama_perf_sampler() / _print() / _reset()
# Both need a real context: a control vector is validated against the
# model's geometry, and sampler timings only exist once a sampler chain
# has run. Listed in tests/testthat.R `heavy`; run with NOT_CRAN=true.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

fresh_ctx <- function(env, n_ctx = 256L) {
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = n_ctx, n_threads = 2L)
    withr::defer({ llama_free_context(ctx); llama_free_model(model) }, envir = env)
    list(model = model, ctx = ctx)
}

# --- control vectors -------------------------------------------------------

test_that("a control vector applies over a layer range", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    info     <- llama_model_info(h$model)
    n_embd   <- info$n_embd
    il_start <- 0L
    il_end   <- min(2L, info$n_layer - 1L)
    n_layers <- il_end - il_start + 1L

    vec <- rep(0.01, n_embd * n_layers)
    expect_silent(llama_apply_control_vector(h$ctx, vec, n_embd, il_start, il_end))
})

test_that("passing NULL clears an applied control vector", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    info   <- llama_model_info(h$model)
    n_embd <- info$n_embd
    il_end <- min(2L, info$n_layer - 1L)

    llama_apply_control_vector(h$ctx, rep(0.01, n_embd * (il_end + 1L)),
                               n_embd, 0L, il_end)
    expect_silent(llama_apply_control_vector(h$ctx, NULL, n_embd, 0L, il_end))
})

test_that("generation still works after a control vector is applied", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    info   <- llama_model_info(h$model)
    n_embd <- info$n_embd
    il_end <- min(2L, info$n_layer - 1L)

    # A zero vector is a no-op mathematically, so the model must still generate.
    llama_apply_control_vector(h$ctx, rep(0, n_embd * (il_end + 1L)),
                               n_embd, 0L, il_end)

    out <- llama_generate(h$ctx, "The capital of France is",
                          max_new_tokens = 4L, temp = 0.0)
    expect_type(out, "character")
})

test_that("a mis-sized control vector is rejected before reaching C++", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    n_embd <- llama_model_info(h$model)$n_embd

    # Length must be n_embd * (il_end - il_start + 1); anything else is caught
    # in R so the user gets a message rather than undefined behaviour.
    expect_error(
        llama_apply_control_vector(h$ctx, rep(0.01, n_embd + 1L), n_embd, 0L, 0L),
        "must have length"
    )
    expect_error(
        llama_apply_control_vector(h$ctx, numeric(0), n_embd, 0L, 0L),
        "must have length"
    )
})

# --- sampler timings -------------------------------------------------------

test_that("llama_perf_sampler reports timings after streaming generation", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    st <- llama_gen_begin(h$ctx, "The capital of France is",
                          max_new_tokens = 8L, temp = 0.0)
    repeat {
        chunk <- llama_gen_next(st)
        if (is.null(chunk)) break
    }
    llama_gen_end(st)

    perf <- llama_perf_sampler(st)
    expect_type(perf, "list")
    expect_true(all(c("t_sample_ms", "n_sample") %in% names(perf)))
    expect_type(perf$t_sample_ms, "double")
    expect_type(perf$n_sample, "integer")
    # Something was sampled, and it took a non-negative amount of time.
    expect_gt(perf$n_sample, 0L)
    expect_gte(perf$t_sample_ms, 0)
})

test_that("llama_perf_sampler_reset zeroes the counters", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    st <- llama_gen_begin(h$ctx, "Two plus two equals",
                          max_new_tokens = 6L, temp = 0.0)
    repeat {
        chunk <- llama_gen_next(st)
        if (is.null(chunk)) break
    }

    expect_gt(llama_perf_sampler(st)$n_sample, 0L)
    llama_perf_sampler_reset(st)
    expect_equal(llama_perf_sampler(st)$n_sample, 0L)
})

test_that("llama_perf_sampler_print runs without error", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    st <- llama_gen_begin(h$ctx, "Hello", max_new_tokens = 4L, temp = 0.0)
    repeat {
        chunk <- llama_gen_next(st)
        if (is.null(chunk)) break
    }

    expect_no_error(llama_perf_sampler_print(st))
})

test_that("sampler timings and context timings count the same generation", {
    skip_if_no_model()
    h <- fresh_ctx(environment())

    st <- llama_gen_begin(h$ctx, "The capital of France is",
                          max_new_tokens = 8L, temp = 0.0)
    n_chunks <- 0L
    repeat {
        chunk <- llama_gen_next(st)
        if (is.null(chunk)) break
        n_chunks <- n_chunks + 1L
    }
    llama_gen_end(st)

    smpl <- llama_perf_sampler(st)
    ctxp <- llama_perf(h$ctx)

    # Sampling happens once per generated token, so the sampler count tracks
    # the decode count rather than exceeding it.
    expect_gt(smpl$n_sample, 0L)
    expect_lte(smpl$n_sample, ctxp$n_eval + 1L)
})

test_that("perf_sampler rejects a non-state pointer", {
    bad <- structure(list(), class = "externalptr")
    expect_error(llama_perf_sampler(bad))
})
