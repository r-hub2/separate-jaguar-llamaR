# ============================================================
# Sampler parameter list and the shared chain builder.
#   llama_sampler_params()  — the parameter list itself
#   sampler = <list>        — on generate / gen_begin / generate_batch
# The list-building tests need no model; the generation tests guard
# themselves with skip_if_no_model().
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

small_ctx <- function(env, n_ctx = 256L, n_seq_max = 1L) {
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = n_ctx, n_threads = 2L,
                               n_seq_max = n_seq_max)
    withr::defer({ llama_free_context(ctx); llama_free_model(model) }, envir = env)
    ctx
}

# --- the parameter list ----------------------------------------------------

test_that("llama_sampler_params returns every documented field", {
    sp <- llama_sampler_params()
    expect_type(sp, "list")
    expect_true(all(c("temp", "top_k", "top_p", "min_p", "typical_p", "seed",
                      "min_keep", "repeat_penalty", "repeat_last_n",
                      "frequency_penalty", "presence_penalty",
                      "mirostat", "mirostat_tau", "mirostat_eta",
                      "dynatemp_range", "dynatemp_exponent",
                      "xtc_probability", "xtc_threshold", "top_n_sigma",
                      "dry_multiplier", "dry_base", "dry_allowed_length",
                      "dry_penalty_last_n", "dry_sequence_breakers",
                      "adaptive_target", "adaptive_decay",
                      "infill", "logit_bias") %in% names(sp)))
})

test_that("defaults disable every optional sampler", {
    sp <- llama_sampler_params()
    expect_equal(sp$dry_multiplier, 0)     # DRY off
    expect_equal(sp$xtc_probability, 0)    # XTC off
    expect_lt(sp$top_n_sigma, 0)           # top-n-sigma off
    expect_lt(sp$adaptive_target, 0)       # adaptive-p off
    expect_equal(sp$dynatemp_range, 0)     # plain temperature
    expect_false(sp$infill)
    expect_null(sp$logit_bias)
})

test_that("numeric arguments are coerced to the types the C layer expects", {
    sp <- llama_sampler_params(temp = 1, top_k = 40, seed = 7,
                               repeat_last_n = 32, dry_allowed_length = 3)
    expect_type(sp$temp, "double")
    expect_type(sp$top_k, "integer")
    expect_type(sp$seed, "integer")
    expect_type(sp$repeat_last_n, "integer")
    expect_type(sp$dry_allowed_length, "integer")
})

test_that("DRY sequence breakers default to the upstream set and are overridable", {
    expect_equal(llama_sampler_params()$dry_sequence_breakers,
                 c("\n", ":", "\"", "*"))
    expect_equal(llama_sampler_params(dry_sequence_breakers = c("|", "#"))$dry_sequence_breakers,
                 c("|", "#"))
})

test_that("logit_bias is normalized and validated", {
    sp <- llama_sampler_params(logit_bias = list(token = c(5, 9), bias = c(-1, 2)))
    expect_type(sp$logit_bias$token, "integer")
    expect_type(sp$logit_bias$bias, "double")

    expect_error(llama_sampler_params(logit_bias = list(token = 1L)),
                 "token.*bias")
    expect_error(llama_sampler_params(logit_bias = list(token = c(1L, 2L), bias = 3)),
                 "same length")
})

# --- wiring into the generation functions ----------------------------------

test_that("a non-list sampler argument is rejected", {
    # The accepted forms are a parameter list or a chain handle; the message
    # names both. Chain-specific cases live in test-sampler-chain.R.
    expect_error(llamaR:::llamar_resolve_sampler("nope", list()),
                 "parameter list.*or a chain")
})

test_that("sampler = NULL folds the flat arguments into a parameter list", {
    sp <- llamaR:::llamar_resolve_sampler(NULL, list(temp = 0.3, top_k = 11L))
    expect_equal(sp$temp, 0.3)
    expect_equal(sp$top_k, 11L)
    # untouched fields keep llama_sampler_params()'s defaults
    expect_equal(sp$top_p, llama_sampler_params()$top_p)
})

test_that("an explicit sampler list overrides the flat arguments", {
    sp <- llamaR:::llamar_resolve_sampler(llama_sampler_params(temp = 0.1),
                                          list(temp = 0.9))
    expect_equal(sp$temp, 0.1)
})

test_that("greedy sampling is reproducible through the sampler argument", {
    skip_if_no_model()
    ctx <- small_ctx(environment())

    sp <- llama_sampler_params(temp = 0.0)
    a <- llama_generate(ctx, "The capital of France is",
                        max_new_tokens = 8L, sampler = sp)
    b <- llama_generate(ctx, "The capital of France is",
                        max_new_tokens = 8L, sampler = sp)
    expect_equal(a, b)

    # same chain reached through the flat arguments
    c_flat <- llama_generate(ctx, "The capital of France is",
                             max_new_tokens = 8L, temp = 0.0)
    expect_equal(a, c_flat)
})

test_that("each newly bound sampler builds a working chain", {
    skip_if_no_model()
    ctx <- small_ctx(environment())

    variants <- list(
        dry         = llama_sampler_params(temp = 0.8, dry_multiplier = 0.8),
        xtc         = llama_sampler_params(temp = 0.8, xtc_probability = 0.5),
        dynatemp    = llama_sampler_params(temp = 0.8, dynatemp_range = 0.5),
        top_n_sigma = llama_sampler_params(temp = 0.8, top_n_sigma = 1.0),
        adaptive_p  = llama_sampler_params(temp = 0.8, adaptive_target = 0.1),
        infill      = llama_sampler_params(temp = 0.8, infill = TRUE),
        logit_bias  = llama_sampler_params(temp = 0.8,
                                           logit_bias = list(token = 0L, bias = -50))
    )

    for (nm in names(variants)) {
        out <- llama_generate(ctx, "Hello", max_new_tokens = 4L,
                              sampler = variants[[nm]])
        expect_type(out, "character")
        expect_length(out, 1L)
    }
})

test_that("logit_bias can suppress a token that greedy decoding would pick", {
    skip_if_no_model()
    ctx <- small_ctx(environment())

    prompt <- "The capital of France is"
    baseline <- llama_generate(ctx, prompt, max_new_tokens = 1L, temp = 0.0)

    # Re-tokenizing the generated text can split it differently than the token
    # actually sampled, so bias every token the text maps to; suppressing a
    # superset of the greedy choice is still a valid test of the binding.
    tok <- llama_tokenize(ctx, baseline, add_special = FALSE)
    skip_if(length(tok) == 0L, "baseline produced no token")

    sp <- llama_sampler_params(
        temp = 0.0,
        logit_bias = list(token = tok, bias = rep(-100, length(tok))))
    biased <- llama_generate(ctx, prompt, max_new_tokens = 1L, sampler = sp)
    expect_false(identical(biased, baseline))
})

test_that("mirostat still works and now reaches generate_batch too", {
    skip_if_no_model()
    ctx <- small_ctx(environment(), n_ctx = 512L, n_seq_max = 2L)

    out <- llama_generate(ctx, "Hello", max_new_tokens = 4L,
                          sampler = llama_sampler_params(temp = 0.8, mirostat = 2L))
    expect_type(out, "character")

    batch <- llama_generate_batch(ctx, c("Hello", "Goodbye"),
                                  max_new_tokens = 4L, mirostat = 2L)
    expect_length(batch, 2L)
    expect_type(batch[[1]]$text, "character")
})

test_that("generate_batch accepts a sampler list and keeps sequences independent", {
    skip_if_no_model()
    ctx <- small_ctx(environment(), n_ctx = 512L, n_seq_max = 2L)

    sp <- llama_sampler_params(temp = 0.0, dry_multiplier = 0.5)
    out <- llama_generate_batch(ctx, c("Hello", "Goodbye"),
                                max_new_tokens = 4L, sampler = sp)
    expect_length(out, 2L)
    for (r in out) {
        expect_type(r$text, "character")
        expect_true(r$finished_reason %in% c("eos", "max_tokens"))
    }
})

test_that("streaming generation accepts a sampler list", {
    skip_if_no_model()
    ctx <- small_ctx(environment())

    st <- llama_gen_begin(ctx, "The capital of France is", max_new_tokens = 4L,
                          sampler = llama_sampler_params(temp = 0.0))
    chunks <- character(0)
    repeat {
        ch <- llama_gen_next(st)
        if (is.null(ch)) break
        chunks <- c(chunks, ch)
    }
    chunks <- c(chunks, llama_gen_end(st))
    expect_type(paste(chunks, collapse = ""), "character")
})
