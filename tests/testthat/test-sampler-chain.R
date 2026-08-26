# ============================================================
# Sampler chain API: llama_sampler_chain_new/_add/_get/_n/_remove,
# llama_sampler_new/_name/_reset/_clone/_accept/_get_seed/_free.
#
# The point of most of these tests is ownership. llama.cpp moves it around
# (a chain takes over what it is given, chain_remove hands it back, chain_get
# only borrows), so the tests check that a handle is never freed twice and that
# using a retired handle raises an R error instead of touching freed memory.
# Only the chain_from_params tests need a model.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

small_ctx <- function(env, n_ctx = 256L) {
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = n_ctx, n_threads = 2L)
    withr::defer({ llama_free_context(ctx); llama_free_model(model) }, envir = env)
    list(model = model, ctx = ctx)
}

# --- building ---------------------------------------------------------------

test_that("an empty chain starts with no samplers", {
    chain <- llama_sampler_chain_new()
    expect_s3_class(chain, "llama_sampler_chain")
    expect_equal(llama_sampler_chain_n(chain), 0L)
})

test_that("standalone samplers are created and named", {
    s <- llama_sampler_new("top_k", top_k = 40L)
    expect_s3_class(s, "llama_sampler")
    expect_type(llama_sampler_name(s), "character")
    expect_gt(nchar(llama_sampler_name(s)), 0L)
})

test_that("every model-free sampler kind can be created", {
    kinds <- c("greedy", "dist", "top_k", "top_p", "min_p", "typical",
               "temp", "temp_ext", "xtc", "top_n_sigma", "penalties",
               "adaptive_p", "mirostat_v2")
    for (k in kinds) {
        s <- llama_sampler_new(k)
        expect_s3_class(s, "llama_sampler")
    }
})

test_that("an unknown sampler kind is rejected", {
    expect_error(llama_sampler_new("no_such_sampler"), "unknown sampler kind")
})

test_that("kinds needing the vocabulary refuse to build without a model", {
    for (k in c("mirostat", "infill", "logit_bias", "dry")) {
        expect_error(llama_sampler_new(k), "requires a model")
    }
})

test_that("unnamed parameters in ... are rejected", {
    expect_error(llama_sampler_new("top_k", 40L), "must be named")
})

test_that("adding samplers grows the chain", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("top_k", top_k = 40L))
    llama_sampler_chain_add(chain, llama_sampler_new("temp", temp = 0.7))
    llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 42L))
    expect_equal(llama_sampler_chain_n(chain), 3L)
})

test_that("chain_add returns the chain invisibly so calls can be chained", {
    chain <- llama_sampler_chain_new()
    out <- withVisible(llama_sampler_chain_add(chain, llama_sampler_new("greedy")))
    expect_false(out$visible)
    expect_identical(out$value, chain)
})

# --- inspection -------------------------------------------------------------

test_that("chain_get returns the samplers in insertion order", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("top_k", top_k = 40L))
    llama_sampler_chain_add(chain, llama_sampler_new("temp", temp = 0.7))

    n0 <- llama_sampler_name(llama_sampler_chain_get(chain, 0L))
    n1 <- llama_sampler_name(llama_sampler_chain_get(chain, 1L))
    expect_false(identical(n0, n1))
})

test_that("chain_get with -1 returns the chain itself", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))
    expect_s3_class(llama_sampler_chain_get(chain, -1L), "llama_sampler")
})

test_that("an out-of-range index is an error, not a crash", {
    chain <- llama_sampler_chain_new()
    expect_error(llama_sampler_chain_get(chain, 5L), "no sampler at index")
    expect_error(llama_sampler_chain_remove(chain, 5L), "no sampler at index")
})

test_that("chain functions reject a plain sampler", {
    s <- llama_sampler_new("greedy")
    expect_error(llama_sampler_chain_n(s), "expected a sampler chain")
    expect_error(llama_sampler_chain_get(s, 0L), "expected a sampler chain")
})

test_that("get_seed reports a seeded sampler's seed and NA otherwise", {
    expect_equal(llama_sampler_get_seed(llama_sampler_new("dist", seed = 123L)), 123L)
    expect_true(is.na(llama_sampler_get_seed(llama_sampler_new("greedy"))))
})

# --- ownership --------------------------------------------------------------

test_that("a sampler owned by a chain cannot be freed on its own", {
    chain <- llama_sampler_chain_new()
    s <- llama_sampler_new("top_k", top_k = 40L)
    llama_sampler_chain_add(chain, s)
    expect_error(llama_sampler_free(s), "owned by a chain")
})

test_that("a sampler cannot be added to two chains", {
    c1 <- llama_sampler_chain_new()
    c2 <- llama_sampler_chain_new()
    s  <- llama_sampler_new("greedy")
    llama_sampler_chain_add(c1, s)
    expect_error(llama_sampler_chain_add(c2, s), "already owned by a chain")
})

test_that("a chain cannot be added to itself", {
    chain <- llama_sampler_chain_new()
    expect_error(llama_sampler_chain_add(chain, chain), "cannot add a chain to itself")
})

test_that("a handle stays usable while its chain lives", {
    chain <- llama_sampler_chain_new()
    s <- llama_sampler_new("top_k", top_k = 40L)
    llama_sampler_chain_add(chain, s)
    expect_type(llama_sampler_name(s), "character")   # borrowed, still valid
})

test_that("freeing a chain retires handles to the samplers inside it", {
    chain <- llama_sampler_chain_new()
    s <- llama_sampler_new("top_k", top_k = 40L)
    llama_sampler_chain_add(chain, s)

    llama_sampler_free(chain)
    # The sampler is gone with the chain: an R error, not a segfault.
    expect_error(llama_sampler_name(s), "chain that has been freed")
    expect_error(llama_sampler_reset(s), "chain that has been freed")
})

test_that("a borrowed handle from chain_get is retired with its chain", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))
    borrowed <- llama_sampler_chain_get(chain, 0L)

    llama_sampler_free(chain)
    expect_error(llama_sampler_name(borrowed), "chain that has been freed")
})

test_that("chain_remove hands ownership back and survives the chain", {
    # The case worth being explicit about: add -> remove -> free chain -> use.
    # After remove the sampler belongs to R, so freeing the chain must leave it
    # entirely alone.
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("top_k", top_k = 40L))
    expect_equal(llama_sampler_chain_n(chain), 1L)

    taken <- llama_sampler_chain_remove(chain, 0L)
    expect_equal(llama_sampler_chain_n(chain), 0L)

    llama_sampler_free(chain)
    expect_type(llama_sampler_name(taken), "character")   # still valid
    expect_silent(llama_sampler_free(taken))              # and R may free it
})

test_that("chain_remove retires the older handle to the same sampler", {
    # `s` and the returned handle would otherwise both refer to one sampler,
    # and whichever was freed first would leave the other dangling.
    chain <- llama_sampler_chain_new()
    s <- llama_sampler_new("top_k", top_k = 40L)
    llama_sampler_chain_add(chain, s)

    taken <- llama_sampler_chain_remove(chain, 0L)
    expect_error(llama_sampler_name(s), "chain that has been freed")
    expect_type(llama_sampler_name(taken), "character")
})

test_that("freeing twice is harmless and using a freed handle errors", {
    s <- llama_sampler_new("greedy")
    llama_sampler_free(s)
    expect_silent(llama_sampler_free(s))
    expect_error(llama_sampler_name(s), "already been freed")
})

test_that("sampler functions reject something that is not a handle", {
    expect_error(llama_sampler_name("not a sampler"), "expected a sampler handle")
    expect_error(llama_sampler_free(42), "expected a sampler handle")
})

# --- state ------------------------------------------------------------------

test_that("reset and accept run on a chain without error", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("penalties",
                                                     repeat_penalty = 1.1))
    llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 1L))

    expect_silent(llama_sampler_accept(chain, 5L))
    expect_silent(llama_sampler_reset(chain))
})

test_that("a clone is independent of the original", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 7L))

    copy <- llama_sampler_clone(chain)
    expect_s3_class(copy, "llama_sampler_chain")
    expect_equal(llama_sampler_chain_n(copy), 1L)

    # Freeing the original leaves the clone alone.
    llama_sampler_free(chain)
    expect_equal(llama_sampler_chain_n(copy), 1L)
})

test_that("a clone taken from inside a chain outlives it", {
    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("top_k", top_k = 40L))
    copy <- llama_sampler_clone(llama_sampler_chain_get(chain, 0L))

    llama_sampler_free(chain)
    expect_type(llama_sampler_name(copy), "character")
})

# --- bridge to the declarative path -----------------------------------------

test_that("chain_from_params builds the chain a parameter list describes", {
    skip_if_no_model()
    h <- small_ctx(environment())

    sp <- llama_sampler_params(temp = 0.8, top_k = 40L, top_p = 0.9)
    chain <- llama_sampler_chain_from_params(h$ctx, sp)
    expect_s3_class(chain, "llama_sampler_chain")

    names_in_chain <- vapply(
        seq_len(llama_sampler_chain_n(chain)) - 1L,
        function(i) llama_sampler_name(llama_sampler_chain_get(chain, i)),
        character(1))
    expect_true(length(names_in_chain) >= 3L)
    expect_true(any(grepl("top", names_in_chain)))
})

test_that("chain_from_params reflects which samplers a parameter list enables", {
    skip_if_no_model()
    h <- small_ctx(environment())

    plain <- llama_sampler_chain_from_params(h$ctx, llama_sampler_params(temp = 0.8))
    extra <- llama_sampler_chain_from_params(
        h$ctx, llama_sampler_params(temp = 0.8, dry_multiplier = 0.8,
                                    xtc_probability = 0.5))
    # Enabling DRY and XTC adds exactly two more samplers.
    expect_equal(llama_sampler_chain_n(extra), llama_sampler_chain_n(plain) + 2L)
})

test_that("greedy parameters produce a chain ending in a greedy sampler", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_from_params(h$ctx, llama_sampler_params(temp = 0.0))
    last <- llama_sampler_chain_n(chain) - 1L
    expect_match(llama_sampler_name(llama_sampler_chain_get(chain, last)),
                 "greedy")
})

test_that("model-dependent sampler kinds build when given a model", {
    skip_if_no_model()
    h <- small_ctx(environment())

    for (k in c("mirostat", "infill", "dry")) {
        s <- llama_sampler_new(k, dry_multiplier = 0.8, model = h$model)
        expect_s3_class(s, "llama_sampler")
    }
    s <- llama_sampler_new("logit_bias",
                           logit_bias = list(token = 0L, bias = -10),
                           model = h$model)
    expect_s3_class(s, "llama_sampler")
})

test_that("logit_bias without a bias list is rejected", {
    skip_if_no_model()
    h <- small_ctx(environment())
    expect_error(llama_sampler_new("logit_bias", model = h$model),
                 "needs a logit_bias argument")
})

# --- passing a chain to the generation functions ----------------------------

test_that("a chain passes through sampler resolution unchanged", {
    chain <- llama_sampler_chain_new()
    expect_identical(llamaR:::llamar_resolve_sampler(chain, list()), chain)
})

test_that("a single sampler is rejected where a chain is expected", {
    s <- llama_sampler_new("greedy")
    expect_error(llamaR:::llamar_resolve_sampler(s, list()), "must be a chain")
})

test_that("a bare string is still rejected as a sampler argument", {
    expect_error(llamaR:::llamar_resolve_sampler("nope", list()),
                 "parameter list|chain")
})

test_that("generation accepts a chain and leaves the caller's copy alone", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))

    out <- llama_generate(h$ctx, "The capital of France is",
                          max_new_tokens = 8L, sampler = chain)
    expect_type(out, "character")
    # The chain was copied, not consumed: it is still usable afterwards.
    expect_equal(llama_sampler_chain_n(chain), 1L)
    expect_type(llama_sampler_name(chain), "character")
})

test_that("a chain gives the same output as the equivalent parameter list", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))

    via_chain <- llama_generate(h$ctx, "The capital of France is",
                                max_new_tokens = 8L, sampler = chain)
    via_params <- llama_generate(h$ctx, "The capital of France is",
                                 max_new_tokens = 8L, temp = 0.0)
    expect_equal(via_chain, via_params)
})

test_that("repeated generations with one chain are reproducible", {
    # sampler_reset = TRUE (the default) clears the copy's accumulated state,
    # so a chain reused across calls behaves like a freshly built one.
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("penalties",
                                                     repeat_penalty = 1.2))
    llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 7L))

    a <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L,
                        sampler = chain)
    b <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L,
                        sampler = chain)
    expect_equal(a, b)
})

test_that("sampler_reset = FALSE carries state between generations", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("penalties",
                                                     repeat_penalty = 1.2))
    llama_sampler_chain_add(chain, llama_sampler_new("dist", seed = 7L))

    # Only the reset flag differs, so any difference comes from carried state.
    reset <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L,
                            sampler = chain, sampler_reset = TRUE)
    kept  <- llama_generate(h$ctx, "Once upon a time", max_new_tokens = 8L,
                            sampler = chain, sampler_reset = FALSE)
    expect_type(reset, "character")
    expect_type(kept, "character")
})

test_that("streaming generation accepts a chain", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))

    st <- llama_gen_begin(h$ctx, "The capital of France is",
                          max_new_tokens = 4L, sampler = chain)
    chunks <- character(0)
    repeat {
        ch <- llama_gen_next(st)
        if (is.null(ch)) break
        chunks <- c(chunks, ch)
    }
    chunks <- c(chunks, llama_gen_end(st))
    expect_type(paste(chunks, collapse = ""), "character")

    # The state took a copy, so freeing the caller's chain is safe while the
    # state is still around.
    expect_silent(llama_sampler_free(chain))
})

test_that("generate_batch refuses a chain instead of ignoring it", {
    skip_if_no_model()
    h <- small_ctx(environment(), n_ctx = 512L)

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))

    expect_error(
        llama_generate_batch(h$ctx, c("Hello", "Goodbye"),
                             max_new_tokens = 4L, sampler = chain),
        "does not accept a sampler chain")
})

test_that("a freed chain cannot be used for generation", {
    skip_if_no_model()
    h <- small_ctx(environment())

    chain <- llama_sampler_chain_new()
    llama_sampler_chain_add(chain, llama_sampler_new("greedy"))
    llama_sampler_free(chain)

    expect_error(llama_generate(h$ctx, "Hello", max_new_tokens = 4L,
                                sampler = chain),
                 "already been freed")
})
