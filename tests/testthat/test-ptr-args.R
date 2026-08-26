# ============================================================
# External-pointer arguments must be type-checked.
#
# R_ExternalPtrAddr() does not check the SEXP type: handed a string, a number
# or a list it returns whatever sits at the pointer field's offset, which the
# usual `if (!p)` guard passes and the code then dereferences — crashing the R
# session rather than raising an error. These tests pin the guards down for one
# entry point per pointer type; a regression here is a segfault, not a failure,
# so the whole file is the canary.
# ============================================================

bad_args <- list(
    string = "not a pointer",
    number = 42,
    list   = list(1, 2),
    null   = NULL,
    lgl    = TRUE
)

# Any error will do — the point is that the call is refused rather than
# crashing R. Some entry points guard on the R side (stopifnot(inherits(...)))
# and others in C, so the message is not pinned down here.
expect_rejects_non_pointers <- function(fn, ...) {
    for (nm in names(bad_args)) {
        expect_error(fn(bad_args[[nm]], ...),
                     info = paste("argument type:", nm))
    }
}

# --- context pointers -------------------------------------------------------

test_that("context entry points reject non-pointers", {
    expect_rejects_non_pointers(llama_n_ctx)
    expect_rejects_non_pointers(llama_n_batch)
    expect_rejects_non_pointers(llama_n_seq_max)
    expect_rejects_non_pointers(llama_context_flash_attn)
    expect_rejects_non_pointers(llama_get_model)
})

test_that("context entry points taking further arguments reject non-pointers", {
    expect_rejects_non_pointers(llama_tokenize, "hello")
    expect_rejects_non_pointers(llama_set_threads, 2L)
})

test_that("free_context rejects non-pointers", {
    # Idempotent for a real handle, but the type still has to be right before
    # anything is written back into it.
    expect_rejects_non_pointers(llama_free_context)
})

# --- model pointers ---------------------------------------------------------

test_that("model entry points reject non-pointers", {
    expect_rejects_non_pointers(llama_model_info)
    expect_rejects_non_pointers(llama_model_meta)
    expect_rejects_non_pointers(llama_vocab_info)
    expect_rejects_non_pointers(llama_model_cls_labels)
})

test_that("free_model rejects non-pointers", {
    expect_rejects_non_pointers(llama_free_model)
})

test_that("chat_build rejects a non-pointer model", {
    expect_error(llama_chat_build("not a model", list(list(role = "user",
                                                           content = "hi"))),
                 "invalid|pointer")
})

# --- generation state pointers ----------------------------------------------

test_that("generation state entry points reject non-pointers", {
    expect_rejects_non_pointers(llama_gen_next)
    expect_rejects_non_pointers(llama_gen_end)
    expect_rejects_non_pointers(llama_perf_sampler)
})

# --- sampler handles --------------------------------------------------------

test_that("sampler entry points reject non-handles", {
    expect_rejects_non_pointers(llama_sampler_name)
    expect_rejects_non_pointers(llama_sampler_reset)
    expect_rejects_non_pointers(llama_sampler_free)
    expect_rejects_non_pointers(llama_sampler_chain_n)
})

test_that("chain functions reject a non-handle chain", {
    s <- llama_sampler_new("greedy")
    expect_error(llama_sampler_chain_add("not a chain", s), "expected|invalid")
    expect_error(llama_sampler_chain_get(42, 0L), "expected|invalid")
})

# --- multimodal pointers ----------------------------------------------------

test_that("mtmd entry points reject non-pointers", {
    skip_if_not(exists("llama_mtmd_support_vision"),
                "mtmd bindings not available")
    expect_rejects_non_pointers(llama_mtmd_support_vision)
})
