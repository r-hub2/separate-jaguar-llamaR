# ============================================================
# Split GGUF support:
#   llama_split_path()   — prefix + chunk numbers -> path
#   llama_split_prefix() — the inverse, when the numbers match
#   llama_load_model_from_splits() — load an explicitly listed set of chunks
# The two path functions are pure string manipulation and need no model; the
# loader is only exercised for its argument validation, since the test model is
# a single file.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

# --- building split paths ---------------------------------------------------

test_that("llama_split_path applies llama.cpp's naming pattern", {
    expect_equal(llama_split_path("/models/ggml-model-q4_0", 2, 4),
                 "/models/ggml-model-q4_0-00002-of-00004.gguf")
})

test_that("split numbers are padded to five digits", {
    expect_equal(llama_split_path("m", 1, 1), "m-00001-of-00001.gguf")
    expect_equal(llama_split_path("m", 12345, 99999),
                 "m-12345-of-99999.gguf")
})

test_that("a relative prefix works as well as an absolute one", {
    expect_equal(llama_split_path("model", 3, 7), "model-00003-of-00007.gguf")
})

test_that("numeric arguments are coerced", {
    expect_equal(llama_split_path("m", 2.0, 4.0), llama_split_path("m", 2L, 4L))
})

test_that("a non-scalar prefix is rejected", {
    expect_error(llama_split_path(c("a", "b"), 1, 2))
    expect_error(llama_split_path(42, 1, 2))
})

test_that("split_no is one-based, matching the number in the file name", {
    # llama.cpp's own function is zero-based and prints split_no + 1; the R
    # interface counts from one, so chunk 1 is the first file.
    expect_equal(llama_split_path("m", 1, 3), "m-00001-of-00003.gguf")
    expect_equal(llama_split_path("m", 3, 3), "m-00003-of-00003.gguf")
})

test_that("an out-of-range split_no is rejected", {
    expect_error(llama_split_path("m", 0, 3), "between 1 and split_count")
    expect_error(llama_split_path("m", 4, 3), "between 1 and split_count")
    expect_error(llama_split_prefix("m-00001-of-00003.gguf", 0, 3),
                 "between 1 and split_count")
})

test_that("a non-positive split_count is rejected", {
    expect_error(llama_split_path("m", 1, 0), "split_count must be at least 1")
})

test_that("NA split numbers are rejected", {
    expect_error(llama_split_path("m", NA_integer_, 3), "must not be NA")
    expect_error(llama_split_path("m", 1, NA_integer_), "must not be NA")
})

# --- recovering the prefix --------------------------------------------------

test_that("llama_split_prefix inverts llama_split_path", {
    expect_equal(
        llama_split_prefix("/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4),
        "/models/ggml-model-q4_0")
})

test_that("path and prefix round-trip for a range of chunk numbers", {
    prefix <- "/tmp/some-model"
    for (i in 1:5) {
        p <- llama_split_path(prefix, i, 5)
        expect_equal(llama_split_prefix(p, i, 5), prefix)
    }
})

test_that("a mismatched chunk number yields NA rather than an error", {
    p <- "/models/ggml-model-q4_0-00002-of-00004.gguf"
    expect_true(is.na(llama_split_prefix(p, 3, 4)))   # wrong split_no
    expect_true(is.na(llama_split_prefix(p, 2, 5)))   # wrong split_count
})

test_that("a path that is not a split path yields NA", {
    expect_true(is.na(llama_split_prefix("/models/plain-model.gguf", 1, 1)))
})

# --- loading from splits ----------------------------------------------------

test_that("an empty or non-character path list is rejected", {
    expect_error(llama_load_model_from_splits(character(0)))
    expect_error(llama_load_model_from_splits(42))
})

test_that("missing split files are reported by name before loading", {
    expect_error(
        llama_load_model_from_splits(c("/no/such/a.gguf", "/no/such/b.gguf")),
        "does not exist")
})

test_that("an invalid split_mode is rejected", {
    skip_if_no_model()
    expect_error(llama_load_model_from_splits(MODEL_PATH, split_mode = "sideways"),
                 "split_mode must be")
})

test_that("a single-file model loads through the splits entry point", {
    # Not a real split, but llama_model_load_from_splits accepts a list of one,
    # which is enough to check that the arguments reach llama.cpp intact.
    skip_if_no_model()
    model <- llama_load_model_from_splits(MODEL_PATH, n_gpu_layers = 0L)
    withr::defer(llama_free_model(model))

    expect_true(inherits(model, "externalptr"))
    info <- llama_model_info(model)
    expect_true(is.list(info))
    expect_gt(info$n_layer, 0)
    expect_gt(info$n_vocab, 0)
})
