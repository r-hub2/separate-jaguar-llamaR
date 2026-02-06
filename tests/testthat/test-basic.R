MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

test_that("package loads correctly", {
    expect_true(require(llamaR, quietly = TRUE))
})

test_that("model loads and info is returned", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    expect_false(is.null(model))

    info <- llama_model_info(model)
    expect_true(is.list(info))
    expect_true(info$n_vocab  > 0)
    expect_true(info$n_embd   > 0)
    expect_true(info$n_layer  > 0)
    expect_true(info$n_head   > 0)
    expect_true(nchar(info$desc) > 0)

    llama_free_model(model)
})

test_that("context can be created and freed", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    expect_false(is.null(ctx))

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("tokenize and detokenize are inverse operations", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    text   <- "Hello, world!"
    tokens <- llama_tokenize(ctx, text)

    expect_true(is.integer(tokens))
    expect_true(length(tokens) > 0)

    recovered <- llama_detokenize(ctx, tokens)
    expect_true(is.character(recovered))
    expect_equal(recovered, text)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("generation produces non-empty output", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 256L, n_threads = 2L)

    result <- llama_generate(ctx, "The capital of France is",
                             max_new_tokens = 20L, temp = 0.1)

    expect_true(is.character(result))
    # use type="bytes" — tiny test model (1 layer) produces garbage that may not be valid UTF-8
    expect_true(nchar(result, type = "bytes") > 0)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("greedy generation is deterministic", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 256L, n_threads = 2L)

    r1 <- llama_generate(ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)
    r2 <- llama_generate(ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)

    expect_equal(r1, r2)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("embeddings have correct dimensionality", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    info  <- llama_model_info(model)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    emb <- llama_embeddings(ctx, "Hello")

    expect_true(is.numeric(emb))
    expect_equal(length(emb), info$n_embd)
    # embeddings should not be all zeros
    expect_true(any(emb != 0))

    llama_free_context(ctx)
    llama_free_model(model)
})

# ============================================================
# Verbosity (no model required)
# ============================================================

test_that("verbosity can be set and retrieved", {
    old <- llama_get_verbosity()

    llama_set_verbosity(0L)
    expect_equal(llama_get_verbosity(), 0L)

    llama_set_verbosity(3L)
    expect_equal(llama_get_verbosity(), 3L)

    # restore original level
    llama_set_verbosity(old)
    expect_equal(llama_get_verbosity(), old)
})

# ============================================================
# GPU support check (no model required)
# ============================================================

test_that("llama_supports_gpu returns logical", {
    result <- llama_supports_gpu()
    expect_true(is.logical(result))
    expect_equal(length(result), 1L)
})

# ============================================================
# Chat templates
# ============================================================

test_that("chat template can be retrieved from model", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    tmpl <- llama_chat_template(model)

    # template may be NULL if model has none, or a string
    expect_true(is.null(tmpl) || is.character(tmpl))

    llama_free_model(model)
})

test_that("chat_apply_template formats messages", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    tmpl <- llama_chat_template(model)
    if (is.null(tmpl)) {
        llama_free_model(model)
        skip("model has no built-in chat template")
    }

    messages <- list(
        list(role = "user", content = "Hello")
    )
    prompt <- llama_chat_apply_template(messages, template = tmpl)

    expect_true(is.character(prompt))
    expect_true(nchar(prompt) > 0)
    # the formatted prompt should contain the original message
    expect_true(grepl("Hello", prompt, fixed = TRUE))

    llama_free_model(model)
})

# ============================================================
# LoRA adapters
# ============================================================

LORA_PATH <- "/mnt/Data2/DS_projects/llm_models/test-lora-adapter.gguf"

test_that("lora_load returns handle or errors on missing file", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)

    # non-existent file should error
    expect_error(llama_lora_load(model, "nonexistent.gguf"))

    if (file.exists(LORA_PATH)) {
        lora <- llama_lora_load(model, LORA_PATH)
        expect_false(is.null(lora))
    }

    llama_free_model(model)
})

test_that("lora_apply and lora_remove work on context", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
    if (!file.exists(LORA_PATH)) skip("test LoRA adapter not available")

    model <- llama_load_model(MODEL_PATH)
    lora <- llama_lora_load(model, LORA_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    # apply should not error
    expect_no_error(llama_lora_apply(ctx, lora, scale = 1.0))

    # remove should return 0 (success)
    result <- llama_lora_remove(ctx, lora)
    expect_equal(result, 0L)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("lora_clear works on context", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
    if (!file.exists(LORA_PATH)) skip("test LoRA adapter not available")

    model <- llama_load_model(MODEL_PATH)
    lora <- llama_lora_load(model, LORA_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    llama_lora_apply(ctx, lora, scale = 0.5)
    expect_no_error(llama_lora_clear(ctx))

    # after clear, remove should return -1 (not applied)
    result <- llama_lora_remove(ctx, lora)
    expect_equal(result, -1L)

    llama_free_context(ctx)
    llama_free_model(model)
})
