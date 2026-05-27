## ----setup, include=FALSE-----------------------------------------------------
# Every chunk needs a GGUF model (and usually a GPU), so this vignette is
# static: the code is shown but not run at build time.
knitr::opts_chunk$set(eval = FALSE)

## -----------------------------------------------------------------------------
# library(llamaR)

## -----------------------------------------------------------------------------
# # List the GGUF files in a repo
# llama_hf_list("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
# 
# # Download one (by filename or by quantization pattern)
# path <- llama_hf_download(
#   "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
#   pattern = "Q4_K_M"
# )

## -----------------------------------------------------------------------------
# model <- llama_load_model(path, n_gpu_layers = -1L)   # -1 = offload all layers
# ctx   <- llama_new_context(model, n_ctx = 4096L)
# 
# llama_model_info(model)   # size, n_params, context length, heads, ...

## -----------------------------------------------------------------------------
# llama_generate(ctx, "The capital of France is", max_new_tokens = 32L)

## -----------------------------------------------------------------------------
# llama_generate(
#   ctx, "Write a haiku about autumn.",
#   max_new_tokens = 64L,
#   temp           = 0.7,
#   top_p          = 0.9,
#   top_k          = 40L,
#   repeat_penalty = 1.1
# )

## -----------------------------------------------------------------------------
# messages <- list(
#   list(role = "system",    content = "You are a helpful assistant."),
#   list(role = "user",      content = "Name three primary colors.")
# )
# 
# prompt <- llama_chat_apply_template(messages)   # uses the model's built-in template
# llama_generate(ctx, prompt, max_new_tokens = 64L)

## -----------------------------------------------------------------------------
# tokens <- llama_tokenize(ctx, "Hello, world!")
# tokens
# 
# llama_detokenize(ctx, tokens)

## -----------------------------------------------------------------------------
# prompt <- llama_chat_apply_template(list(list(role = "user", content = "hi")))
# llama_tokenize(ctx, prompt, parse_special = TRUE)

## -----------------------------------------------------------------------------
# emb_model <- llama_load_model("embedding-model.gguf")
# emb_ctx   <- llama_new_context(emb_model, embedding = TRUE)
# 
# v <- llama_embeddings(emb_ctx, "The quick brown fox")
# length(v)

## -----------------------------------------------------------------------------
# m <- llama_embed_batch(emb_ctx, c("first text", "second text", "third text"))
# dim(m)   # one row per input

## -----------------------------------------------------------------------------
# library(ragnar)
# 
# store <- ragnar_store_create(
#   location = "store.duckdb",
#   embed    = embed_llamar(model = "embedding-model.gguf", n_gpu_layers = -1L)
# )
# ragnar_store_insert(store, documents)
# ragnar_store_build_index(store)
# ragnar_retrieve(store, "search query")

