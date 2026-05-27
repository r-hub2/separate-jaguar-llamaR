## ----setup, include=FALSE-----------------------------------------------------
# Every chunk needs a GGUF model (and usually a GPU), so this vignette is
# static: the code is shown but not run at build time.
knitr::opts_chunk$set(eval = FALSE)

## -----------------------------------------------------------------------------
# library(llamaR)

## -----------------------------------------------------------------------------
# chat <- chat_llamar(model_path = "Ministral-3B-Instruct.gguf")
# 
# chat$chat("Why is the sky blue?")
# 
# chat_llamar_stop(chat)   # stop the spawned server (or just let GC do it)

## -----------------------------------------------------------------------------
# chat <- chat_llamar(model_path = "Qwen3-14B-Q8_0.gguf", timeout = 300)

## -----------------------------------------------------------------------------
# # In another process / shell:
# #   llama_serve_openai("model.gguf", port = 11434L)
# 
# chat <- chat_llamar(base_url = "http://127.0.0.1:11434/v1")
# chat$chat("Hello!")

## -----------------------------------------------------------------------------
# chat <- chat_llamar(
#   model_path    = "Ministral-3B-Instruct.gguf",
#   system_prompt = "You are a concise assistant. Answer in one sentence."
# )
# chat$chat("What is R?")

## -----------------------------------------------------------------------------
# llama_serve_openai("model.gguf", port = 11434L, n_ctx = 8192L)

## -----------------------------------------------------------------------------
# library(ragnar)
# 
# store <- ragnar_store_create(
#   location = "store.duckdb",
#   embed    = embed_llamar(model = "embedding-model.gguf")
# )
# ragnar_store_insert(store, documents)
# ragnar_store_build_index(store)
# 
# chat <- chat_llamar(model_path = "Ministral-3B-Instruct.gguf")
# ragnar_register_tool_retrieve(chat, store)
# chat$chat("What do the documents say about X?")

## -----------------------------------------------------------------------------
# ports <- c(11434L, 11435L, 11436L)
# chats <- lapply(ports, function(p)
#   chat_llamar(base_url = sprintf("http://127.0.0.1:%d/v1", p)))

