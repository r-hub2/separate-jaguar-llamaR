## -----------------------------------------------------------------------------
# library(llamaR)
# llama_backend_devices()   # "Vulkan0", "Vulkan1", ...

## -----------------------------------------------------------------------------
# model <- llama_load_model("model.gguf", n_gpu_layers = -1L,
#                           devices = "Vulkan0", split_mode = "none")
# ctx <- llama_new_context(model, n_ctx = 2048L)

## -----------------------------------------------------------------------------
# # Pipeline: fewest cross-device copies
# model <- llama_load_model("big-model.gguf", n_gpu_layers = -1L,
#                           devices = c("Vulkan0", "Vulkan1"),
#                           split_mode = "layer")
# 
# # Tensor: more parallelism per token, far more traffic
# model <- llama_load_model("big-model.gguf", n_gpu_layers = -1L,
#                           devices = c("Vulkan0", "Vulkan1"),
#                           split_mode = "row")

## -----------------------------------------------------------------------------
# model <- llama_load_model("model.gguf", devices = "Vulkan0", split_mode = "none")

## -----------------------------------------------------------------------------
# # Replica A, in its own process:
# llama_load_model("big.gguf", devices = c("Vulkan0", "Vulkan1"), split_mode = "row")
# # Replica B, in another process:
# llama_load_model("big.gguf", devices = c("Vulkan2", "Vulkan3"), split_mode = "row")

## -----------------------------------------------------------------------------
# system.file("examples", "bench_pp_tp_dp.sh", package = "llamaR")

## -----------------------------------------------------------------------------
# llama_serve_anthropic("model.gguf", port = 11435L, split_mode = "none")

## -----------------------------------------------------------------------------
# llama_serve_anthropic("big-model.gguf", port = 11435L, split_mode = "row")

