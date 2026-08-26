#!/usr/bin/env Rscript
#
# End-to-end check of `tool_calls` in the OpenAI server.
#
# Starts llama_serve_openai() in a background process and drives it through
# the four cases that matter:
#
#   1. blocking, tools offered      -> message.tool_calls + finish_reason
#   2. streaming, tools offered     -> tool_calls in the finishing chunk,
#                                      and no tool markup leaking into the
#                                      streamed text
#   3. the tool-result round trip   -> role="tool" reply feeds back in and
#                                      the model answers in prose
#   4. no tools                     -> nothing changed for plain requests
#
#   Rscript inst/examples/serve_openai_tools.R /path/to/model.gguf
#   Rscript inst/examples/serve_openai_tools.R /path/to/model.gguf 18270 8192
#
# Positional args after the model path are [port] [n_ctx]
# (defaults: port 18270, n_ctx 8192).
#
# Use a model that can actually call tools — Qwen3/Qwen3.5 work well,
# Ministral-3B is weak at it, and tiny test models cannot do it at all.
# Needs the `callr` package and the `curl` command-line tool.

library(llamaR)

`%||%` <- function(a, b) if (is.null(a)) b else a

args       <- commandArgs(trailingOnly = TRUE)
model_path <- if (length(args) >= 1) args[[1]] else
    stop("usage: serve_openai_tools.R <model.gguf> [port] [n_ctx]")
nums  <- suppressWarnings(as.integer(args[!grepl("^--", args)][-1]))
port  <- if (length(nums) >= 1 && !is.na(nums[1])) nums[1] else 18270L
n_ctx <- if (length(nums) >= 2 && !is.na(nums[2])) nums[2] else 8192L

stopifnot(requireNamespace("callr", quietly = TRUE),
          requireNamespace("jsonlite", quietly = TRUE),
          nzchar(Sys.which("curl")))

base_url <- sprintf("http://127.0.0.1:%d", port)
MODEL_ID <- "tools-demo"

server <- callr::r_bg(
    function(model_path, port, n_ctx, model_id) {
        library(llamaR)
        llama_serve_openai(model_path, port = port, n_ctx = n_ctx,
                           model_id = model_id)
    },
    args = list(model_path = model_path, port = port, n_ctx = n_ctx,
                model_id = MODEL_ID))

on.exit(server$kill(), add = TRUE)

message("loading model, waiting for ", base_url, " ...")
ok <- FALSE
deadline <- Sys.time() + 300          # a 9B model takes a while to load
repeat {
    code <- suppressWarnings(system2(
        "curl", c("-s", "-o", "/dev/null", "-w", "%{http_code}",
                  "--max-time", "2", paste0(base_url, "/v1/models")),
        stdout = TRUE, stderr = NULL))
    if (length(code) && identical(code, "200")) { ok <- TRUE; break }
    if (!server$is_alive() || Sys.time() > deadline) break
    Sys.sleep(1)
}
if (!ok) {
    cat(paste(server$read_error_lines(), collapse = "\n"), "\n")
    stop("server did not come up")
}

post <- function(body, stream = FALSE, timeout = "300") {
    bf <- tempfile(fileext = ".json"); on.exit(unlink(bf))
    writeLines(body, bf)
    system2("curl", c(if (stream) "-N", "-s", "--max-time", timeout,
                      "-H", "Content-Type: application/json",
                      "--data", paste0("@", bf),
                      paste0(base_url, "/v1/chat/completions")),
            stdout = TRUE, stderr = NULL)
}

# One tool the model can only answer with by calling it.
TOOLS <- '[{"type":"function","function":{
  "name":"get_weather",
  "description":"Get the current weather in a given city",
  "parameters":{"type":"object",
    "properties":{"city":{"type":"string","description":"City name"}},
    "required":["city"]}}}]'

ask <- '"What is the weather in Paris? Use the tool."'

pass <- function(ok, label, detail = "") {
    cat(sprintf("  [%s] %s%s\n", if (isTRUE(ok)) "PASS" else "FAIL", label,
                if (nzchar(detail)) paste0(" — ", detail) else ""))
    isTRUE(ok)
}
results <- logical(0)

# ---------------------------------------------------------------------------
# 1. Blocking request with tools
# ---------------------------------------------------------------------------
cat("\n== 1. blocking, tools offered ==\n")
body1 <- sprintf('{"model":"%s","messages":[{"role":"user","content":%s}],
                  "tools":%s,"max_tokens":256,"temperature":0}',
                 MODEL_ID, ask, TOOLS)
raw1 <- paste(post(body1), collapse = "")
res1 <- tryCatch(jsonlite::fromJSON(raw1, simplifyVector = FALSE),
                 error = function(e) NULL)
if (is.null(res1)) {
    cat("  raw response:\n", substr(raw1, 1, 500), "\n")
    stop("could not parse the response as JSON")
}
ch1 <- res1$choices[[1]]
tc1 <- ch1$message$tool_calls
cat("  finish_reason:", ch1$finish_reason %||% "<none>", "\n")
if (length(tc1) > 0) {
    for (tc in tc1) {
        cat("  tool call    :", tc$`function`$name,
            "args:", tc$`function`$arguments, "\n")
    }
}
results <- c(results,
    pass(length(tc1) > 0, "a tool call is returned"),
    pass(identical(ch1$finish_reason, "tool_calls"),
         'finish_reason is "tool_calls"', ch1$finish_reason %||% ""),
    pass(length(tc1) > 0 && identical(tc1[[1]]$type, "function"),
         'tool call has type "function"'),
    pass(length(tc1) > 0 && nzchar(tc1[[1]]$id %||% ""),
         "tool call carries an id"),
    pass(length(tc1) > 0 &&
         is.character(tc1[[1]]$`function`$arguments) &&
         !is.null(tryCatch(jsonlite::fromJSON(tc1[[1]]$`function`$arguments),
                           error = function(e) NULL)),
         "arguments are a JSON string"),
    pass(!grepl("<tool_call>|</tool_call>|\\[TOOL_CALLS\\]", raw1),
         "no raw tool markup in the response"))

# ---------------------------------------------------------------------------
# 2. Streaming request with tools
# ---------------------------------------------------------------------------
cat("\n== 2. streaming, tools offered ==\n")
body2 <- sprintf('{"model":"%s","messages":[{"role":"user","content":%s}],
                  "tools":%s,"max_tokens":256,"temperature":0,"stream":true}',
                 MODEL_ID, ask, TOOLS)
lines2  <- post(body2, stream = TRUE)
frames  <- lines2[nzchar(lines2)]
payload <- sub("^data: ", "", frames[frames != "data: [DONE]"])
objs    <- lapply(payload, function(p)
    tryCatch(jsonlite::fromJSON(p, simplifyVector = FALSE),
             error = function(e) NULL))
objs <- Filter(Negate(is.null), objs)

streamed_text <- paste0(vapply(objs, function(o) {
    cc <- o$choices[[1]]$delta$content
    if (is.null(cc) || is.na(cc)) "" else as.character(cc)
}, character(1)), collapse = "")

last2 <- if (length(objs)) objs[[length(objs)]]$choices[[1]] else NULL
tc2   <- last2$delta$tool_calls

cat("  frames        :", length(frames), "\n")
cat("  streamed text :", substr(streamed_text, 1, 120), "\n")
cat("  finish_reason :", last2$finish_reason %||% "<none>", "\n")
if (length(tc2) > 0) {
    for (tc in tc2) {
        cat("  tool call     :", tc$`function`$name,
            "args:", tc$`function`$arguments, "\n")
    }
}
results <- c(results,
    pass(identical(frames[length(frames)], "data: [DONE]"),
         "stream ends with [DONE]"),
    pass(length(tc2) > 0, "finishing chunk carries tool_calls"),
    pass(identical(last2$finish_reason, "tool_calls"),
         'finish_reason is "tool_calls"', last2$finish_reason %||% ""),
    pass(!grepl("<tool_call>|</tool_call>|\\[TOOL_CALLS\\]", streamed_text),
         "no tool markup leaked into the streamed text"),
    pass(!grepl("<think>|</think>", streamed_text),
         "no reasoning markup in the streamed text"))

# ---------------------------------------------------------------------------
# 3. Tool-result round trip
# ---------------------------------------------------------------------------
cat("\n== 3. tool result fed back ==\n")
if (length(tc1) == 0) {
    cat("  skipped: step 1 produced no tool call to answer\n")
} else {
    call_id  <- tc1[[1]]$id
    call_nm  <- tc1[[1]]$`function`$name
    call_arg <- tc1[[1]]$`function`$arguments
    body3 <- sprintf('{"model":"%s","messages":[
        {"role":"user","content":%s},
        {"role":"assistant","content":null,"tool_calls":[
           {"id":"%s","type":"function","function":{"name":"%s","arguments":%s}}]},
        {"role":"tool","tool_call_id":"%s","content":"18 degrees and sunny"}],
      "tools":%s,"max_tokens":128,"temperature":0}',
      MODEL_ID, ask, call_id, call_nm,
      as.character(jsonlite::toJSON(call_arg, auto_unbox = TRUE)),
      call_id, TOOLS)
    raw3 <- paste(post(body3), collapse = "")
    res3 <- tryCatch(jsonlite::fromJSON(raw3, simplifyVector = FALSE),
                     error = function(e) NULL)
    if (is.null(res3)) {
        cat("  raw response:\n", substr(raw3, 1, 500), "\n")
        results <- c(results, pass(FALSE, "round-trip response parses as JSON"))
    } else {
        ch3 <- res3$choices[[1]]
        answer <- ch3$message$content
        cat("  answer       :", substr(answer %||% "", 1, 160), "\n")
        cat("  finish_reason:", ch3$finish_reason %||% "<none>", "\n")
        results <- c(results,
            pass(!is.null(answer) && !is.na(answer) && nzchar(answer),
                 "model answers in prose after the tool result"),
            pass(grepl("18|sunny", answer %||% "", ignore.case = TRUE),
                 "the answer uses the tool result"))
    }
}

# ---------------------------------------------------------------------------
# 4. Plain request (no tools) — the old path must be untouched
# ---------------------------------------------------------------------------
cat("\n== 4. no tools (regression check) ==\n")
# A thinking model spends its first few hundred tokens inside <think>, which
# the server strips: too small a budget yields a legitimately empty answer,
# so give it room to finish reasoning and actually reply.
body4 <- sprintf('{"model":"%s","messages":[{"role":"user","content":"Say hello."}],
                  "max_tokens":1024,"temperature":0}', MODEL_ID)
res4 <- tryCatch(jsonlite::fromJSON(paste(post(body4), collapse = ""),
                                    simplifyVector = FALSE),
                 error = function(e) NULL)
ch4 <- if (!is.null(res4)) res4$choices[[1]] else NULL
cat("  content      :", substr(ch4$message$content %||% "", 1, 120), "\n")
cat("  finish_reason:", ch4$finish_reason %||% "<none>", "\n")
results <- c(results,
    pass(!is.null(ch4) && nzchar(ch4$message$content %||% ""),
         "plain request still answers"),
    pass(!grepl("<think>|</think>", ch4$message$content %||% ""),
         "reasoning is stripped from a plain answer"),
    pass(!is.null(ch4) && is.null(ch4$message$tool_calls),
         "no tool_calls when no tools were offered"),
    pass(!is.null(ch4) && ch4$finish_reason %in% c("stop", "length"),
         'finish_reason is "stop" or "length"'))

body5 <- sprintf('{"model":"%s","messages":[{"role":"user","content":"Say hello."}],
                  "max_tokens":1024,"temperature":0,"stream":true}', MODEL_ID)
lines5 <- post(body5, stream = TRUE)
p5 <- lines5[nzchar(lines5)]
results <- c(results,
    pass(length(p5) > 2 && identical(p5[length(p5)], "data: [DONE]"),
         "plain streaming still works"))

# ---------------------------------------------------------------------------
cat(sprintf("\n== %d/%d checks passed ==\n", sum(results), length(results)))
cat("shutting down server.\n")
if (!all(results)) quit(save = "no", status = 1)
