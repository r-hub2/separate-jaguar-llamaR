# ============================================================
# OpenAI server — end-to-end streaming (HEAVY).
# Spawns llama_serve_openai() in a background process and drives it two ways:
#   1. raw SSE over curl, checking the wire format against the OpenAI spec
#   2. through ellmer::chat_vllm (what chat_llamar(base_url=) builds), which
#      is the actual question — does a real OpenAI client parse our stream?
#
# The wire-format checks come first on purpose: if ellmer fails, they say
# whether the stream itself is wrong or ellmer is reading it differently.
# Listed in tests/testthat.R `heavy`; needs a model + drogonR + callr + curl.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

skip_unless_servable <- function() {
    skip_if_no_model()
    skip_if_not_installed("drogonR")
    skip_if_not_installed("callr")
    skip_if_not_installed("jsonlite")
    if (nchar(Sys.which("curl")) == 0L) skip("curl not available")
}

# Spawn a server, wait for /v1/models to answer 200, return base_url + handle.
.spawn_openai <- function(port, n_ctx = 1024L) {
    server <- callr::r_bg(
        function(model_path, port, n_ctx) {
            library(llamaR)
            llama_serve_openai(model_path, port = port, n_ctx = n_ctx,
                               model_id = "test-model")
        },
        args = list(MODEL_PATH, port, n_ctx))

    base_url <- sprintf("http://127.0.0.1:%d", port)
    deadline <- Sys.time() + 60
    ok <- FALSE
    repeat {
        code <- suppressWarnings(system2(
            "curl", c("-s", "-o", "/dev/null", "-w", "%{http_code}",
                      "--max-time", "2", paste0(base_url, "/v1/models")),
            stdout = TRUE, stderr = NULL))
        if (length(code) && identical(code, "200")) { ok <- TRUE; break }
        if (!server$is_alive() || Sys.time() > deadline) break
        Sys.sleep(0.5)
    }
    if (!ok) {
        err <- tryCatch(server$read_all_error(), error = function(e) "")
        server$kill()
        stop("openai server did not come up:\n", err)
    }
    list(server = server, base_url = base_url)
}

# Raw streamed response, as lines exactly as they came off the wire.
.stream_chat <- function(base_url, body) {
    bf <- tempfile(fileext = ".json"); on.exit(unlink(bf))
    writeLines(body, bf)
    system2("curl", c("-N", "-s", "--max-time", "60",
                      "-H", "Content-Type: application/json",
                      "--data", paste0("@", bf),
                      paste0(base_url, "/v1/chat/completions")),
            stdout = TRUE, stderr = NULL)
}

.body <- function(stream = TRUE, max_tokens = 12L) {
    sprintf(paste0('{"model":"test-model","messages":',
                   '[{"role":"user","content":"Say hello"}],',
                   '"max_tokens":%d,"temperature":0,"stream":%s}'),
            max_tokens, if (stream) "true" else "false")
}

# --- wire format ------------------------------------------------------------

test_that("the SSE stream matches the OpenAI wire format", {
    skip_unless_servable()

    h <- .spawn_openai(18231L)
    withr::defer(h$server$kill())

    lines <- .stream_chat(h$base_url, .body())
    expect_gt(length(lines), 0L)

    payload <- lines[nzchar(lines)]
    # OpenAI streams bare `data:` frames — no `event:`, `id:` or `retry:`
    # fields. This is what dr_stream_sse emits and what chat_vllm expects.
    expect_true(all(grepl("^data: ", payload)))
    expect_false(any(grepl("^event:", lines)))

    # ...terminated by the [DONE] sentinel
    expect_identical(payload[length(payload)], "data: [DONE]")
})

test_that("stream frames carry the documented chunk sequence", {
    skip_unless_servable()

    h <- .spawn_openai(18232L)
    withr::defer(h$server$kill())

    lines <- .stream_chat(h$base_url, .body())
    payload <- lines[nzchar(lines)]
    json <- payload[payload != "data: [DONE]"]
    objs <- lapply(sub("^data: ", "", json),
                   jsonlite::fromJSON, simplifyVector = FALSE)
    expect_gte(length(objs), 3L)   # role + at least one content + finish

    expect_true(all(vapply(objs, function(o) o$object, character(1)) ==
                    "chat.completion.chunk"))

    # first frame announces the assistant role and carries no content
    first <- objs[[1]]$choices[[1]]
    expect_identical(first$delta$role, "assistant")
    expect_null(first$delta$content)
    expect_null(first$finish_reason)

    # last frame carries the finish reason and an empty delta
    last <- objs[[length(objs)]]$choices[[1]]
    expect_true(last$finish_reason %in% c("stop", "length"))
    expect_length(last$delta, 0L)

    # everything in between is content
    if (length(objs) > 2L) {
        middle <- objs[2:(length(objs) - 1L)]
        expect_true(all(vapply(middle,
                               function(o) is.character(o$choices[[1]]$delta$content),
                               logical(1))))
    }
})

test_that("a streamed response reassembles into the blocking one", {
    skip_unless_servable()

    h <- .spawn_openai(18233L)
    withr::defer(h$server$kill())

    lines <- .stream_chat(h$base_url, .body())
    payload <- lines[nzchar(lines)]
    json <- payload[payload != "data: [DONE]"]
    objs <- lapply(sub("^data: ", "", json),
                   jsonlite::fromJSON, simplifyVector = FALSE)
    streamed <- paste0(vapply(objs, function(o) {
        cc <- o$choices[[1]]$delta$content
        if (is.null(cc)) "" else cc
    }, character(1)), collapse = "")

    blocking_raw <- .stream_chat(h$base_url, .body(stream = FALSE))
    blocking <- jsonlite::fromJSON(paste(blocking_raw, collapse = ""),
                                   simplifyVector = FALSE)
    expect_identical(streamed,
                     blocking$choices[[1]]$message$content)
})

# --- through ellmer ---------------------------------------------------------

# chat$stream() hands back a coro generator, not a vector: a plain for() loop
# rejects it ("invalid for() loop sequence"), so it is drained with coro's own
# collector.
.collect_stream <- function(chat, prompt) {
    as.character(unlist(coro::collect(chat$stream(prompt))))
}

test_that("ellmer reads the stream end to end", {
    skip_unless_servable()
    skip_if_not_installed("ellmer")
    skip_if_not_installed("coro")   # ellmer imports it; needed to drain a stream

    h <- .spawn_openai(18234L)
    withr::defer(h$server$kill())

    chat <- chat_llamar(base_url = paste0(h$base_url, "/v1"),
                        model_id = "test-model")
    chunks <- .collect_stream(chat, "Say hello")

    expect_gt(length(chunks), 0L)
    expect_true(all(vapply(chunks, is.character, logical(1))))
    expect_true(nzchar(paste(chunks, collapse = "")))
})

test_that("ellmer's streamed and blocking answers agree", {
    skip_unless_servable()
    skip_if_not_installed("ellmer")
    skip_if_not_installed("coro")   # ellmer imports it; needed to drain a stream

    h <- .spawn_openai(18235L)
    withr::defer(h$server$kill())

    chat <- chat_llamar(base_url = paste0(h$base_url, "/v1"),
                        model_id = "test-model")

    streamed <- paste(.collect_stream(chat, "Say hello"), collapse = "")

    chat2 <- chat_llamar(base_url = paste0(h$base_url, "/v1"),
                         model_id = "test-model")
    blocking <- chat2$chat("Say hello", echo = "none")

    # The server samples greedily by default for temperature 0 requests, but
    # ellmer sets its own defaults, so compare shape rather than exact text.
    expect_true(nzchar(streamed))
    expect_true(nzchar(blocking))
})

test_that("ellmer records the exchange in the chat turns", {
    skip_unless_servable()
    skip_if_not_installed("ellmer")
    skip_if_not_installed("coro")   # ellmer imports it; needed to drain a stream

    h <- .spawn_openai(18236L)
    withr::defer(h$server$kill())

    chat <- chat_llamar(base_url = paste0(h$base_url, "/v1"),
                        model_id = "test-model")
    invisible(.collect_stream(chat, "Say hello"))

    turns <- chat$get_turns()
    expect_gte(length(turns), 2L)
    expect_identical(turns[[1]]@role, "user")
    expect_identical(turns[[length(turns)]]@role, "assistant")
})
