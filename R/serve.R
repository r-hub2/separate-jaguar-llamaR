# OpenAI-compatible HTTP server on top of drogonR + llamaR streaming.
# See llama_serve_openai() below.

`%||%` <- function(a, b) if (is.null(a)) b else a

# Build a non-stream chat.completion JSON body (matches llama-server's
# to_json_oaicompat_chat shape closely enough for OpenAI clients).
.openai_completion <- function(id, model_id, text, created,
                               prompt_tokens, completion_tokens, finish_reason,
                               tool_calls = NULL) {
    msg <- list(role = "assistant", content = text)
    # OpenAI sends content as JSON null (not "") when the reply is only tool
    # calls, and carries the calls alongside it.
    if (length(tool_calls) > 0) {
        msg$content    <- if (nzchar(text)) text else NA
        msg$tool_calls <- tool_calls
    }
    list(
        id      = id,
        object  = "chat.completion",
        created = created,
        model   = model_id,
        choices = list(list(
            index         = 0L,
            message       = msg,
            finish_reason = finish_reason
        )),
        usage = list(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens
        )
    )
}

# One streaming chunk (object = chat.completion.chunk). `delta` is a named
# list; a finishing chunk passes delta = list() and a finish_reason. A
# non-finishing chunk leaves finish_reason = NA, which serialises to JSON
# null (the OpenAI shape); NULL would be dropped by jsonlite.
.openai_chunk <- function(id, model_id, created, delta, finish_reason = NA) {
    list(
        id      = id,
        object  = "chat.completion.chunk",
        created = created,
        model   = model_id,
        choices = list(list(
            index         = 0L,
            delta         = delta,
            finish_reason = finish_reason
        ))
    )
}

# llama_chat_parse() returns tool calls as a data.frame (id, name, arguments);
# turn it into OpenAI's tool_calls array. `arguments` stays a JSON *string*,
# which is what the OpenAI schema specifies.
.openai_tool_calls <- function(parsed) {
    tc <- parsed$tool_calls
    if (is.null(tc) || nrow(tc) == 0) return(NULL)
    lapply(seq_len(nrow(tc)), function(i) {
        list(
            id   = if (nzchar(tc$id[i])) tc$id[i] else
                   paste0("call_", paste0(sample(c(0:9, letters), 20, replace = TRUE),
                                          collapse = "")),
            type = "function",
            `function` = list(
                name      = tc$name[i],
                arguments = tc$arguments[i]
            )
        )
    })
}

# The finish_reason OpenAI expects: tool_calls when the model asked for one,
# length when the token budget ran out, stop otherwise.
.openai_finish_reason <- function(parsed, hit_limit) {
    if (!is.null(parsed$tool_calls) && nrow(parsed$tool_calls) > 0) return("tool_calls")
    if (hit_limit) return("length")
    "stop"
}

#' Serve an OpenAI-compatible HTTP API for a local model
#'
#' Loads a GGUF model once and exposes it over an OpenAI-compatible HTTP
#' API so any OpenAI client (OpenCode, ellmer, the `openai` Python SDK, …)
#' can talk to it. Implements `GET /v1/models` and
#' `POST /v1/chat/completions` (both blocking and `stream = true`). The
#' HTTP/SSE layer is provided by \pkg{drogonR}; generation runs through
#' llamaR's streaming API (\code{\link{llama_gen_begin}} /
#' \code{\link{llama_gen_next}} / \code{\link{llama_gen_end}}).
#'
#' The server is single-sequence: requests are handled one at a time on the
#' main R thread (each streamed token is one event-loop pump). This is meant
#' for a single local user/agent, not concurrent load.
#'
#' \code{drogonR} is an optional dependency (\code{Suggests}); install it
#' with \code{install.packages("drogonR")} (or from its repository) before
#' calling this function.
#'
#' @param model_path Path to a GGUF model file.
#' @param port Port to listen on. Default \code{11434} (the Ollama port, so
#'   clients pointed at a local Ollama work unchanged).
#' @param n_ctx Context size for the loaded model.
#' @param n_gpu_layers Layers to offload to GPU (\code{-1} = all).
#' @param model_id Identifier reported in \code{/v1/models} and echoed in
#'   responses. Defaults to the model file's base name.
#' @param host Address to bind. Default \code{"127.0.0.1"} (local only).
#' @param template Chat template string, or \code{NULL} to use the model's
#'   built-in template.
#' @param max_tokens Default \code{max_new_tokens} when a request omits it.
#' @param strip_thinking Drop \code{<think>...</think>} reasoning from the
#'   answer, so a thinking model returns its conclusion rather than its
#'   monologue. \code{TRUE} by default, since OpenAI clients expect the reply
#'   itself; set \code{FALSE} to pass the reasoning through untouched.
#' @param ... Reserved for future options.
#'
#' @return Invisibly \code{NULL}. Blocks serving until \code{drogonR::dr_stop()}
#'   is called (typically from another process or an interrupt).
#' @seealso [llama_gen_begin], [llama_generate]
#' @export
#' @examples
#' \dontrun{
#' llama_serve_openai("model.gguf", port = 11434L)
#' # In another shell, point any OpenAI client at
#' #   http://127.0.0.1:11434/v1
#' # e.g. GET /v1/models and POST /v1/chat/completions
#' }
llama_serve_openai <- function(model_path, port = 11434L,
                               n_ctx = 4096L, n_gpu_layers = -1L,
                               model_id = NULL, host = "127.0.0.1",
                               template = NULL, max_tokens = 512L,
                               strip_thinking = TRUE, ...) {
    if (!requireNamespace("drogonR", quietly = TRUE)) {
        stop("llama_serve_openai() requires the 'drogonR' package: ",
             "install.packages('drogonR')", call. = FALSE)
    }
    if (!file.exists(model_path)) {
        stop("model file not found: ", model_path, call. = FALSE)
    }
    if (is.null(model_id)) {
        model_id <- tools::file_path_sans_ext(basename(model_path))
    }

    # --- load model once; keep model/ctx alive for the server's lifetime ---
    model <- llama_load_model(model_path, n_gpu_layers = as.integer(n_gpu_layers))
    ctx   <- llama_new_context(model, n_ctx = as.integer(n_ctx))
    ctx_size <- llama_n_ctx(ctx)
    chat_template <- template %||% tryCatch(llama_chat_template(model),
                                            error = function(e) NULL)

    # A message's `content` is either a plain string or, equally valid in the
    # OpenAI schema, an array of typed parts: [{"type":"text","text":"..."}].
    # ellmer sends the array form, so flatten it to the string the chat
    # template needs, keeping only the text parts.
    flatten_content <- function(content) {
        if (is.null(content)) return("")
        if (is.character(content)) return(paste(content, collapse = ""))
        if (is.data.frame(content)) {
            txt <- content$text[!is.na(content$text)]
            return(paste(txt, collapse = ""))
        }
        if (is.list(content)) {
            parts <- vapply(content, function(p) {
                if (is.character(p)) return(paste(p, collapse = ""))
                if (is.list(p) && !is.null(p$text)) return(as.character(p$text))
                ""
            }, character(1))
            return(paste(parts, collapse = ""))
        }
        as.character(content)
    }

    # Normalise an OpenAI `messages` array for the tool-aware chat layer. The
    # body is parsed with simplifyVector = FALSE, so each message is a plain
    # list; only `content` needs flattening. Assistant `tool_calls` and the
    # `tool_call_id` on tool replies are passed through untouched — the
    # vendored chat layer understands both natively.
    normalise_messages <- function(messages) {
        if (is.data.frame(messages)) {
            messages <- lapply(seq_len(nrow(messages)), function(i) {
                as.list(messages[i, , drop = FALSE])
            })
        }
        lapply(messages, function(m) {
            out <- list(role = as.character(m$role %||% "user"),
                        content = flatten_content(m$content))
            if (!is.null(m$tool_calls))   out$tool_calls   <- m$tool_calls
            if (!is.null(m$tool_call_id)) out$tool_call_id <- as.character(m$tool_call_id)
            if (!is.null(m$name))         out$name         <- as.character(m$name)
            out
        })
    }

    # Prompt for a plain (tool-free) request: the chat template, as before.
    build_prompt <- function(messages) {
        llama_chat_apply_template(normalise_messages(messages),
                                  template = chat_template,
                                  add_generation_prompt = TRUE)
    }

    # Extract sampling params from a parsed request body, applying defaults.
    gen_args <- function(body) {
        list(
            max_new_tokens = as.integer(body$max_tokens %||% max_tokens),
            temp           = as.double(body$temperature %||% 0.8),
            top_p          = as.double(body$top_p %||% 0.9),
            seed           = as.integer(body$seed %||% 42L)
        )
    }

    new_id <- function() paste0("chatcmpl-", paste0(
        sample(c(0:9, letters), 24, replace = TRUE), collapse = ""))

    app <- drogonR::dr_app()

    # --- GET /v1/models ---
    app <- drogonR::dr_get(app, "/v1/models", function(req) {
        drogonR::dr_json(list(
            object = "list",
            data = list(list(
                id       = model_id,
                object   = "model",
                created  = as.integer(Sys.time()),
                owned_by = "llamaR"
            ))
        ))
    })

    # --- POST /v1/chat/completions ---
    app <- drogonR::dr_post(app, "/v1/chat/completions", function(req) {
        # simplifyVector = FALSE keeps JSON arrays as lists: a single-element
        # "required":["city"] would otherwise collapse to a scalar and objects
        # to data.frames, which corrupts a tool's parameter schema on its way
        # to the C++ JSON-schema converter.
        body <- jsonlite::fromJSON(drogonR::dr_body(req, as = "text") %||% "null",
                                   simplifyVector = FALSE)
        if (is.null(body$messages)) {
            return(drogonR::dr_json(
                list(error = list(message = "missing 'messages'", type = "invalid_request_error")),
                status = 400L))
        }

        messages <- normalise_messages(body$messages)
        tools    <- if (length(body$tools) > 0) body$tools else NULL
        # tool_choice is either a string ("auto"/"none"/"required") or an
        # object naming a function; the chat layer takes the string form.
        tool_choice <- if (is.null(body$tool_choice)) NULL
                       else if (is.character(body$tool_choice)) body$tool_choice
                       else body$tool_choice$`function`$name %||% NULL

        # With tools, the prompt/grammar/parser come from the tool-aware chat
        # layer; without them, the plain chat template is enough and cheaper.
        built <- NULL
        if (!is.null(tools)) {
            built <- llama_chat_build(model, messages, tools = tools,
                                      tool_choice = tool_choice)
            prompt <- built$prompt
        } else {
            prompt <- build_prompt(body$messages)
        }

        args    <- gen_args(body)
        if (!is.null(built) && nzchar(built$grammar)) {
            args$grammar <- built$grammar
            # Triggers apply only to lazy grammars (see llama_chat_build).
            if (isTRUE(built$grammar_lazy)) {
                args$trigger_patterns <- built$trigger_patterns
                args$trigger_tokens   <- built$trigger_tokens
            }
        }
        created <- as.integer(Sys.time())
        id      <- new_id()
        # parse_special = TRUE to match the generation path's tokenization
        # (role markers are control tokens), so this count matches reality.
        prompt_tokens <- length(llama_tokenize(ctx, prompt, parse_special = TRUE))
        stream  <- isTRUE(body$stream)

        # Parse a finished (or partial) generation through the tool-aware layer
        # when tools are in play; otherwise the raw text is the answer. Either
        # way the reasoning of a thinking model is dropped unless the caller
        # asked to keep it — a client asking for a chat completion wants the
        # answer, not the model's <think> monologue.
        parse_out <- function(raw, is_partial = FALSE) {
            out <- if (is.null(built)) list(content = raw, tool_calls = NULL)
                   else llama_chat_parse(raw, format = built$format,
                                         is_partial = is_partial,
                                         parser = built$parser)
            if (isTRUE(strip_thinking)) {
                out$content <- .strip_thinking(out$content %||% "")
            }
            out
        }

        # Reject prompts that don't leave room to generate, instead of
        # silently returning empty content: once the prompt fills the KV
        # cache the very first sample is EOG. Mirror llama-server's 400.
        if (prompt_tokens + args$max_new_tokens > ctx_size) {
            return(drogonR::dr_json(
                list(error = list(
                    message = sprintf(paste0(
                        "prompt is too long: %d tokens + %d requested ",
                        "exceed the context window of %d. Increase n_ctx ",
                        "when starting the server or shorten the input."),
                        prompt_tokens, args$max_new_tokens, ctx_size),
                    type = "invalid_request_error",
                    code = "context_length_exceeded")),
                status = 400L))
        }

        if (!stream) {
            # blocking: drain the whole generation into one string
            st <- do.call(llama_gen_begin, c(list(ctx, prompt), args))
            chunks <- character(0)
            n_completion <- 0L
            repeat {
                chunk <- llama_gen_next(st)
                if (is.null(chunk)) break
                chunks <- c(chunks, chunk)
                n_completion <- n_completion + 1L
            }
            chunks <- c(chunks, llama_gen_end(st))
            raw <- paste0(chunks, collapse = "")
            parsed <- parse_out(raw)
            tcs    <- .openai_tool_calls(parsed)
            finish <- .openai_finish_reason(parsed,
                                            n_completion >= args$max_new_tokens)
            return(drogonR::dr_json(.openai_completion(
                id, model_id, parsed$content %||% "", created,
                prompt_tokens, n_completion, finish, tool_calls = tcs)))
        }

        # streaming: one SSE event per generated token
        st <- do.call(llama_gen_begin, c(list(ctx, prompt), args))
        state <- new.env(parent = emptyenv())
        state$role_sent <- FALSE
        state$flushed   <- FALSE   # finish chunk emitted
        state$n         <- 0L
        state$finishing <- FALSE
        state$text      <- character(0)   # raw generation so far
        state$emitted   <- ""             # parsed content already streamed
        state$since_parse <- 0L
        state$tool_frozen <- FALSE
        state$parsed      <- NULL         # final parse, for the finish chunk
        # The GENERIC tool format (id 1) wraps the whole reply in one JSON
        # object, so no prefix of it is valid content and it cannot be streamed
        # live: accumulate silently and emit everything at the end. The other
        # tool formats start as ordinary prose and only later emit markup, so
        # they do stream.
        defer <- !is.null(built) && identical(as.integer(built$format), 1L)
        # A raw token can be streamed as-is only when nothing has to be
        # filtered out of it. With tools the parser holds back markup, and with
        # strip_thinking it holds back reasoning, so both need the incremental
        # parse path below.
        raw_ok <- is.null(built) && !isTRUE(strip_thinking)

        emit <- function(obj, s) {
            list(data = as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                 state = s, done = FALSE)
        }

        # NB: every return must carry a real SSE payload. dr_stream_sse turns
        # data = "" into a `data:` frame with an empty body, which an OpenAI
        # client then tries to parse as JSON and fails on ("premature EOF").
        # So where generation produces nothing to send yet — accumulating
        # between parses, or after a tool call froze the text — the generator
        # loops and pulls the next token instead of returning.
        generator <- function(s, cancelled) {
            if (cancelled) {
                llama_gen_end(st)
                return(list(data = "", state = s, done = TRUE))
            }
            # first event carries the assistant role with empty content
            if (!state$role_sent) {
                state$role_sent <- TRUE
                return(emit(.openai_chunk(id, model_id, created,
                                          list(role = "assistant")), s))
            }
            while (!state$finishing) {
                chunk <- llama_gen_next(st)
                if (!is.null(chunk)) {
                    state$n <- state$n + 1L

                    # Nothing to filter: the raw token IS the content.
                    if (raw_ok) {
                        return(emit(.openai_chunk(id, model_id, created,
                                                  list(content = chunk)), s))
                    }

                    state$text <- c(state$text, chunk)
                    # Deferred formats, and everything after a tool call has
                    # surfaced, produce no live text: keep generating.
                    if (defer || isTRUE(state$tool_frozen)) next

                    # Stream from a partial parse of the accumulated raw text,
                    # never from the raw token: the parser strips reasoning and
                    # holds back tool markup, so only genuine content reaches
                    # the client. Parsing every token would rescan a nearly
                    # identical string each step, so throttle: parse on a
                    # newline, on a space/punctuation boundary once a couple of
                    # tokens have accrued, else force one every 8th token.
                    state$since_parse <- state$since_parse + 1L
                    do_parse <- grepl("\n", chunk, fixed = TRUE) ||
                                (grepl("[[:space:][:punct:]]", chunk) &&
                                 state$since_parse >= 2L) ||
                                state$since_parse >= 8L
                    if (!do_parse) next
                    state$since_parse <- 0L

                    parsed <- parse_out(paste0(state$text, collapse = ""),
                                        is_partial = TRUE)
                    # A tool call surfacing means the tail is markup, not prose:
                    # freeze the text stream and let the finish chunk carry the
                    # calls, so the markup never leaks to the client.
                    if (!is.null(parsed$tool_calls) && nrow(parsed$tool_calls) > 0) {
                        state$tool_frozen <- TRUE
                        next
                    }
                    full <- parsed$content %||% ""
                    already <- state$emitted
                    # Only emit when the parse strictly extends what was sent;
                    # a partial parse can otherwise rewrite its own tail.
                    if (!(startsWith(full, already) && nchar(full) > nchar(already))) next
                    delta <- substr(full, nchar(already) + 1L, nchar(full))
                    state$emitted <- full
                    return(emit(.openai_chunk(id, model_id, created,
                                              list(content = delta)), s))
                }
                # generation ended: flush any buffered tail, then finish
                tail <- llama_gen_end(st)
                state$finishing <- TRUE
                if (raw_ok) {
                    if (nzchar(tail)) {
                        return(emit(.openai_chunk(id, model_id, created,
                                                  list(content = tail)), s))
                    }
                } else {
                    state$text <- c(state$text, tail)
                    state$parsed <- parse_out(paste0(state$text, collapse = ""))
                    # Whatever content the final parse reveals beyond what was
                    # streamed goes out now (all of it, for deferred formats).
                    full <- state$parsed$content %||% ""
                    if (!state$tool_frozen && startsWith(full, state$emitted) &&
                        nchar(full) > nchar(state$emitted)) {
                        delta <- substr(full, nchar(state$emitted) + 1L, nchar(full))
                        state$emitted <- full
                        return(emit(.openai_chunk(id, model_id, created,
                                                  list(content = delta)), s))
                    }
                }
            }
            if (!state$flushed) {
                state$flushed <- TRUE
                hit_limit <- state$n >= args$max_new_tokens
                if (raw_ok) {
                    finish <- if (hit_limit) "length" else "stop"
                    return(emit(.openai_chunk(id, model_id, created,
                                              structure(list(), names = character()),
                                              finish_reason = finish), s))
                }
                parsed <- state$parsed %||% list(content = "", tool_calls = NULL)
                tcs    <- .openai_tool_calls(parsed)
                finish <- .openai_finish_reason(parsed, hit_limit)
                # Tool calls ride in the delta of the finishing chunk, which is
                # what OpenAI clients that do not stream arguments expect.
                delta  <- if (length(tcs) > 0) list(tool_calls = tcs)
                          else structure(list(), names = character())
                return(emit(.openai_chunk(id, model_id, created, delta,
                                          finish_reason = finish), s))
            }
            # final SSE frame: the OpenAI sentinel, then close
            list(data = "[DONE]", state = s, done = TRUE)
        }

        drogonR::dr_stream_sse(generator, state = state)
    })

    message(sprintf("llamaR OpenAI server: http://%s:%d  (model '%s')",
                    host, as.integer(port), model_id))
    drogonR::dr_serve(app, port = as.integer(port))

    # dr_serve() returns immediately; Drogon runs in background C++ threads
    # and dispatches to R handlers via the later event loop. Block here,
    # pumping the loop, so the server stays up until interrupted (then stop
    # cleanly — otherwise Drogon's destructor aborts at process exit).
    on.exit(drogonR::dr_stop(), add = TRUE)
    tryCatch(
        while (drogonR::dr_running()) {
            later::run_now(timeoutSecs = 1, all = TRUE)
        },
        interrupt = function(e) invisible(NULL)
    )
    invisible(NULL)
}
