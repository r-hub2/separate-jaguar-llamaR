// Leak-free error reporting across the R/C++ boundary.
//
// Rf_error() unwinds by longjmp, which skips C++ destructors: any std::vector,
// std::string or RAII guard live on the stack at that moment leaks its heap
// allocation. valgrind reports these as "definitely lost" (see the
// llamar_sampler_params trace in the rhub valgrind run).
//
// The fix, in the shape Rcpp uses: inside C++ code raise a C++ exception
// instead, and convert it back to Rf_error() at the extern "C" boundary, by
// which point every destructor between the throw site and the boundary has
// run. llamar_error() below formats like Rf_error() but throws; the
// LLAMAR_ENTRYPOINT macro pair installs the matching catch.
//
// Rule of thumb: use llamar_error() everywhere below the boundary, and wrap
// every extern "C" SEXP entrypoint body in BEGIN/END.

#ifndef R_LLAMA_THROW_H
#define R_LLAMA_THROW_H

#include <cstdarg>
#include <cstdio>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include <R.h>
#include <Rinternals.h>

// An error raised below the boundary. Carries an already-formatted message so
// the boundary only has to hand it to Rf_error("%s", ...).
struct llamar_exception : public std::runtime_error {
    explicit llamar_exception(const std::string & what) : std::runtime_error(what) {}
};

// printf-style, like Rf_error, but throws instead of longjmp-ing. Declared
// noreturn so the compiler keeps treating the call sites as terminal (no
// spurious "may be used uninitialized" warnings after an error branch).
#ifdef __GNUC__
__attribute__((noreturn, format(printf, 1, 2)))
#endif
inline void llamar_error(const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    throw llamar_exception(buf);
}

// Render a caught exception into `buf`. llamar_error() messages are already
// fully formatted and carry their own "llamaR:" prefix, so they pass through
// unchanged; anything else gets prefixed so the origin stays visible.
inline void llamar_copy_what_(char * buf, size_t n, const std::exception & e) {
    if (dynamic_cast<const llamar_exception *>(&e) != nullptr) {
        snprintf(buf, n, "%s", e.what());
    } else if (dynamic_cast<const std::bad_alloc *>(&e) != nullptr) {
        snprintf(buf, n, "llamaR: out of memory");
    } else {
        snprintf(buf, n, "llamaR: %s", e.what());
    }
}

// Hand a finished message to R. Marked noreturn so both the compiler and rchk
// know control stops here: Rf_error() longjmps, which is also what resets the
// protection stack after a throw that jumped over an UNPROTECT.
#ifdef __GNUC__
__attribute__((noreturn))
#endif
inline void llamar_raise_(const char * msg) {
    Rf_error("%s", msg);
    while (1) {}  // Rf_error never returns; silences -Wreturn-type
}

// Boundary guard. Every extern "C" SEXP entrypoint wraps its body:
//
//     extern "C" SEXP r_llama_foo(SEXP x) {
//         LLAMAR_ENTRYPOINT_BEGIN
//         ...
//         return ans;
//         LLAMAR_ENTRYPOINT_END
//     }
//
// Rf_error() is called only here, after unwinding has destroyed every C++
// object between the throw site and this frame.
//
// Two things the handler has to get right:
//
//  1. Rf_error() must NOT be called from inside a catch block. It leaves by
//     longjmp, so the handler never finishes and __cxa_end_catch() never runs,
//     which leaks the exception object itself (valgrind: "possibly lost"
//     inside __cxa_allocate_exception). The catch blocks therefore only copy
//     the message into a local buffer; the block ends, the exception is
//     released, and only then does Rf_error() longjmp out.
//
//  2. The failure path must end in a call that does not return. R resets the
//     protection stack itself when Rf_error() longjmps, so a throw between a
//     PROTECT and its UNPROTECT needs no manual unwinding here -- but only if
//     control cannot fall through to an ordinary `return` afterwards. Routing
//     the failure through a noreturn helper says exactly that, and keeps rchk
//     from reading the boundary as a protection stack imbalance.
#define LLAMAR_ENTRYPOINT_BEGIN                                                \
    char llamar_err_buf_[1024];                                                \
    bool llamar_failed_ = false;                                               \
    try {

#define LLAMAR_ENTRYPOINT_END                                                  \
    } catch (const std::exception & e) {                                       \
        llamar_copy_what_(llamar_err_buf_, sizeof(llamar_err_buf_), e);        \
        llamar_failed_ = true;                                                 \
    } catch (...) {                                                            \
        snprintf(llamar_err_buf_, sizeof(llamar_err_buf_),                     \
                 "llamaR: unknown C++ exception");                             \
        llamar_failed_ = true;                                                 \
    }                                                                          \
    /* outside the handler: the exception object is gone by now */             \
    if (llamar_failed_) llamar_raise_(llamar_err_buf_);                        \
    return R_NilValue;  /* not reached when llamar_failed_ */

#endif // R_LLAMA_THROW_H
