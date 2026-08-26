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
// NB: the catch blocks must not let a C++ exception escape into R's C code,
// and Rf_error() itself must be the last thing they do -- it longjmps out.
#define LLAMAR_ENTRYPOINT_BEGIN try {

#define LLAMAR_ENTRYPOINT_END                                                  \
    } catch (const llamar_exception & e) {                                     \
        Rf_error("%s", e.what());                                              \
    } catch (const std::bad_alloc &) {                                         \
        Rf_error("llamaR: out of memory");                                     \
    } catch (const std::exception & e) {                                       \
        Rf_error("llamaR: %s", e.what());                                      \
    } catch (...) {                                                            \
        Rf_error("llamaR: unknown C++ exception");                             \
    }                                                                          \
    return R_NilValue;  /* not reached: every catch above longjmps */

#endif // R_LLAMA_THROW_H
