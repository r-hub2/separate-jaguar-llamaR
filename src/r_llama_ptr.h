#pragma once

// Type-checked access to external-pointer arguments coming from R.
//
// R_ExternalPtrAddr() does NOT check the SEXP type: handed a string, a number
// or a list it reads whatever sits at the pointer field's offset and returns
// garbage, which a plain `if (!p)` check happily passes. Dereferencing that
// crashes the R session, so every externalptr argument goes through these.
//
// Finalizers are exempt: R only ever calls them with the pointer it created,
// and they must not raise R errors.

#include <R.h>
#include <Rinternals.h>

// Rinternals.h defines length() as a macro which conflicts with C++ methods
#ifdef length
#undef length
#endif

#include "r_llama_throw.h"  // llamar_error(): throws instead of longjmp-ing

// There are two flavours of every accessor here, because the two ways of
// reporting an error each break something in the other's context:
//
//   *_arg        reports with Rf_error(), i.e. by longjmp. R resets the
//                protection stack itself, so this is what rchk expects to see;
//                but a longjmp skips C++ destructors, so it may only be used
//                where nothing with a destructor is live on the stack.
//
//   *_arg_throw  reports with llamar_error(), i.e. by throwing, so destructors
//                run. Callers MUST sit inside LLAMAR_ENTRYPOINT_BEGIN/END,
//                which converts the exception back into an R error at the
//                extern "C" boundary. Use these in the entry points that hold
//                a std::vector / std::string / RAII guard.
//
// The throwing flavour is deliberately the narrower one: a catch handler
// reachable from a PROTECT reads as a protection stack imbalance to rchk, so
// only the entry points that genuinely need destructor safety pay that cost.
//
// Finalizers use neither: R only ever calls them with the pointer it created,
// and they must not raise R errors at all.

// --- longjmp flavour (default; for entry points with no C++ objects) --------

// Address of an externalptr argument, or NULL for a handle that has already
// been freed. Callers that accept NULL are the idempotent free_* entry points;
// everything else should use llamar_ptr_arg.
static inline void * llamar_ptr_addr_or_null(SEXP x, const char * what) {
    if (TYPEOF(x) != EXTPTRSXP) {
        Rf_error("llamaR: invalid %s pointer", what);
    }
    return R_ExternalPtrAddr(x);
}

// As above, but a freed (NULL) handle is an error too.
static inline void * llamar_ptr_arg(SEXP x, const char * what) {
    void * p = llamar_ptr_addr_or_null(x, what);
    if (!p) Rf_error("llamaR: invalid %s pointer", what);
    return p;
}

// --- throwing flavour (for entry points that own C++ objects) ---------------

static inline void * llamar_ptr_addr_or_null_throw(SEXP x, const char * what) {
    if (TYPEOF(x) != EXTPTRSXP) {
        llamar_error("llamaR: invalid %s pointer", what);
    }
    return R_ExternalPtrAddr(x);
}

static inline void * llamar_ptr_arg_throw(SEXP x, const char * what) {
    void * p = llamar_ptr_addr_or_null_throw(x, what);
    if (!p) llamar_error("llamaR: invalid %s pointer", what);
    return p;
}
