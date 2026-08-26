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
