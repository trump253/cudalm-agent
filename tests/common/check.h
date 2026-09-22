// CUDALM — minimal test assertion helpers (no external test framework).
//
// Usage: include this header; call CHECK(cond), CHECK_EQ(a,b), etc. On
// failure the macro prints a diagnostic and returns 1 from the enclosing
// function (so each test function returns int and `return` on failure).
// This keeps tests dependency-free and readable.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n", #cond,       \
                   __FILE__, __LINE__);                                \
      return 1;                                                        \
    }                                                                  \
  } while (0)

#define CHECK_EQ(a, b)                                                 \
  do {                                                                 \
    auto _va = (a);                                                    \
    auto _vb = (b);                                                    \
    if (!(_va == _vb)) {                                               \
      std::fprintf(stderr, "CHECK_EQ failed: %s == %s  (%s:%d)\n",     \
                   #a, #b, __FILE__, __LINE__);                        \
      return 1;                                                        \
    }                                                                  \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                          \
  do {                                                                 \
    double _va = static_cast<double>(a);                               \
    double _vb = static_cast<double>(b);                               \
    if (std::fabs(_va - _vb) > (tol)) {                                \
      std::fprintf(stderr, "CHECK_NEAR failed: |%s - %s| <= %s  (%s:%d)" \
                   "  [got %.9g vs %.9g]\n",                           \
                   #a, #b, #tol, __FILE__, __LINE__, _va, _vb);        \
      return 1;                                                        \
    }                                                                  \
  } while (0)

// Report a test's outcome.
#define TEST_PASS(name) std::fprintf(stderr, "[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg); return 1; } while (0)
