// Minimal test harness: micro CHECK framework + helpers for loading the
// Python-generated golden files (npy tensor dumps and a CSV decision trace).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace tv_testing {

using TestFn = std::function<void()>;
std::vector<std::pair<std::string, TestFn>>& Registry();

#define TEST(name)                                                     \
  static void TestBody_##name();                                       \
  namespace {                                                          \
  struct Register_##name {                                             \
    Register_##name() {                                                \
      tv_testing::Registry().emplace_back(#name, TestBody_##name);     \
    }                                                                  \
  } register_##name;                                                   \
  }                                                                    \
  static void TestBody_##name()

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__,    \
                   __LINE__, #cond);                                   \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

inline bool Close(double a, double b, double tol) {
  return std::fabs(a - b) <= tol;
}

// Load a float32 NumPy .npy (fortran_order=False). Returns flattened data.
std::vector<float> LoadNpyF32(const std::string& path);
// Load a uint8 NumPy .npy.
std::vector<uint8_t> LoadNpyU8(const std::string& path);
// Basic 2-D CSV data loaded as fully-typed text cells.
std::vector<std::vector<std::string>> LoadCsv(const std::string& path);

int RunAll();
}  // namespace tv_testing