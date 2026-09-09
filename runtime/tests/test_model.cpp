#include <algorithm>
#include <cstdlib>

#include "harness.hpp"
#include "tinyvision/model.hpp"

namespace {

std::string ModelPath() {
  const char* env = std::getenv("TINYVISION_MODEL_ONNX");
  if (env && *env) return env;
  return "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx";
}
std::string GoldenInput() { return "runtime/tests/golden_model_input.npy"; }
std::string GoldenLogits() { return "runtime/tests/golden_model_logits.npy"; }

}  // namespace

// The C++ ONNX session must reproduce the Python onnxruntime logits for the
// same fixed input tensor (identical weights, deterministic CPU graph).
TEST(ModelMatchesPythonOnnxRuntime) {
  tinyvision::OrtModel model;
  model.Load(ModelPath());
  CHECK(model.Loaded());
  CHECK(model.Height() == 224 && model.Width() == 224);

  const std::vector<float> input = tv_testing::LoadNpyF32(GoldenInput());
  CHECK(input.size() == static_cast<size_t>(3 * 224 * 224));
  const std::vector<float> golden = tv_testing::LoadNpyF32(GoldenLogits());
  CHECK(golden.size() == 2);

  const std::vector<float> logits = model.RunLogits(input);
  CHECK(logits.size() == 2);
  if (!tv_testing::Close(logits[0], golden[0], 1e-5) ||
      !tv_testing::Close(logits[1], golden[1], 1e-5)) {
    std::fprintf(stderr, "logit mismatch: got [%g, %g] want [%g, %g]\n",
                 logits[0], logits[1], golden[0], golden[1]);
    std::exit(1);
  }

  // P(class=failing) via fail_idx=0 softmax.
  const double p = model.PredictFailing(input);
  const double e0 = std::exp(golden[0] - std::max(golden[0], golden[1]));
  const double e1 = std::exp(golden[1] - std::max(golden[0], golden[1]));
  const double expected = e0 / (e0 + e1);
  if (!tv_testing::Close(p, expected, 1e-5)) {
    std::fprintf(stderr, "P(failing) mismatch: got %g want %g\n", p, expected);
    std::exit(1);
  }
  CHECK(p > 0.5);  // synthetic input is failing-dominant by construction
}

TEST(ModelLoadFailsClearlyOnMissingFile) {
  tinyvision::OrtModel model;
  bool threw = false;
  try {
    model.Load("runtime/tests/does_not_exist.onnx");
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(!model.Loaded());
}

TEST(ModelRejectsBadInputShape) {
  tinyvision::OrtModel model;
  model.Load(ModelPath());
  bool threw = false;
  try {
    model.RunLogits(std::vector<float>(4 * 224 * 224, 0.0f));
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(ModelRunningBeforeLoadThrows) {
  tinyvision::OrtModel model;
  bool threw = false;
  try {
    model.RunLogits(std::vector<float>(3 * 224 * 224, 0.0f));
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}