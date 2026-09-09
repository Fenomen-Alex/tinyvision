#include <cstdlib>
#include <stdexcept>

#include "harness.hpp"
#include "tinyvision/cli.hpp"

namespace {

using tinyvision::ParseArgs;

// argv-like helper: builds {argv[0], args...} and returns argc.
struct Argv {
  std::vector<std::string> store;
  std::vector<char*> ptrs;
  int argc = 0;
  explicit Argv(std::vector<std::string> args) : store(std::move(args)) {
    ptrs.reserve(store.size());
    for (auto& s : store) ptrs.push_back(s.data());
    argc = static_cast<int>(ptrs.size());
  }
};

tinyvision::Cli Parse(std::vector<std::string> args) {
  Argv a(std::move(args));
  return ParseArgs(a.argc, a.ptrs.data());
}

bool Throws(std::vector<std::string> args) {
  try {
    Parse(std::move(args));
    return false;
  } catch (const std::exception&) {
    return true;
  }
}

}  // namespace

TEST(CliDefaultsMatchValidatedConfig) {
  const tinyvision::RuntimeConfig& d = Parse({"tinyvision", "--help"}).cfg;
  CHECK(d.image_size == 224);
  CHECK(d.threshold == 0.5);
  CHECK(d.sample_fps == 0.0);  // 0 = every frame
  CHECK(d.temporal.alpha == 0.15);
  CHECK(d.temporal.baseline_window == 60);
  CHECK(d.temporal.warmup == 30);
  CHECK(d.temporal.suspect_threshold == 0.05);
  CHECK(d.temporal.fail_threshold == 0.30);
  CHECK(d.temporal.recover_threshold == 0.01);
  CHECK(d.temporal.fail_threshold_abs == 0.65);
  CHECK(d.temporal.confirm == 3);
  CHECK(!d.temporal.use_delta);
}

TEST(CliHelpFlag) {
  const tinyvision::Cli cli = Parse({"tinyvision", "--help"});
  CHECK(!cli.help.empty());
  const tinyvision::Cli cli2 = Parse({"tinyvision", "-h"});
  CHECK(!cli2.help.empty());
  CHECK(Parse({"tinyvision", "--help", "--input", "x.mp4"}).help.empty() == false);
}

TEST(CliParsesStandardOptions) {
  const tinyvision::Cli cli = Parse({
      "tinyvision", "--input", "in.mp4", "--model", "m.onnx",
      "--image-size", "256", "--sample-fps", "1", "--max-frames", "50",
      "--threshold", "0.4", "--output-json", "out.json",
      "--alpha", "0.2", "--baseline-window", "90", "--warmup", "45",
      "--suspect-threshold", "0.1", "--fail-threshold", "0.4",
      "--recover-threshold", "0.02", "--fail-threshold-abs", "0.7",
      "--confirm", "5", "--use-delta",
  });
  const tinyvision::RuntimeConfig& c = cli.cfg;
  CHECK(c.input_path == "in.mp4");
  CHECK(c.model_path == "m.onnx");
  CHECK(c.image_size == 256);
  CHECK(c.sample_fps == 1.0);
  CHECK(c.max_frames == 50);
  CHECK(c.threshold == 0.4);
  CHECK(c.output_json == "out.json");
  CHECK(c.temporal.alpha == 0.2);
  CHECK(c.temporal.baseline_window == 90);
  CHECK(c.temporal.warmup == 45);
  CHECK(c.temporal.suspect_threshold == 0.1);
  CHECK(c.temporal.fail_threshold == 0.4);
  CHECK(c.temporal.recover_threshold == 0.02);
  CHECK(c.temporal.fail_threshold_abs == 0.7);
  CHECK(c.temporal.confirm == 5);
  CHECK(c.temporal.use_delta);
}

TEST(CliDefaultModelPathIsBaselineArtifact) {
  // No --model given must fall back to the production artifact path.
  const tinyvision::Cli cli = Parse({"tinyvision", "--input", "in.mp4"});
  CHECK(cli.cfg.model_path ==
        "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx");
}

TEST(CliRequiresExactlyOneSource) {
  CHECK(Throws({"tinyvision"}));                            // neither
  CHECK(Throws({"tinyvision", "--input", "a.mp4", "--camera", "0"}));  // both
  CHECK(Throws({"tinyvision", "--input", "a.mp4", "--camera", "1"}));
  CHECK(!Throws({"tinyvision", "--input", "a.mp4"}));
  CHECK(!Throws({"tinyvision", "--camera", "0"}));
}

TEST(CliRejectsUnknownOption) {
  CHECK(Throws({"tinyvision", "--bogus", "--input", "a.mp4"}));
  CHECK(Throws({"tinyvision", "-x"}));
}

TEST(CliRejectsMissingValue) {
  CHECK(Throws({"tinyvision", "--input"}));
  CHECK(Throws({"tinyvision", "--model", "--input", "a.mp4"}));
  CHECK(Throws({"tinyvision", "--input", "a.mp4", "--sample-fps"}));
}

TEST(CliRejectsOutOfRangeValues) {
  CHECK(Throws({"tinyvision", "--input", "a", "--threshold", "-0.1"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--threshold", "1.5"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--sample-fps", "-1"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--max-frames", "-1"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--image-size", "0"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--warmup", "-5"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--confirm", "0"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--baseline-window", "0"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--alpha", "0"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--alpha", "1.5"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--camera", "-2"}));
  CHECK(!Throws({"tinyvision", "--input", "a", "--threshold", "0.5"}));
  CHECK(!Throws({"tinyvision", "--input", "a", "--confirm", "1"}));
}

TEST(CliRejectsEmptyModelOrEmptyInputString) {
  CHECK(Throws({"tinyvision", "--input", "", "--model", "m"}));
  CHECK(Throws({"tinyvision", "--input", "a.mp4", "--model", ""}));
}

TEST(CliNumericScanIsStrict) {
  // Non-numeric values in numeric options must be rejected, not silently parsed.
  CHECK(Throws({"tinyvision", "--input", "a", "--sample-fps", "abc"}));
  CHECK(Throws({"tinyvision", "--input", "a", "--image-size", "1e9"}));
}