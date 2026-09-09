#include "tinyvision/cli.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyvision {

namespace {

// Throw if the option named `arg` has no following value.
const std::string& Value(const std::vector<std::string>& a, size_t i,
                         const std::string& arg) {
  if (i + 1 >= a.size()) {
    throw std::runtime_error("missing value for " + arg);
  }
  return a[i + 1];
}

double DoubleAfter(const std::vector<std::string>& a, size_t& i,
                   const std::string& arg) {
  const std::string& v = Value(a, i, arg);
  char* end = nullptr;
  const double d = std::strtod(v.c_str(), &end);
  if (end == v.c_str()) {
    throw std::runtime_error("invalid number for " + arg + ": '" + v + "'");
  }
  ++i;
  return d;
}

int IntAfter(const std::vector<std::string>& a, size_t& i,
             const std::string& arg) {
  const std::string& v = Value(a, i, arg);
  size_t consumed = 0;
  try {
    const int n = std::stoi(v, &consumed);
    if (consumed != v.size()) {
      throw std::runtime_error("invalid integer for " + arg + ": '" + v + "'");
    }
    ++i;
    return n;
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("invalid integer for " + arg + ": '" + v + "'");
  } catch (const std::out_of_range&) {
    throw std::runtime_error("value out of range for " + arg + ": '" + v + "'");
  }
}

std::string StrAfter(const std::vector<std::string>& a, size_t& i,
                     const std::string& arg) {
  const std::string& v = Value(a, i, arg);
  ++i;
  return v;
}

}  // namespace

Cli ParseArgs(int argc, char** argv) {
  Cli cli;
  std::vector<std::string> a(argv + 1, argv + argc);
  TemporalParams& t = cli.cfg.temporal;
  cli.cfg.model_path =
      "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx";
  for (size_t i = 0; i < a.size(); ++i) {
    const std::string& arg = a[i];
    if (arg == "--help" || arg == "-h") {
      cli.help = "usage";
    } else if (arg == "--input" || arg == "-i") {
      cli.cfg.input_path = StrAfter(a, i, arg);
      cli.used_input = true;
    } else if (arg == "--camera") {
      cli.cfg.camera_index = IntAfter(a, i, arg);
      cli.cfg.use_camera = true;
      cli.used_camera = true;
    } else if (arg == "--model" || arg == "-m") {
      cli.cfg.model_path = StrAfter(a, i, arg);
    } else if (arg == "--image-size") {
      cli.cfg.image_size = IntAfter(a, i, arg);
    } else if (arg == "--sample-fps") {
      cli.cfg.sample_fps = DoubleAfter(a, i, arg);
    } else if (arg == "--max-frames") {
      cli.cfg.max_frames = IntAfter(a, i, arg);
    } else if (arg == "--threshold") {
      cli.cfg.threshold = DoubleAfter(a, i, arg);
    } else if (arg == "--output-json") {
      cli.cfg.output_json = StrAfter(a, i, arg);
    } else if (arg == "--alpha") {
      t.alpha = DoubleAfter(a, i, arg);
    } else if (arg == "--baseline-window") {
      t.baseline_window = IntAfter(a, i, arg);
    } else if (arg == "--warmup") {
      t.warmup = IntAfter(a, i, arg);
    } else if (arg == "--suspect-threshold") {
      t.suspect_threshold = DoubleAfter(a, i, arg);
    } else if (arg == "--fail-threshold") {
      t.fail_threshold = DoubleAfter(a, i, arg);
    } else if (arg == "--recover-threshold") {
      t.recover_threshold = DoubleAfter(a, i, arg);
    } else if (arg == "--fail-threshold-abs") {
      t.fail_threshold_abs = DoubleAfter(a, i, arg);
    } else if (arg == "--confirm") {
      t.confirm = IntAfter(a, i, arg);
    } else if (arg == "--use-delta") {
      t.use_delta = true;
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (!cli.help.empty()) {
    return cli;
  }

  // --- Deployment config validation (no behavioral tuning) ---
  if (cli.used_input && cli.used_camera) {
    throw std::runtime_error("--input and --camera are mutually exclusive");
  }
  if (!cli.used_input && !cli.used_camera) {
    throw std::runtime_error("provide either --input FILE or --camera N");
  }
  if (cli.used_input && cli.cfg.input_path.empty()) {
    throw std::runtime_error("--input path must not be empty");
  }
  if (cli.cfg.model_path.empty()) {
    throw std::runtime_error("--model path must not be empty");
  }
  if (cli.cfg.image_size <= 0) {
    throw std::runtime_error("--image-size must be a positive integer");
  }
  if (cli.cfg.camera_index < 0) {
    throw std::runtime_error("--camera index must be >= 0");
  }
  if (cli.cfg.sample_fps < 0.0) {
    throw std::runtime_error("--sample-fps must be >= 0 (0 = every frame)");
  }
  if (cli.cfg.max_frames < 0) {
    throw std::runtime_error("--max-frames must be >= 0 (0 = unlimited)");
  }
  if (cli.cfg.threshold < 0.0 || cli.cfg.threshold > 1.0 ||
      !std::isfinite(cli.cfg.threshold)) {
    throw std::runtime_error("--threshold must be in [0, 1]");
  }
  if (!(t.alpha > 0.0 && t.alpha <= 1.0)) {
    throw std::runtime_error("--alpha must be in (0, 1]");
  }
  if (t.baseline_window < 1) {
    throw std::runtime_error("--baseline-window must be >= 1");
  }
  if (t.warmup < 0) {
    throw std::runtime_error("--warmup must be >= 0");
  }
  if (t.confirm < 1) {
    throw std::runtime_error("--confirm must be >= 1");
  }
  for (const auto& item : {std::pair<const char*, double>{
           "--suspect-threshold", t.suspect_threshold},
           {"--fail-threshold", t.fail_threshold},
           {"--recover-threshold", t.recover_threshold},
           {"--fail-threshold-abs", t.fail_threshold_abs}}) {
    if (!(item.second >= 0.0 && item.second <= 1.0)) {
      throw std::runtime_error(std::string(item.first) + " must be in [0, 1]");
    }
  }
  return cli;
}

}  // namespace tinyvision