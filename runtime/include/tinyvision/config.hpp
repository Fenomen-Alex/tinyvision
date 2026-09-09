// TinyVision runtime configuration (mirrors training/tv/poc.py PocConfig).
#pragma once

#include <cstdint>
#include <string>

namespace tinyvision {

struct TemporalParams {
  double alpha = 0.15;
  int baseline_window = 60;
  int warmup = 30;
  double suspect_threshold = 0.05;
  double fail_threshold = 0.30;
  double recover_threshold = 0.01;
  double fail_threshold_abs = 0.65;
  int confirm = 3;
  bool use_delta = false;

  bool operator==(const TemporalParams& o) const {
    return alpha == o.alpha && baseline_window == o.baseline_window &&
           warmup == o.warmup && suspect_threshold == o.suspect_threshold &&
           fail_threshold == o.fail_threshold &&
           recover_threshold == o.recover_threshold &&
           fail_threshold_abs == o.fail_threshold_abs &&
           confirm == o.confirm && use_delta == o.use_delta;
  }
};

struct RuntimeConfig {
  // Model
  std::string model_path;
  int image_size = 224;
  double threshold = 0.5;  // raw per-frame decision threshold on P(failing)

  // Source: exactly one of input_path / camera_index is used.
  std::string input_path;
  bool use_camera = false;
  int camera_index = 0;

  double sample_fps = 0.0;  // 0 => every frame
  int max_frames = 0;       // 0 => unlimited (sampled frames)

  std::string output_json;  // empty => no JSON report file

  TemporalParams temporal;
};

}  // namespace tinyvision