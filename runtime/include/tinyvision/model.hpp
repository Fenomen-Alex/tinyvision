// Minimal ONNX Runtime session wrapper for the frozen binary classifier.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace tinyvision {

class OrtModel {
 public:
  OrtModel() = default;
  ~OrtModel() = default;
  OrtModel(const OrtModel&) = delete;
  OrtModel& operator=(const OrtModel&) = delete;

  // Throws std::runtime_error with a clear message on missing/invalid model.
  void Load(const std::string& onnx_path, int threads = 0);

  // Run a preprocessed input (NCHW float, batch 1) and return the 2 logits.
  // Throws std::runtime_error on inference failure.
  std::vector<float> RunLogits(const std::vector<float>& input_nchw) const;

  // P(class=failing) for the binary model (fail_idx=0, matching the frozen
  // split convention failing=0 / normal=1).
  double PredictFailing(const std::vector<float>& input_nchw) const;

  bool Loaded() const { return session_ != nullptr; }
  int Height() const { return static_cast<int>(h_); }
  int Width() const { return static_cast<int>(w_); }

 private:
  std::shared_ptr<Ort::Env> env_;      // must outlive session_ (declared first)
  std::shared_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  int64_t h_ = 0;
  int64_t w_ = 0;
  int64_t num_channels_ = 0;
};

}  // namespace tinyvision