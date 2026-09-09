#include "tinyvision/model.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tinyvision {

void OrtModel::Load(const std::string& onnx_path, int threads) {
  auto env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "tinyvision");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (threads > 0) {
    opts.SetIntraOpNumThreads(threads);
  }
  try {
    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "tinyvision");
    session_ =
        std::make_shared<Ort::Session>(*env_, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    auto in_info = session_->GetInputTypeInfo(0);
    const std::vector<int64_t> shape =
        in_info.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4) {
      throw std::runtime_error("model input must be NCHW (got dims=" +
                               std::to_string(shape.size()) + ")");
    }
    h_ = shape[2];
    w_ = shape[3];
    num_channels_ = shape[1];
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("ONNX model load failed: ") + e.what());
  }
  if (!session_) {
    throw std::runtime_error("model load produced no session");
  }
}

std::vector<float> OrtModel::RunLogits(const std::vector<float>& input_nchw) const {
  if (!session_) {
    throw std::runtime_error("model not loaded");
  }
  if (h_ <= 0 || w_ <= 0) {
    throw std::runtime_error("model has invalid input shape");
  }
  const int64_t plane = h_ * w_;
  if (static_cast<int64_t>(input_nchw.size()) != 3 * plane) {
    throw std::runtime_error("input tensor size mismatch");
  }
  auto* sess = session_.get();
  const std::array<int64_t, 4> in_shape = {1, num_channels_, h_, w_};
  const std::array<int64_t, 2> out_shape = {1, 2};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(input_nchw.data()), input_nchw.size(),
      in_shape.data(), in_shape.size());
  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  std::vector<Ort::Value> outputs;
  try {
    outputs = sess->Run(Ort::RunOptions{nullptr}, input_names, &in_tensor, 1,
                        output_names, 1);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("ONNX inference failed: ") + e.what());
  }
  if (outputs.empty() || !outputs[0].IsTensor()) {
    throw std::runtime_error("model returned no tensor output");
  }
  auto* data = outputs[0].GetTensorMutableData<float>();
  const auto& out_shape_info = outputs[0].GetTensorTypeAndShapeInfo();
  const size_t n = static_cast<size_t>(out_shape_info.GetElementCount());
  return std::vector<float>(data, data + n);
}

double OrtModel::PredictFailing(const std::vector<float>& input_nchw) const {
  const std::vector<float> logits = RunLogits(input_nchw);
  if (logits.size() != 2) {
    throw std::runtime_error("binary model expected, got " +
                             std::to_string(logits.size()) + " logits");
  }
  // failing_idx == 0 convention (frozen split class_to_idx: failing=0, normal=1)
  const double z0 = static_cast<double>(logits[0]);
  const double z1 = static_cast<double>(logits[1]);
  const double m = z0 > z1 ? z0 : z1;  // numerically stable softmax
  const double e0 = std::exp(z0 - m);
  const double e1 = std::exp(z1 - m);
  return e0 / (e0 + e1);
}

}  // namespace tinyvision