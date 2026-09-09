// Frame preprocessing for the frozen image classifier.
//
// The Python POC resizes via torchvision Resize, which (observed on the frozen
// videos and verified to 0/255) executes Pillow's BILINEAR resampling on the
// PIL-RGB input. To make the C++ runtime reproduce the POC exactly, the default
// kernel is a faithful port of Pillow's two-pass bilinear resampler
// (src/libImaging/Resample.c), including its fixed-point quantization.
// cv::INTER_AREA / INTER_LINEAR remain available for comparison/measurement.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace tinyvision {

enum class ResizeMode : int {
  kPilBilinear = 0,  // default: exact Pillow BILINEAR (matches the POC)
  kArea = 1,         // cv::INTER_AREA (measurement/comparison only)
  kLinear = 2,       // cv::INTER_LINEAR (measurement/comparison only)
};

// Convert BGR -> image_size x image_size RGB (uint8).
void ToRgbResized(const cv::Mat& bgr, cv::Mat& rgb_out, int image_size,
                  ResizeMode mode = ResizeMode::kPilBilinear);

// Faithful port of Pillow's two-pass BILINEAR resample. rgb8 is a CV_8UC3 RGB
// image; out is written CV_8UC3 and matches Image.resize((out_w,out_h),
// Image.BILINEAR) bit-for-bit for the 8-bit path.
void PilBilinearResize(const cv::Mat& rgb8, int out_w, int out_h, cv::Mat& out);

// Preprocess a BGR frame into the ONNX input tensor:
//   out is resized to image_size*image_size*3, populated as CHW float,
//   pixel = (rgb[f32] / 255 - mean[c]) / std[c]  (ImageNet stats), N=1.
// Throws on empty input.
void PreprocessFrame(const cv::Mat& bgr, int image_size,
                     std::vector<float>& out,
                     ResizeMode mode = ResizeMode::kPilBilinear);

}  // namespace tinyvision