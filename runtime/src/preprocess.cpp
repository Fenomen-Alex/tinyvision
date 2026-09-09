// Frame preprocessing / Pillow BILINEAR resize port.
//
// The resize implementation below is a line-by-line port of Pillow's
// ImagingResample (src/libImaging/Resample.c, HPND license) restricted to the
// 8-bit RGB downscale/upscale case used by the POC. It reproduces Pillow's
// fixed-point arithmetic (PRECISION_BITS = 32 - 8 - 2 = 22) so that output
// pixels are bit-identical to Image.resize(..., Image.BILINEAR).
#include "tinyvision/preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace tinyvision {

namespace {

constexpr double kMean[3] = {0.485, 0.456, 0.406};
constexpr double kStd[3] = {0.229, 0.224, 0.225};

inline double BilinearFilter(double x) {
  if (x < 0.0) x = -x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

struct Coeffs {
  int ksize = 0;
  std::vector<int> bounds_min, bounds_max;
  std::vector<int32_t> kk;  // quantized coefficients, Pillow layout
};

// Port of Pillow precompute_coeffs + normalize_coeffs_8bpc for BILINEAR.
Coeffs PilPrecompute(int in_size, int out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = 1.0 * filterscale;  // bilinear filter support=1
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  const double inv_filterscale = 1.0 / filterscale;

  Coeffs c;
  c.ksize = ksize;
  c.bounds_min.resize(out_size);
  c.bounds_max.resize(out_size);
  c.kk.resize(static_cast<size_t>(out_size) * ksize);

  // Keep raw (unnormalized) weights, then normalise, exactly like Pillow.
  std::vector<double> raw(static_cast<size_t>(out_size) * ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = 0.0 + (xx + 0.5) * scale;
    double ww = 0.0;
    // Pillow: truncating casts of (center +/- support + 0.5), then clamp.
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;

    double* k = &raw[static_cast<size_t>(xx) * ksize];
    for (int x = 0; x < xmax; ++x) {
      const double w =
          BilinearFilter((x + xmin - center + 0.5) * inv_filterscale);
      k[x] = w;
      ww += w;
    }
    if (ww != 0.0) {
      for (int x = 0; x < xmax; ++x) {
        k[x] = k[x] / ww;
      }
    }
    for (int x = xmax; x < ksize; ++x) k[x] = 0;
    c.bounds_min[xx] = xmin;
    c.bounds_max[xx] = xmax;
  }
  // normalize_coeffs_8bpc: integer quantize to 22 fractional bits.
  for (size_t i = 0; i < raw.size(); ++i) {
    const double v = raw[i];
    c.kk[i] = static_cast<int32_t>(v < 0.0 ? -0.5 + v * (1 << 22)
                                          : 0.5 + v * (1 << 22));
  }
  return c;
}

inline int Clip8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

}  // namespace

void PilBilinearResize(const cv::Mat& rgb8, int out_w, int out_h, cv::Mat& out) {
  if (rgb8.type() != CV_8UC3) {
    throw std::runtime_error("PilBilinearResize expects CV_8UC3 RGB input");
  }
  const int in_w = rgb8.cols;
  const int in_h = rgb8.rows;
  constexpr int kPrecisionBits = 22;

  const Coeffs cx = PilPrecompute(in_w, out_w);
  const Coeffs cy = PilPrecompute(in_h, out_h);

  // Horizontal pass: temp is out_w wide, in_h tall (RGB, 3 channels).
  std::vector<int32_t> temp(static_cast<size_t>(in_h) * out_w * 3);
  for (int yy = 0; yy < in_h; ++yy) {
    const uint8_t* line_in = rgb8.ptr<uint8_t>(yy);
    int32_t* line_out = &temp[static_cast<size_t>(yy) * out_w * 3];
    for (int xx = 0; xx < out_w; ++xx) {
      const int xmin = cx.bounds_min[xx];
      const int xmax = cx.bounds_max[xx];
      const int32_t* k = &cx.kk[static_cast<size_t>(xx) * cx.ksize];
      int32_t ss0 = 1 << (kPrecisionBits - 1);
      int32_t ss1 = 1 << (kPrecisionBits - 1);
      int32_t ss2 = 1 << (kPrecisionBits - 1);
      for (int x = 0; x < xmax; ++x) {
        const uint8_t* p = line_in + (x + xmin) * 3;
        ss0 += p[0] * k[x];
        ss1 += p[1] * k[x];
        ss2 += p[2] * k[x];
      }
      line_out[xx * 3 + 0] = Clip8(static_cast<int>(ss0 >> kPrecisionBits));
      line_out[xx * 3 + 1] = Clip8(static_cast<int>(ss1 >> kPrecisionBits));
      line_out[xx * 3 + 2] = Clip8(static_cast<int>(ss2 >> kPrecisionBits));
    }
  }

  // Vertical pass -> out.
  out.create(out_h, out_w, CV_8UC3);
  for (int yy = 0; yy < out_h; ++yy) {
    const int ymin = cy.bounds_min[yy];
    const int ymax = cy.bounds_max[yy];
    const int32_t* k = &cy.kk[static_cast<size_t>(yy) * cy.ksize];
    uint8_t* line_out = out.ptr<uint8_t>(yy);
    for (int xx = 0; xx < out_w; ++xx) {
      int32_t ss0 = 1 << (kPrecisionBits - 1);
      int32_t ss1 = 1 << (kPrecisionBits - 1);
      int32_t ss2 = 1 << (kPrecisionBits - 1);
      for (int y = 0; y < ymax; ++y) {
        const int32_t* p =
            &temp[static_cast<size_t>(y + ymin) * out_w * 3 + xx * 3];
        ss0 += p[0] * k[y];
        ss1 += p[1] * k[y];
        ss2 += p[2] * k[y];
      }
      line_out[xx * 3 + 0] = Clip8(static_cast<int>(ss0 >> kPrecisionBits));
      line_out[xx * 3 + 1] = Clip8(static_cast<int>(ss1 >> kPrecisionBits));
      line_out[xx * 3 + 2] = Clip8(static_cast<int>(ss2 >> kPrecisionBits));
    }
  }
}

void ToRgbResized(const cv::Mat& bgr, cv::Mat& rgb_out, int image_size,
                  ResizeMode mode) {
  if (bgr.empty()) {
    throw std::runtime_error("preprocess: empty input frame");
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  switch (mode) {
    case ResizeMode::kPilBilinear:
      PilBilinearResize(rgb, image_size, image_size, rgb_out);
      break;
    case ResizeMode::kArea:
      cv::resize(rgb, rgb_out, cv::Size(image_size, image_size), 0, 0,
                 cv::INTER_AREA);
      break;
    case ResizeMode::kLinear:
      cv::resize(rgb, rgb_out, cv::Size(image_size, image_size), 0, 0,
                 cv::INTER_LINEAR);
      break;
  }
}

void PreprocessFrame(const cv::Mat& bgr, int image_size,
                     std::vector<float>& out, ResizeMode mode) {
  cv::Mat rgb;
  ToRgbResized(bgr, rgb, image_size, mode);

  const int h = image_size;
  const int w = image_size;
  out.resize(static_cast<size_t>(3) * h * w);

  for (int c = 0; c < 3; ++c) {
    float* dst = out.data() + static_cast<size_t>(c) * h * w;
    for (int y = 0; y < h; ++y) {
      const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < w; ++x) {
        const double p = static_cast<double>(row[x][c]);
        dst[static_cast<size_t>(y) * w + x] =
            static_cast<float>((p / 255.0 - kMean[c]) / kStd[c]);
      }
    }
  }
}

}  // namespace tinyvision