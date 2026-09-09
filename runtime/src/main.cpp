// TinyVision edge runtime CLI.
//
//   tinyvision --input video.mp4 [--model model.onnx] [options]
//   tinyvision --camera 0        [--model model.onnx] [options]
//
// Reproduces the behavioral semantics of the Python POC (`tv poc`):
// camera -> preprocess -> ONNX model -> P(failing) -> EWMA+FSM ->
// failing event. Prints per-frame FSM events on stdout and, with
// --output-json, writes a full report (schema mirroring the POC).
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <algorithm>
#include <opencv2/videoio.hpp>

#include "tinyvision/cli.hpp"
#include "tinyvision/config.hpp"
#include "tinyvision/model.hpp"
#include "tinyvision/preprocess.hpp"
#include "tinyvision/temporal.hpp"

namespace tv = tinyvision;

namespace {

// Graceful shutdown on SIGINT/SIGTERM: stop sampling, write the final report,
// exit 0. Kept trivial and async-signal-safe.
volatile std::sig_atomic_t g_stop = 0;
void HandleSignal(int) { g_stop = 1; }

const char* kUsage =
    "tinyvision --input FILE.mp4 | --camera N [options]\n"
    "  -m, --model PATH          ONNX model (default: baseline artifacts/model.onnx)\n"
    "      --image-size N        resize target (default 224, must match model)\n"
    "      --sample-fps F        sampling cadence (default: every frame)\n"
    "      --max-frames N        stop after N sampled frames\n"
    "      --threshold P         raw binary decision threshold (default 0.5)\n"
    "      --output-json PATH    write JSON report\n"
    "      --alpha       --baseline-window   --warmup\n"
    "      --suspect-threshold   --fail-threshold  --recover-threshold\n"
    "      --fail-threshold-abs  --confirm   --use-delta\n";

}  // namespace

void PrintUsage() { std::cout << kUsage; }

int main(int argc, char** argv) {
  tv::Cli cli;
  try {
    cli = tv::ParseArgs(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    PrintUsage();
    return 2;
  }
  if (!cli.help.empty()) {
    PrintUsage();
    return 0;
  }

  tv::RuntimeConfig& cfg = cli.cfg;

  // --- Model load (fail clearly, never silently fall back) ---
  tv::OrtModel model;
  try {
    model.Load(cfg.model_path);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (cfg.image_size != model.Height() || cfg.image_size != model.Width()) {
    std::cerr << "error: configured image_size (" << cfg.image_size
              << ") does not match model input " << model.Width() << "x"
              << model.Height() << "\n";
    return 1;
  }

  // --- Source (fail clearly on missing file / unavailable camera) ---
  cv::VideoCapture cap;
  const bool from_file = !cfg.input_path.empty();
  double native_fps = 0.0;
  int total_frames = 0;
  int width = 0, height = 0;
  if (from_file) {
    cap.open(cfg.input_path);
    if (!cap.isOpened()) {
      std::cerr << "error: cannot open video file: " << cfg.input_path << "\n";
      return 1;
    }
  } else {
    cap.open(cfg.camera_index);
    if (!cap.isOpened()) {
      std::cerr << "error: cannot open camera device " << cfg.camera_index
                << "\n";
      return 1;
    }
  }
  native_fps = cap.get(cv::CAP_PROP_FPS);
  total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  int skip = 1;
  if (cfg.sample_fps > 0.0 && native_fps > 0.0) {
    skip = static_cast<int>(std::lround(native_fps / cfg.sample_fps));
    if (skip < 1) skip = 1;
  }

  tv::DecisionTracker tracker(cfg.temporal, cfg.threshold);
  std::vector<double> decode_ms, preprocess_ms, infer_ms;
  auto t0 = std::chrono::steady_clock::now();

  // Graceful shutdown: SIGINT/SIGTERM stops sampling and finalises the report.
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // --- Frame loop ---
  cv::Mat frame;
  int frame_idx = 0;
  for (;;) {
    if (g_stop) break;
    if (cfg.max_frames > 0 && static_cast<int>(tracker.rows().size()) >= cfg.max_frames) {
      break;
    }
    auto td0 = std::chrono::steady_clock::now();
    // Sampling mirrors POC `_next_frame`: discard `skip` frames, then read one
    // more as the sampled frame (POC keeps the (skip+1)-th read).
    bool got = false;
    for (int i = 0; i < skip; ++i) {
      got = cap.read(frame);
      ++frame_idx;
      if (!got) break;
    }
    if (got) {
      got = cap.read(frame);
      ++frame_idx;
    }
    if (!got || frame.empty()) {
      if (from_file) {
        break;  // end of stream
      }
      // Camera frame malformed/missing: fail clearly rather than hang.
      std::cerr << "error: camera produced no frame at sample index "
                << tracker.rows().size() << "\n";
      return 1;
    }
    auto td1 = std::chrono::steady_clock::now();
    decode_ms.push_back(0.001 * std::chrono::duration_cast<std::chrono::microseconds>(td1 - td0).count());

    std::vector<float> tensor;
    auto tp0 = std::chrono::steady_clock::now();
    tv::PreprocessFrame(frame, cfg.image_size, tensor);
    auto tp1 = std::chrono::steady_clock::now();
    preprocess_ms.push_back(0.001 * std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count());

    auto ti0 = std::chrono::steady_clock::now();
    const double fail_score = model.PredictFailing(tensor);
    auto ti1 = std::chrono::steady_clock::now();
    infer_ms.push_back(0.001 * std::chrono::duration_cast<std::chrono::microseconds>(ti1 - ti0).count());

    const tv::FrameDecision d = tracker.Step(fail_score);
    tracker.SetFrameIdx(d.frame, frame_idx);
    if (!d.event.empty()) {
      std::printf("frame=%d raw=%.4f signal=%.4f state=%s event=%s\n", frame_idx,
                  d.raw, d.signal, d.state.c_str(), d.event.c_str());
    }
    if (d.frame % std::max(25, skip * 25) == 0) {
      const tv::FrameDecision& last = tracker.rows().back();
      std::printf("frame=%d P(failing)=%.4f state=%s raw=%s\n", frame_idx,
                  last.raw, last.state.c_str(),
                  last.raw_failing ? "failing" : "normal");
    }
  }
  cap.release();

  auto t1 = std::chrono::steady_clock::now();
  const double stream_sec = std::chrono::duration<double>(t1 - t0).count();

  // --- Summary ---
  const std::vector<tv::FrameDecision>& rows = tracker.rows();
  int first_event_frame = -1;
  int failing_frames = 0;
  int obs = 0, norm = 0, susp = 0, fail = 0, paused = 0;
  std::vector<int> event_frames;
  for (const tv::FrameDecision& r : rows) {
    if (r.state == "OBSERVING") ++obs;
    if (r.state == "NORMAL") ++norm;
    if (r.state == "SUSPECT") ++susp;
    if (r.state == "FAILING") { ++fail; ++failing_frames; }
    if (r.state == "PAUSED") ++paused;
    if (!r.event.empty()) {
      if (first_event_frame < 0) first_event_frame = r.frame;
      event_frames.push_back(r.frame);
    }
  }

  auto ms_stats = [](std::vector<double>& v) -> std::string {
    if (v.empty()) return "{\"count\": 0}";
    std::sort(v.begin(), v.end());
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    const size_t i95 = static_cast<size_t>(v.size() * 0.95);
    const size_t p95 = i95 >= v.size() ? v.size() - 1 : i95;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"count\": %zu, \"mean_ms\": %.4f, \"p50_ms\": %.4f, \"p95_ms\": %.4f, \"max_ms\": %.4f}",
                  v.size(), mean, v[v.size() / 2], v[p95], v.back());
    return buf;
  };

  std::cout
      << "frames processed: " << rows.size() << " (of "
      << (from_file ? std::to_string(total_frames) : std::string("live")) << ")\n"
      << "states: OBSERVING=" << obs << " NORMAL=" << norm << " SUSPECT=" << susp
      << " FAILING=" << fail << " PAUSED=" << paused << "\n"
      << "events: " << (event_frames.empty() ? std::string("none")
                                             : std::to_string(event_frames.size()))
      << "\n";
  if (g_stop) {
    std::cout << "interrupted: true (final report written)\n";
  }
  if (first_event_frame >= 0) {
    std::cout << "first failing event at sample frame " << first_event_frame << "\n";
  }
  if (!rows.empty()) {
    std::cout << "cadence: " << (rows.size() / stream_sec) << " processed fps over "
              << stream_sec << " s\n";
    std::cout << "decode: " << ms_stats(decode_ms) << "\n"
              << "preprocess: " << ms_stats(preprocess_ms) << "\n"
              << "inference: " << ms_stats(infer_ms) << "\n";
  }

  // --- Optional JSON report ---
  if (!cfg.output_json.empty()) {
    FILE* f = std::fopen(cfg.output_json.c_str(), "w");
    if (!f) {
      std::cerr << "error: cannot write " << cfg.output_json << "\n";
      return 1;
    }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"schema\": \"poc-inference\",\n");
    std::fprintf(f, "  \"schema_version\": 1,\n");
    std::fprintf(f, "  \"runtime\": \"cpp-onnxruntime\",\n");
    std::fprintf(f, "  \"model\": {\"path\": \"%s\", \"image_size\": %d, \"threshold\": %.4f, \"backend\": \"onnxruntime-cpu\"},\n",
                 cfg.model_path.c_str(), cfg.image_size, cfg.threshold);
    std::fprintf(f, "  \"source\": {\"input\": \"%s\", \"resolution\": \"%dx%d\", \"native_fps\": %.3f, \"total_frames_in_video\": %d, \"sample_fps\": %.3f, \"skip\": %d},\n",
                 from_file ? cfg.input_path.c_str() : ("camera/" + std::to_string(cfg.camera_index)).c_str(),
                 width, height, native_fps, from_file ? total_frames : 0, cfg.sample_fps, skip);
    const tv::TemporalParams& t = cfg.temporal;
    std::fprintf(f, "  \"temporal\": {\"alpha\": %.6f, \"baseline_window\": %d, \"warmup\": %d, \"suspect_threshold\": %.6f, \"fail_threshold\": %.6f, \"recover_threshold\": %.6f, \"fail_threshold_abs\": %.6f, \"confirm\": %d, \"use_delta\": %s},\n",
                 t.alpha, t.baseline_window, t.warmup, t.suspect_threshold,
                 t.fail_threshold, t.recover_threshold, t.fail_threshold_abs,
                 t.confirm, t.use_delta ? "true" : "false");
    std::fprintf(f, "  \"summary\": {\"frames\": %zu, \"interrupted\": %s, \"state_counts\": {\"OBSERVING\": %d, \"NORMAL\": %d, \"SUSPECT\": %d, \"FAILING\": %d, \"PAUSED\": %d}, \"first_failing_event_frame\": %d, \"failing_frames\": %d, \"last_state\": \"%s\", \"cadence_fps\": %.3f},\n",
                 rows.size(), g_stop ? "true" : "false",
                 obs, norm, susp, fail, paused, first_event_frame,
                 failing_frames,
                 rows.empty() ? "NONE" : rows.back().state.c_str(),
                 rows.size() / stream_sec);

    auto ms = [&](std::vector<double>& v) { return ms_stats(v); };
    std::fprintf(f, "  \"timing\": {\"processed_frames\": %zu, \"stream_seconds\": %.4f, \"cadence_fps\": %.3f, \"decode_ms\": %s, \"preprocess_ms\": %s, \"inference_ms\": %s},\n",
                 rows.size(), stream_sec, rows.size() / stream_sec,
                 ms(decode_ms).c_str(), ms(preprocess_ms).c_str(),
                 ms(infer_ms).c_str());
    std::fprintf(f, "  \"annotated_output\": null,\n");
    std::fprintf(f, "  \"per_frame\": [\n");
    for (size_t i = 0; i < rows.size(); ++i) {
      const tv::FrameDecision& r = rows[i];
      std::fprintf(f, "    {\"frame\": %d, \"frame_idx\": %d, \"raw\": %.6f, \"signal\": %.6f, \"baseline\": %.6f, \"delta\": %.6f, \"state\": \"%s\", \"raw_decision\": \"%s\", \"event\": %s}%s\n",
                   r.frame, r.raw_frame, r.raw, r.signal, r.baseline, r.delta,
                   r.state.c_str(), r.raw_failing ? "failing" : "normal",
                   r.event.empty() ? "null" : ("\"" + r.event + "\"").c_str(),
                   (i + 1 == rows.size()) ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::cout << "JSON report written to " << cfg.output_json << "\n";
  }

  return 0;
}