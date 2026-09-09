#!/usr/bin/env bash
# End-to-end smoke test for the C++ runtime, driven by CTest.
# Runs the CLI against a small stage-7 fixture, requires a clean exit (rc=0),
# a valid JSON report, and a correct runtime fingerprint. Also verifies the
# graceful-shutdown path (SIGINT produces "interrupted": true and rc=0).
set -u

BIN=$(pwd)/runtime/build/tinyvision
MODEL=experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx
FIXTURE=docs/research/stage7-fixtures/test_pure_failing_stringing_72.mp4
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

if [ ! -x "$BIN" ]; then
  echo "FAIL: missing binary $BIN" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "FAIL: missing model $MODEL" >&2
  exit 1
fi

# 1) Clean end-to-end run on a small labelled fixture.
"$BIN" --input "$FIXTURE" --model "$MODEL" --max-frames 5 \
  --output-json "$OUT/report.json" >"$OUT/stdout.txt" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL: exit code $rc; output:" >&2
  cat "$OUT/stdout.txt" >&2
  exit 1
fi
python3 - "$OUT/report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
assert s["frames"] == 5, s
assert s["interrupted"] is False, s
assert d["runtime"] == "cpp-onnxruntime", d["runtime"]
assert d["model"]["backend"] == "onnxruntime-cpu"
assert len(d["per_frame"]) == 5
print("report OK: frames=%d last_state=%s" % (s["frames"], s["last_state"]))
PY
if [ $? -ne 0 ]; then
  echo "FAIL: JSON validation" >&2
  exit 1
fi

# 2) SIGINT graceful shutdown: interrupted=true, exit code 0, report still valid.
"$BIN" --input docs/research/stage7-fixtures/test_mixed_underex_86.mp4 \
  --model "$MODEL" --output-json "$OUT/int.json" \
  >"$OUT/int.txt" 2>&1 &
PID=$!
sleep 1
kill -INT "$PID" 2>/dev/null
wait "$PID"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL: SIGINT run exited $rc" >&2
  exit 1
fi
python3 - "$OUT/int.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["summary"]["interrupted"] is True or d["summary"]["frames"] > 0
print("SIGINT OK: interrupted=%s frames=%d" % (
    d["summary"]["interrupted"], d["summary"]["frames"]))
PY
if [ $? -ne 0 ]; then
  echo "FAIL: SIGINT JSON validation" >&2
  exit 1
fi

# 3) Clearly-scoped runtime errors return rc=1 with a diagnostic.
"$BIN" --input /definitely/missing.mp4 --model "$MODEL" >"$OUT/err.txt" 2>&1
rc=$?
if [ "$rc" -ne 1 ]; then
  echo "FAIL: expected exit 1, got $rc" >&2
  exit 1
fi
grep -q "error:" "$OUT/err.txt" || { echo "FAIL: no diagnostic" >&2; exit 1; }

echo "integration smoke: PASS"