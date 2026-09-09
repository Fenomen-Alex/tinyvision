# Recording Guide for TinyVision Contributors

The goal is **one camera environment**: a fixed camera, fixed viewpoint,
fixed(ish) lighting, recording a print continuously in real time.

## Setup

* Mount the camera so it does not move between sessions. A phone on a
  stand/arm, a webcam, or the printer's built-in camera all work. Note where
  it is: front, side, above, or inside the chamber.
* Frame the whole printer (or at least the build plate plus the nozzle).
  Keep the bed and a visible part of the printer in frame — the model uses
  the visible printer appearance as context.
* Keep the light on and constant (a lamp or daylight); avoid a light source
  moving across the frame or a window behind the printer.
* Do not move furniture, tools, or the printer between sessions.
* Record the same scene every time, even if you change filament or printer.

## Recording

* Use **real-time continuous** footage (normal speed). Do **not** use
  timelapse mode, do not cut or speed things up, do not superimpose text or
  logos — those break the cadence the model is trained on.
* Resolution: whatever the camera gives (720p+ preferred; 1080p ideal).
* Duration: one full print is excellent. A capture of at least several minutes
  is useful. Shorter clips are fine for failure moments.
* Filename convention (keeps provenance obvious):
  `<printer-or-phone>_<date>_<session>.mp4` (e.g. `ender3_2026-09-07_01.mp4`).

## What to record

Priority order (what we are missing the most):

1. **Failed prints**: stringing, spaghetti/bird's-nest, bed-adhesion failure /
   first-layer peeling, under-extrusion, layer shift, warping. Record the
   whole event from the moment it starts until you stop the print.
2. **Normal prints** that go start to finish without any defect.

Record only your own printing sessions (see the consent form about you owning
the footage).

## Metadata sheet

Send a small text block with the footage:

```
printer: <make/model, e.g. Ender 3 Pro>
camera: <phone/webcam model, or printer built-in camera>
camera_position: <front / side / above / inside chamber>
lighting: <indoor constant / window daylight / shop light>
background_notes: <e.g. wooden desk, white wall>
print_result: normal | failed
if_failed_failure_type: <stringing | spaghetti | bed_adhesion | under_extrusion | layer_shift | warping | other>
failure_onset_guess: <approx time into the print when it started going wrong, if known>
permission: <see contributor-consent.md>
```

That is the whole ask. Questions: open an issue on the project repository.