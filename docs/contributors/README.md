# TinyVision Contributor Program

TinyVision is an open-source edge computer-vision project studying 3D-print
failure detection with a focus on small, low-resource models running on the
printer (or a nearby computer).

You can help by contributing **short, fixed-camera recordings of 3D prints**
— either normal prints or prints that fail (stringing, spaghetti/bird's nest,
bed-adhesion failure, under-extrusion, layer shift, warping).

This page and the linked guide are what we hand to a prospective contributor.
They are intentionally short and non-scary.

## Why you are needed

Public footage is dominated by a handful of camera environments. A model
trained on "five identical bed-mounted webcams" does not generalise to a
different camera, angle, or room. Each genuinely new contributor adds a new
environment that makes the model more robust for everyone.

## What we ask for

1. **A fixed camera** (phone, webcam, or printer camera) pointed at your
   printer, recorded with the frame stable and continuous.
2. **At least one session**, recorded in real time (not a timelapse).
3. A **short metadata sheet**: printer model, camera, angle/position,
   lighting, and the print result (see `recording-guide.md`).
4. Your **permission** for how the footage may be used
   (see `contributor-consent.md`).

## What we do with your footage

* It is stored together with its `environment_id` so that frames from your
  camera never leak across our training/evaluation split.
* It is used to train and evaluate TinyVision models. Footage may be published
  as part of the open dataset if you grant redistribution rights; otherwise it
  is used for training/evaluation only and never redistributed.
* Footage is attributed to you under the credit you choose.

## What we never do

* We never claim your raw footage contains guaranteed-correct labels — labels
  are initially marked `pending` and are only certified after human review.
* We never mix your footage into training without recording its provenance.

## Getting started

1. Read `recording-guide.md`.
2. Fill in and return `contributor-consent.md` (delete the section on
   redistribution if you do not want your footage published).
3. Upload the recordings and the metadata sheet, or run the capture tool:
   `tv capture --output-dir <dataset> --class normal|failing ...`.

Tip: the fastest useful contribution is **one full failed print from a camera
you keep mounted in the same place** (a fixed phone or webcam), because
failure environments — especially spaghetti-type and under-extrusion — are the
rarest thing we are missing.