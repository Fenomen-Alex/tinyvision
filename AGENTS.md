# TinyVision Agent Guide

## Project

TinyVision is an open-source edge computer-vision project focused on small, specialized visual sensors.

The first target domain is expected to be 3D-print failure detection, but this must be validated during the initial research phase.

The long-term goal is a reusable local inference platform capable of running specialized vision models on ordinary hardware.

## Core principles

1. Local inference is a first-class requirement.
2. Small models and low resource consumption are important project goals.
3. Reproducibility matters more than clever experimentation.
4. Every experiment must have a recorded configuration and measurable results.
5. Dataset licensing must be verified before incorporating external data.
6. Do not fabricate dataset statistics, benchmark results, papers, licenses, or technical claims.
7. Prefer existing high-quality datasets and pretrained models before collecting large amounts of new data.
8. Keep training infrastructure separate from production inference.
9. Production inference should target C++.
10. Python is primarily a training and experimentation tool.
11. TypeScript is used for web tooling and demonstrations.
12. Avoid unnecessary frameworks and infrastructure.
13. Prefer deterministic CLI tools over notebook-only workflows.
14. Keep the repository usable by other AI coding agents.

## Technology direction

### Training

Python ecosystem.

Likely candidates:

* PyTorch
* Ultralytics
* OpenCV
* NumPy
* ONNX

The exact training stack must be validated during research.

### Production inference

C++.

Likely components:

* OpenCV
* ONNX Runtime
* CMake
* Ninja

The production runtime must remain independent from the Python training environment.

### Web

TypeScript.

The browser application exists primarily as an accessible demonstration and may eventually support local browser inference.

## Repository conventions

Use this general structure:

runtime/
C++ inference runtime

cli/
command-line tools

web/
browser demonstration

training/
model training and evaluation

datasets/
dataset metadata and tooling

models/
model metadata and release configuration

experiments/
reproducible experiment configurations and reports

benchmarks/
runtime and model benchmarks

scripts/
automation

docs/
technical documentation

## Agent behavior

Before implementing a significant feature:

1. Inspect the existing repository.
2. Read relevant documentation.
3. Identify assumptions.
4. Research current information when necessary.
5. Prefer evidence over guesses.
6. Explain important architectural decisions in documentation.
7. Implement the smallest useful change.
8. Run tests.
9. Run relevant benchmarks.
10. Update documentation.

Do not rewrite working components without a clear reason.

Do not introduce dependencies merely for convenience.

Do not silently change model architecture, dataset definitions, evaluation criteria, or benchmark methodology.

## Dataset policy

Every external dataset must have a record containing:

* name
* source
* URL
* license
* image count
* annotation format
* classes
* resolution information when available
* collection methodology when available
* known limitations
* intended usage
* retrieval date

Unknown or incompatible licensing means the dataset cannot be included until its licensing status is resolved.

Never mix datasets into training without recording their provenance.

## Experiment policy

Every training experiment must record:

* experiment ID
* model
* model version
* dataset version
* dataset split
* image size
* epochs
* batch size
* optimizer
* learning rate
* augmentation configuration
* hardware
* software versions
* random seed where applicable
* precision
* evaluation metrics
* model size
* inference latency

Experiments must be reproducible.

## Evaluation

Accuracy alone is insufficient.

Track appropriate metrics such as:

* precision
* recall
* F1
* mAP where applicable
* false-positive rate
* false-negative rate
* inference latency
* FPS
* memory consumption
* model size

For failure detection, false negatives require particular attention.

Evaluation datasets must be kept separate from training data.

Avoid data leakage.

## Edge requirements

Every candidate model should eventually be evaluated for:

* model size
* RAM consumption
* CPU latency
* FPS
* image resolution
* quantization impact
* accuracy degradation after export
* behavior under poor lighting
* behavior with different cameras

A model that performs well on a workstation but cannot run efficiently on an ordinary edge device is not automatically considered successful.

## Current task

The first task is RESEARCH ONLY.

Do not start implementing the complete application.

Investigate the candidate problem areas and determine whether 3D-print failure detection is the strongest first project.

Produce a research report covering:

1. Candidate problem
2. Target users
3. Existing solutions
4. Existing open-source projects
5. Public datasets
6. Dataset size
7. Dataset licensing
8. Dataset annotations
9. Dataset quality
10. Dataset limitations
11. Existing pretrained models
12. Training difficulty
13. Expected edge inference requirements
14. Potential for collecting our own data
15. Potential for synthetic data
16. Potential project differentiation
17. Estimated engineering effort
18. Risks
19. Recommended MVP

Investigate at least these candidates:

* 3D-print failure detection
* machine-state detection
* agricultural fruit monitoring
* waste sorting
* small-workshop visual inspection/safety

Do not select a winner based on intuition alone.

Create:

docs/research/problem-selection.md

Create any supporting structured data under:

docs/research/

Do not download large datasets yet.

Do not begin model training yet.

Do not create fake benchmark results.

After completing the research, stop and present the findings and recommendation.
