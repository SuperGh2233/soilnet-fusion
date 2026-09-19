# REQ/PLAN: GRSL major-revision experiments

## Problem and business outcome

The submitted GRSL letter reports SoilNet-Fusion on one geographically mixed
split and mostly single-run results. Both reviewers requested evidence against
spatial leakage, evaluation on original unclipped SOC, repeated trials with
uncertainty and significance, competitive fusion baselines, direct reporting
of learned residual scales, hyperparameter sensitivity, and clearer temporal
and static-covariate provenance. The outcome is a reproducible revision
experiment package that supports only claims demonstrated by the revised
evidence.

## Users and workflow

The paper authors run one revision experiment matrix from this repository,
resume interrupted runs without overwriting completed outputs, aggregate all
seed-level predictions, and use the generated tables and maps in the response
letter and revised manuscript.

## Scope

- Freeze the January 2026 SAP-RF implementation used for the submitted result
  as an explicit reproducibility mode.
- Generate a deterministic geographically blocked 70:15:15 split from the
  available China samples and enforce a 50 km buffer between subsets.
- Fit climate/static preprocessing statistics on the training subset only and
  reuse them unchanged for validation and test data.
- Evaluate both the clipped range and original unclipped SOC, including a
  separate upper-tail report for SOC greater than 60 g/kg.
- Run at least five fixed seeds for the key C+V+S comparisons: concatenation,
  SAP-RF without InfoNCE, and SAP-RF with InfoNCE.
- Add pooled gated fusion and token-level cross-attention baselines using the
  same encoders, split, preprocessing, optimizer, and seed set.
- Persist final alpha values and seed-level predictions; report mean, sample
  standard deviation, confidence intervals, and paired bootstrap comparisons.
- Run focused sensitivity experiments for alpha initialization, InfoNCE weight,
  and temperature.
- Generate a train/validation/test map and geographic residual maps with a
  diverging residual scale.
- Inventory 2014 imagery availability and clearly separate feasible temporal
  experiments from manuscript-only justification.
- After the revision pipeline is stable, remove code and documents unrelated
  to this experiment dependency graph, then publish a clean repository history.

## Non-goals

- Do not delete, move, overwrite, or commit local datasets, checkpoints, or
  historical results.
- Do not use the best seed as the headline repeated-run result.
- Do not present the current random/geographically mixed split as spatially
  blocked.
- Do not claim temporally matched validation unless 2014 imagery is acquired
  and evaluated.
- Do not remove citation, license, or scientific provenance merely to alter the
  GitHub contributor display.

## Current evidence and constraints

- The paper uses 3,455 samples from 2014; 48 original SOC values exceed the
  60 g/kg clipping threshold.
- The active image split contains 2,405/520/525 train/validation/test samples.
  In the test subset, 218 samples are within 10 km, 386 within 25 km, and 484
  within 50 km of a training sample. It is not a buffered spatial split.
- A 3-degree geographic block split with a 50 km buffer is feasible on the
  available coordinates and retains approximately 2,067/494/531 samples before
  filtering the five CSV records whose imagery is unavailable.
- `ChinaSNDatasetClimateStatic` currently fits missing-value means,
  `StandardScaler`, and category mappings on the complete static CSV.
  `NormalizeClimDF` likewise computes climate minima and maxima across every
  row loaded from the complete climate tables. Both must become train-only.
- The published 0.5404 result is a single selected seed using alpha=2.0,
  lambda=0.1, tau=0.07, batch size 32, and the legacy vector-input parallel
  residual implementation. The current merged model also contains a newer
  token-aware implementation, so the legacy path must be explicit in the CLI.
- An existing six-seed run for the submitted configuration reports seed-level
  R2 values, but it predates the spatial/preprocessing corrections and lacks
  every seed's saved predictions. It is supporting evidence, not the final
  revised protocol.
- One RTX 3080 with 10 GB VRAM is available. A five- or six-seed deep run takes
  roughly five to eight hours with the submitted settings.
- No local 2014 imagery directory was found; available imagery directories
  correspond to the existing 2015 pipeline. A 2014-vs-2015 ablation therefore
  requires external data acquisition before it can run.

## Requirements

1. Every experiment records split ID, preprocessing provenance, model/fusion
   mode, seed, alpha/lambda/tau, checkpoint, predictions, and wall time.
2. All compared deep models use identical encoders and training settings; only
   the stated fusion or loss component may differ.
3. The same five seeds are used for paired comparisons.
4. Test-set model selection is forbidden; validation loss selects checkpoints.
5. Metrics are reported on original SOC and, separately, the clipped range and
   upper tail. The number of evaluated samples accompanies each metric block.
6. Paired bootstrap resamples matched test points and reports the confidence
   interval and two-sided p-value for R2, RMSE, and MAE differences.
7. Final alpha values are stored per seed, not recovered from console logs.
8. Generated split manifests and result summaries are small, reviewable files;
   image rasters, model weights, and raw predictions remain ignored local
   artifacts unless explicitly packaged outside Git.

## Data/API/state contracts

- The canonical sample key is normalized string `Point_id`.
- The spatial split artifact contains `Point_id`, subset, geographic block,
  and buffer distance/provenance. A point appears in exactly one subset.
- Climate and static preprocessing parameters are fit from training point IDs
  only and serialized with each experiment.
- Prediction artifacts contain at least experiment ID, seed, Point_id,
  original target, clipped target, prediction, latitude, and longitude.
- Existing result/checkpoint files are read-only and remain reproducible through
  the legacy fusion mode.

## Implementation plan

1. Add a non-copying split/manifest generator and validation check for the
   3-degree grouped, 50 km buffered split; generate the split map.
2. Refactor dataset preprocessing so the training dataset owns fitted climate
   and static statistics and validation/test datasets consume those statistics.
3. Expose explicit fusion modes and the legacy SAP-RF path through one training
   CLI; save per-seed predictions, alpha values, and complete run metadata.
4. Add minimal gated and cross-attention baselines around the existing encoded
   representations and verify parameter counts.
5. Add one revision runner with resumable jobs for the five-seed core matrix and
   one aggregation script for uncertainty, paired bootstrap, tail metrics, and
   spatial figures.
6. Run fast checks, then launch core experiments in this order: corrected
   concatenation, SAP-RF without InfoNCE, SAP-RF with InfoNCE, gated fusion,
   cross-attention. Run sensitivity jobs only after the core matrix succeeds.
7. Record results in `docs/evals/`, update manuscript tables/figures and the
   reviewer response, then remove code outside the proven revision dependency
   graph and perform the requested clean-history publication.

## Risks and rollback

- Correct spatial blocking and train-only preprocessing may lower accuracy.
  Report the corrected result; do not tune the test set to recover the original
  number.
- Buffering reduces training size and changes target distribution. Preserve the
  split manifest and report subset counts/distributions.
- The manuscript equations may not exactly describe the legacy code path.
  Freeze and test the path before rerunning; revise the equations or method name
  to match the executed implementation.
- Long GPU jobs may be interrupted. Each seed writes to an independent output
  directory and the runner skips only verified completed jobs.
- Historical results and current remote history provide rollback. No force push
  occurs until the revision code and result dependency graph are finalized.

## Acceptance criteria

- Split validation proves zero Point_id overlap and at least 50 km nearest-set
  separation after filtering available imagery.
- A regression check proves validation/test preprocessing uses training-only
  fitted statistics.
- Five completed seeds exist for all five core fusion configurations on the
  corrected protocol.
- The aggregate table reports mean +/- sample standard deviation and paired
  bootstrap inference with matched sample counts.
- Clipped, original-unclipped, and SOC>60 g/kg metrics are all present.
- Alpha convergence and alpha/lambda/tau sensitivity tables are generated.
- Split and residual maps are generated at publication resolution.
- `train.py --help`, focused fusion/data checks, and `git diff --check` pass.
- No dataset, checkpoint, or historical result is deleted or committed.

## Verification evidence

- Paper and both reviewer reports were inspected in full on 2026-09-19.
- The current random split's nearest-neighbor leakage and the original-target
  tail count were measured directly from local data.
- CUDA is available through the existing `pytorchGPU` environment on an NVIDIA
  GeForce RTX 3080.
- The generated 3-degree/50 km manifest contains 2,064 training, 492 validation,
  531 test, and 363 buffer-excluded samples; pairwise subset distances are all
  at least 50 km.
- A focused contract check confirmed that climate and static preprocessing
  statistics fitted on training IDs are reused unchanged by validation/test;
  the normalized training static columns have mean zero and unit variance.
- A one-epoch end-to-end smoke run confirmed validation-only checkpoint
  selection, per-seed prediction output, and revision metadata serialization.
- Preliminary reanalysis of the submitted predictions produced original-value
  and upper-tail metrics plus paired-bootstrap evidence in
  `docs/evals/EVAL-20260919-submitted-split-reanalysis.md`.
- The corrected repository revision `c399d97` was deployed to an NVIDIA A100
  40 GB server. The transferred experiment subset was verified as 3,451 image
  files and 694 climate files; both the transfer archive and split manifest
  matched their local SHA-256 hashes.
- An A100 one-epoch SAP-RF smoke run completed validation-only checkpoint
  selection and wrote predictions, metadata, and `run_summary.json`.
- The resumable 25-job core queue started on 2026-09-19 at approximately 22:25
  CST. At 22:49 it was on concat seed 1, epoch 45/60, with no error logs.
- A separate guarded follow-up process now waits for exactly 25 verified core
  markers, runs the seven sensitivity jobs, requires 32 total markers, and
  then runs the 10,000-sample aggregate/bootstrap analysis.

## Status and next action

Status: active. The corrected protocol is deployed and the five-configuration,
five-seed core A100 queue is running. The next executable action is to monitor
the completion markers and error logs; the guarded follow-up process will run
the sensitivity matrix and aggregate analysis automatically after the core
queue succeeds. The 2014-vs-2015 temporal experiment remains unavailable until
matched 2014 imagery is acquired.
