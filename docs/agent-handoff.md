# Agent handoff: GRSL revision experiment queue

## Objective

Complete the corrected spatially buffered GRSL revision experiment matrix,
aggregate seed-level results, and use only verified evidence in the revised
paper and reviewer response. Durable scope and acceptance criteria remain in
`docs/plans/active/PLAN-20260919-grsl-major-revision-experiments.md`.

## Git state

- Local branch: `main` at deployed source revision `c399d97` before this
  handoff update.
- Remote repository: `SuperGh2233/soilnet-fusion`.
- The source worktree was clean when the experiment was deployed.
- Experiment data, checkpoints, predictions, and generated outputs remain
  ignored and must not be committed.

## Running tasks

- Server checkout: `/home/u104754241373/soilnet-fusion`.
- Core runner PID file: `revision_outputs/runner_core.pid`.
- Core runner log: `revision_outputs/runner_core.log`.
- Per-job logs: `revision_outputs/logs/`.
- Verified completion markers: `revision_outputs/completed/`.
- Guarded follow-up PID file: `revision_outputs/runner_followup.pid`.
- Guarded follow-up log: `revision_outputs/runner_followup.log`.
- The core runner executes five fusion configurations by five seeds, 60
  epochs each. The follow-up waits for 25 core markers, runs seven sensitivity
  jobs, requires 32 total markers, and then writes the corrected aggregate
  analysis under `revision_outputs/analysis/corrected_spatial/`.

No password or authentication material is stored in the repository.

## Latest verification

- At 2026-09-19 22:49 CST, concat seed 1 had reached epoch 45/60.
- The core runner and its direct training child were alive.
- No job log contained `Traceback`, `RuntimeError`, or CUDA out-of-memory text.
- The A100 smoke run produced its expected checkpoint, prediction, metadata,
  and summary artifacts.
- Server data counts were 3,451 files in `dataset/l8_images_CN` and 694 files
  below `dataset/Climate`; the split manifest SHA-256 was
  `a283ac88af7e44d25ac1ad93540141ca8d957f35b2003877e373e9a2c1b6ea72`.

## Problems and risks

- No matched 2014 image directory is available, so the 2014-vs-2015 temporal
  ablation is not part of the running batch.
- Correct spatial blocking may reduce reported accuracy; do not tune against
  the test set or substitute the submitted geographically mixed split.
- Do not rewrite Git history for contributor cleanup while the deployed
  experiment revision is active.
- Do not treat a runner exit as success: require 25 core markers and 32 total
  markers exactly, plus zero error logs.

## Next executable action

Connect to the existing server account using credentials supplied outside the
repository, change to the server checkout, and inspect:

```bash
ps -p "$(cat revision_outputs/runner_core.pid)" -o pid,stat,cmd
ps -p "$(cat revision_outputs/runner_followup.pid)" -o pid,stat,cmd
find revision_outputs/completed -maxdepth 1 -type f | wc -l
grep -l -E 'Traceback|CUDA out of memory|RuntimeError' revision_outputs/logs/*.log
tail -n 40 revision_outputs/runner_core.log
tail -n 40 revision_outputs/runner_followup.log
```

When `runner_followup.log` ends with `PIPELINE_COMPLETE`, verify the aggregate
tables and residual maps, copy the small result summaries back to the local
workspace, update the evaluation report, and draft the manuscript/reviewer
response changes. If the follow-up aborts, diagnose the first failed job and
rerun the resumable matrix; completed marker files will be skipped.
