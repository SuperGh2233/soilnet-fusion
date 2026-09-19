# SoilNet agent guide

## Purpose and boundaries

This repository is the SoilNet research/fusion codebase: model definitions,
training, dataset loaders, experiments, and standalone checks. The regional
survey application lives in the separate repository
`SuperGh2233/soilnet-sys`; do not copy its backend, frontend, or model-service
directories into this repository.

## Commands

- Model smoke checks: `python soilnet/test_simple.py`
- Fusion checks: `python test_semantic_aligned_fusion.py` and
  `python test_scmrl_integration.py`
- Spectral encoder check: `python test_spectral_cnn.py`
- Revision protocol check: `python test_revision_protocol.py`
- Buffered split: `python prepare_revision_split.py`
- Revision queue: `python run_revision_experiments.py --matrix core`
- Result aggregation: `python analyze_revision_results.py --help`
- Training help: `python train.py --help` and `python train_ssl.py --help`
- Environment definition: `requirements/pytorch_reqs.yml`

Use the `pytorchGPU`/PyTorch environment for model work when available. Do not
run full training or regenerate datasets as part of a source merge.

## Architecture invariants

- Root `soilnet/` is the single canonical Python model package.
- `train.py` and `train_ssl.py` remain runnable from the repository root.
- Checkpoint-compatible fusion, standard S-CMRL sequential/parallel fusion,
  FiLM fusion, spectral CNN, and `ViT-CoMerV2` remain available when explicitly
  selected.
- `soilnet-sys` consumes this repository's model code through an explicit
  checkout/path or packaged artifact; it is not a second copy of the model.

## Data and safety constraints

- `dataset/`, `results/`, `bestmodel/`, `pre_model/`, `checkpoints/`, `.venv/`,
  and generated outputs are local assets. Do not delete, move, or stage them
  unless the user explicitly asks.
- Never commit secrets, model weights, generated databases, or experiment
  outputs.
- Do not force-reset or checkout over the dirty worktree.

## Test and quality gates

- Prefer focused standalone smoke checks over full experiments.
- Check `git status` before and after source changes.
- Run `git diff --check` before publishing.
- Record the active merge scope and acceptance evidence in the active plan.

## Code map

- `soilnet/`: canonical model and neural-network submodules.
- `dataset/`: dataset loaders and local data layout.
- `train.py`, `train_ssl.py`, `train_utils.py`: supervised and self-supervised
  training.
- `test_*.py`, `soilnet/test_simple.py`: focused model/data checks.
- `requirements/`: environment definitions.
- `docs/`: durable plans and documentation.

## Documentation map

- `docs/index.md`: document map and status registry.
- `docs/plans/active/PLAN-20260919-unify-soilnet-repositories.md`: active
  two-repository consolidation scope and acceptance criteria.
- `docs/plans/active/PLAN-20260919-grsl-major-revision-experiments.md`:
  reviewer-driven experiment protocol, acceptance criteria, and run order.
- `docs/evals/EVAL-20260919-submitted-split-reanalysis.md`: preliminary
  unclipped-target and bootstrap reanalysis of the submitted split.
- To change model behavior or merge remote model code, read the active plan
  first.
- To run, interpret, or prune code for the GRSL revision, read the revision
  plan and evaluation report first.
- To resume interrupted work, read `docs/agent-handoff.md` when present.

## Session recovery

Goals and acceptance criteria live in `docs/plans/active/`; transient progress
and the next executable action belong in `docs/agent-handoff.md`; durable
architecture decisions belong in `docs/architecture/`; contracts belong in
`docs/specs/`; operational rollback notes belong in `docs/operations/`.
