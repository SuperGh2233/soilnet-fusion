# REQ/PLAN: Unify SoilNet fusion code while separating the system application

## Problem and business outcome

The local checkout contains active SoilNet experiments, while
`SuperGh2233/soilnet-sys` contains another copy of the model plus a regional
survey application. Different computers have therefore run different model
implementations. The outcome is a single fusion/research codebase published as
`SuperGh2233/soilnet-fusion`, while the application remains independently
managed in `SuperGh2233/soilnet-sys`.

## Users and workflow

Researchers continue to run `train.py`/`train_ssl.py` from the local repository
root. The system repository may consume the same model source through an
explicit checkout/path or packaged artifact, but owns its backend, frontend,
and model-service code separately.

## Scope

- Keep the local root `soilnet/` package as the canonical model location.
- Merge the remote `soilnet-sys/Soilnet/soilnet` model capabilities into that
  package rather than copying the whole system repository.
- Preserve local model/experiment features, including FiLM, spectral CNN, and
  parallel S-CMRL fusion.
- Bring in remote checkpoint-compatible fusion and `ViT-CoMerV2` support.
- Keep local training scripts, dataset loaders, experiment documents, and
  local assets available.
- Point the local Git `origin` at `https://github.com/SuperGh2233/soilnet-fusion.git`
  and publish the source branch as `main` after focused verification.
- Leave `SuperGh2233/soilnet-sys` unchanged; it remains the system-application
  repository.

## Non-goals

- Do not copy `backend/`, `frontend/`, `model-service/`, or system-only docs
  into this repository.
- Do not merge or move `dataset/`, `results/`, `bestmodel/`, `pre_model/`,
  `checkpoints/`, `.venv/`, or generated local assets.
- Do not run full training or regenerate data.
- Do not force-push, rewrite the system repository, or open a pull request.

## Current evidence and constraints

- Local `main` is `282229e3` and has user-owned tracked edits plus many
  untracked experiment files and large local assets.
- The system repository default branch is
  `0b1aa9ff563705b04711c398acbe6cce97c66e48`; its model copy is under
  `Soilnet/soilnet`.
- The new fusion repository exists, is public, and currently has no branch
  refs; it is safe to publish the prepared local source as its initial `main`.
- Windows path handling makes maintaining both top-level `Soilnet` and
  `soilnet` packages unsafe; the root lowercase package is the canonical one.

## Requirements

1. Root `soilnet/` remains importable by all existing training scripts.
2. Remote checkpoint compatibility is available without requiring the system
   repository's duplicate package.
3. Existing explicit fusion options remain available: standard S-CMRL
   sequential/parallel, legacy/parallel compatibility modes, and FiLM.
4. Existing explicit visual encoder options remain available, including local
   spectral CNN and remote `ViT-CoMerV2`.
5. Protected local data/result directories remain present and unstaged.
6. The new GitHub repository receives only intended source/docs files and the
   branch is named `main`.

## Data/API/state contracts

- The research repository keeps its current root-relative dataset/config paths.
- Model checkpoint state-dict names for the released checkpoint-compatible
  fusion remain loadable.
- `soilnet-sys` application contracts are out of scope for this repository;
  only the shared model package capability is consolidated here.

## Implementation plan

1. Record this boundary and inspect local/remote model differences.
2. Merge the remote model capability union into root `soilnet/`.
3. Add focused compatibility/smoke checks without touching local data.
4. Review Git ignore/staging boundaries and local status.
5. Rename the existing original GitHub remote to a read-only upstream name,
   add `soilnet-fusion` as `origin`, and push `main`.
6. Record verification evidence and the final remote commit.

## Risks and rollback

- Fusion state-dict names can differ between experiments. Keep the remote
  checkpoint-compatible classes and run their compatibility-oriented checks.
- The worktree is dirty. Do not stash with `--include-untracked` because local
  data is large; make only targeted source edits.
- If verification fails, do not push; restore only the touched source files from
  the temporary remote clone/reference and leave local assets untouched.
- If publication fails after local verification, keep `origin` configured and
  report the exact remote error; no force push is authorized.

## Acceptance criteria

- `D:\SoilNet\soilnet` is the only model implementation in the fusion repo.
- It supports local FiLM/spectral/parallel paths and remote
  checkpoint-compatible/ViT-CoMerV2 paths.
- Focused model tests pass or report only a clearly identified missing optional
  dependency/asset.
- `git diff --check` passes for touched source/docs.
- No protected local asset is deleted, moved, or staged.
- `origin` is `SuperGh2233/soilnet-fusion` and `main` is pushed successfully.

## Verification evidence

- `python -m py_compile` passed for the merged model and fusion modules.
- With `D:\\Anaconda\\envs\\pytorchGPU\\python.exe`,
  `test_semantic_aligned_fusion.py`, `test_scmrl_integration.py`, and
  `test_spectral_cnn.py` passed.
- `train.py --help` and `train_ssl.py --help` passed; `ViT-CoMerV2` is exposed
  by the supervised training CLI.
- A focused smoke check passed for checkpoint-compatible fusion parameter names,
  FiLM forward output, checkpoint model construction, and a small
  `ViT-CoMerV2` forward pass.
- The default `python` interpreter has no PyTorch, so model checks were run in
  the existing `pytorchGPU` environment; no dependency was installed.
- `soilnet/test_simple.py` remains blocked by its pre-existing import of
  `enhanced_climate_transformer` from the wrong top-level path and a Windows
  console encoding error in its failure message; it was not changed.
- `git diff --check` passed for all files touched by this plan; unrelated
  pre-existing blank-at-EOF warnings remain in other local files.

## Status and next action

Status: ready for publication. Next action: commit the staged source/docs
changes, switch `origin` to `SuperGh2233/soilnet-fusion`, push `main`, and then
record the resulting commit here.
