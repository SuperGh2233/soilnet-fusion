# Documentation index

## Start here

- `AGENTS.md` (stable guide): repository boundaries, commands, invariants, and recovery pointers.
- `README.md` (human quickstart): research training and local experiment setup.

## Active requirements and plans

- `docs/plans/active/PLAN-20260919-grsl-major-revision-experiments.md` (active): reviewer-to-experiment matrix, corrected evaluation protocol, and execution order for the GRSL major revision.
- `docs/plans/active/PLAN-20260919-unify-soilnet-repositories.md` (completed): merge local model experiments with the remote SoilNet model code while keeping the system application in its own repository.

## Evaluations

- `docs/evals/EVAL-20260919-submitted-split-reanalysis.md` (active): preliminary unclipped-target and paired-bootstrap analysis of the submitted A4/A6 predictions; replace with corrected spatial-protocol results when complete.

## Architecture and specifications

- None yet; add a dedicated ADR/spec only when a future change introduces a durable architecture or contract decision.

## Operations and recovery

- `docs/agent-handoff.md` (active): transient A100 queue state, recovery checks, and the next executable action.

## Document discipline

New durable documents belong under a type directory, use date-based names when no issue number exists, and are registered here in the same change.
