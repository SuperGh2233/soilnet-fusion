# EVAL: Submitted-split unclipped-target reanalysis

## Purpose

Provide immediate evidence for reviewer comments about target clipping and
statistical support while the corrected spatially blocked experiments run.
This report is preliminary because it uses the geographically mixed submitted
split and one selected seed per method.

## Inputs and protocol

- Source labels: `dataset/CN-SOC-3500_new.csv`.
- Concat predictions: the submitted C+V+S A4 output.
- SAP-RF predictions: the submitted C+V+S A6 output.
- Predictions were joined to original SOC by normalized `Point_id`.
- Original-scale metrics use the untouched SOC values; clipped metrics use
  `min(SOC, 60)` only for comparison with the manuscript.
- A paired point-level bootstrap used 2,000 resamples for this preliminary
  check. The final corrected analysis will use 10,000 resamples and five seeds.

## Results

| Method | Target | n | MAE | RMSE | R2 | RPIQ | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Concat | clipped | 525 | 4.9909 | 7.0235 | 0.5093 | 1.3142 | 0.6771 |
| SAP-RF | clipped | 525 | 4.7317 | 6.7969 | 0.5404 | 1.3580 | 0.7032 |
| Concat | original | 525 | 5.1761 | 7.9266 | 0.4943 | 1.1644 | 0.6441 |
| SAP-RF | original | 525 | 4.9169 | 7.6706 | 0.5264 | 1.2033 | 0.6721 |
| Concat | SOC > 60 | 5 | 43.3197 | 45.3028 | -5.9160 | 0.4046 | 0.0829 |
| SAP-RF | SOC > 60 | 5 | 41.1331 | 43.6385 | -5.4172 | 0.4200 | 0.0911 |

For SAP-RF minus concat on original SOC, the paired bootstrap estimated:

| Metric | Difference | 95% CI | two-sided p |
|---|---:|---:|---:|
| MAE | -0.2592 | [-0.4663, -0.0528] | 0.013 |
| RMSE | -0.2560 | [-0.5868, 0.0922] | 0.139 |
| R2 | +0.0321 | [-0.0117, 0.0717] | 0.139 |

## Interpretation

The original-value evaluation lowers both methods' aggregate scores and shows
severe upper-tail underestimation. Only five submitted test points exceed
60 g/kg, so upper-tail R2 is unstable and must not support a broad tail claim.
The one-seed paired bootstrap supports an MAE improvement but not RMSE or R2 at
the 0.05 level. These values should be described as preliminary and replaced by
the corrected five-seed spatial protocol.

## Status and next action

Status: active preliminary evidence. Next action: rerun the same aggregation on
the five corrected core configurations and 10,000 bootstrap resamples.
