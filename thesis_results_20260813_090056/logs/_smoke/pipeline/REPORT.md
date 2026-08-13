# Pipeline report

Generated 2026-08-13T09:00:56+00:00 · profile **smoke** · seeds [42] · GPU

> **Not reportable.** This profile shortens the fixed training protocol, so these numbers check the wiring, not the science. Re-run with `--profile full` for results.

## Stages

5 blocked · 22 failed · 3 skipped

| Stage | Status | Minutes |
|---|---|---|
| `step04_audit` | failed | 0.2 |
| `step06_preprocessing` | failed | 0.2 |
| `step06_materialise` | skipped | - |
| `step08_imbalance` | failed | 0.2 |
| `step09_baselines/baseline_simple_cnn/seed_42` | failed | 0.2 |
| `step09_baselines/baseline_resnet50/seed_42` | failed | 0.2 |
| `step09_baselines/baseline_efficientnet_b0/seed_42` | failed | 0.2 |
| `step09_baselines/baseline_vit/seed_42` | failed | 0.3 |
| `step09_baselines/baseline_swin/seed_42` | failed | 0.2 |
| `step09_baselines/baseline_fixed_qcnn/seed_42` | failed | 0.2 |
| `step09_baselines/baseline_fixed_multiscale/seed_42` | failed | 0.2 |
| `step10_classical/seed_42` | failed | 0.2 |
| `step10_embeddings` | skipped | - |
| `step11_arm_ablation/arm1_fixed_3x3/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm2_fixed_5x5/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm3_fixed_dilated/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm4_concat_nogate/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm5_global_gate/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm6_spatial_gate/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm7_spatial_gate_quantum/seed_42` | failed | 0.2 |
| `step11_arm_ablation/arm8_global_gate_quantum/seed_42` | failed | 0.2 |
| `step11_gate_morphology` | skipped | - |
| `step12_adaptive_quantum/seed_42` | failed | 0.2 |
| `features` | blocked | - |
| `step13_fusion` | failed | 0.2 |
| `step14_loss_selection` | failed | 0.2 |
| `step15_final/seed_42` | blocked | - |
| `step16_internal` | blocked | - |
| `step17_external` | blocked | - |
| `step18_robustness` | blocked | - |

## Headline results

_No summaries written yet._
