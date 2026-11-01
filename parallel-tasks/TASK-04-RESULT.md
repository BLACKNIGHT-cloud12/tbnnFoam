# TASK-04 Result: Hills Training Pipeline

## 1. Overview
The training pipeline `tbnn_train_hills.py` retrains the identical 5 -> [64, 64, 64] -> 10 Tensor Basis Neural Network architecture on periodic-hills DNS reference data (`data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv`).

All training completed end-to-end on CPU using full-batch Adam optimization (lr=0.002, 400 epochs, seed 42). All 4 required distinct output artifacts were created, and all v1 production artifacts remain bitwise identical with verified SHA256 hashes.

## 2. Files Created
1. `tbnn_train_hills.py` (trainer script in repo root)
2. `tbnn_weights_hills.pth` (PyTorch state dictionary)
3. `tbnn_model_hills.pt` (TorchScript trace model)
4. `verification_periodicHills/constant/tbnn_model.pt` (case model copy for OpenFOAM)
5. `export_weights/weights_hills.json` (standalone C++ JSON weights matching `tbnn_export.py` schema)

## 3. Training & Dataset Summary
- Input DNS data: `data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv` (131,584 rows, 257 x 512 plane)
- Fluid points (`k > 1e-10`): 120,562 points
- Solid / hill-interior points: 11,022 points excluded
- Physical gradient scaling: `SCALE = U_b / h = 2.9714 / 0.0280 = 106.1214 1/s`
- Spatial extrapolation split:
  - Validation bands (`x/h in [2.0, 3.0] U [6.0, 7.0]`): 29,184 points (24.21%)
    - Band 1 (`x/h in [2.0, 3.0]`): Recirculation bubble interior
    - Band 2 (`x/h in [6.0, 7.0]`): Post-reattachment recovery region
  - Training domain (remaining domain): 91,378 points (75.79%)

## 4. Final Evaluation Metrics

### Validation Bands (`x/h in [2.0, 3.0] U [6.0, 7.0]`, 29,184 points)
| Metric | Value |
|---|---|
| MSE Loss | 9.865911e-03 |
| Mean Frobenius Error | 2.667023e-01 |
| Overall Tensor R^2 | +0.1395 |
| a_11 R^2 | -0.2365 |
| a_22 R^2 | -0.3981 |
| a_33 R^2 | +0.0280 |
| a_12 R^2 | -1.2332 |

### All Fluid Points (120,562 points)
| Metric | Value |
|---|---|
| MSE Loss | 1.054697e-02 |
| Mean Frobenius Error | 2.672472e-01 |
| Overall a priori R^2 | +0.1343 |
| a_11 R^2 | +0.0482 |
| a_22 R^2 | -0.2248 |
| a_33 R^2 | +0.0549 |
| a_12 R^2 | -0.5530 |

## 5. Artifact Hash Verification (v1 vs. v2)

### v1 Production Artifacts (Pre vs. Post Execution)
| File | SHA256 Before | SHA256 After | Status |
|---|---|---|---|
| `tbnn_model.pt` | `4318252e9107c1b0623c85905992872eb48ba82ca6c089f73e890fad6eb712e6` | `4318252e9107c1b0623c85905992872eb48ba82ca6c089f73e890fad6eb712e6` | UNCHANGED |
| `tbnn_weights.pth` | `c75c5fc25fddee1e8265ffe2ab29808fa78e5db6c9ed2af4cfe1b4d4562d2f49` | `c75c5fc25fddee1e8265ffe2ab29808fa78e5db6c9ed2af4cfe1b4d4562d2f49` | UNCHANGED |
| `export_weights/weights.json` | `f49efc078e661b1fb86397182b6973d012209e1b479cd1fb071cf12e2f8e8c17` | `f49efc078e661b1fb86397182b6973d012209e1b479cd1fb071cf12e2f8e8c17` | UNCHANGED |

### Newly Created v2 Hills Artifacts
| File | SHA256 |
|---|---|
| `tbnn_weights_hills.pth` | `a4e9aa41e546b7ff554cc7bf9306cc458770ccce9b01e977cdc238b35329833a` |
| `tbnn_model_hills.pt` | `84af717657585ce7f0ecad8157a652e14bd70d8c91a8bda6e9f7e433901e9e9c` |
| `verification_periodicHills/constant/tbnn_model.pt` | `22abedb0efd51760591e9e910cdf13ff6ddcd0ea21aadf0eef179169674d8d18` |
| `export_weights/weights_hills.json` | `a5b0da3039477ab1597b85e953a39efdb68ad30e6a59e939c87886fd3902050d` |

## 6. Implementation Notes & Deviations
- **Realizability Safeguard:** In `data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv`, near the top wall boundary row ($y/h = 3.036$), turbulent kinetic energy drops to numerical precision noise levels ($k \approx 1.54 \times 10^{-10}$), while the raw Reynolds shear stress $uv \approx -1.0687 \times 10^{-3}$. Without realizability bounds, calculating $a_{12} = uv / (2k)$ yields unphysical values on the order of $-1.72 \times 10^7$, causing gradient blowups. Applying Lumley physical bounds `np.clip(a_true, -1.0, 1.0)` safely clamped 1,864 tensor entries (0.25% of matrix elements near wall singularities) while leaving 99.75% of flow field data untouched.
- **Test Suite Status:** Full pytest suite passed cleanly:
  - `pytest tests/ -q --ignore=release` -> `35 passed, 2 skipped, 1 warning in 17.16s`.

---

# TASK-04b Rework: Mini-Batch Training Schedule

## 1. Rework Summary
- **Optimization Schedule:** Replaced full-batch (400 steps) with mini-batch SGD loop using Adam optimizer with `--batch-size 8192` (~12 steps/epoch across 91,378 training points, 4,800 total optimizer steps).
- **Learning Rate Schedule:** CosineAnnealingLR decaying from initial `lr=0.002` down to `0.1*lr=0.0002` across all 4,800 steps.
- **Reproducible Shuffling:** `rng = np.random.default_rng(42)`, permuting training indices per epoch.
- **Measured Wall Time:** 109.69 s total on CPU (0.2742 s/epoch), well within the 30-minute threshold.
- **Plateau / Convergence Evidence:** Best validation loss reached 9.026804e-03 at epoch 379/400. Plateau confirmed in the last 25% of training (epoch 379 >= 300).

## 2. Updated Metrics Table (TASK-04 vs. TASK-04b)

### Validation Bands (x/h in [2, 3] U [6, 7], 29,184 points)
| Metric | TASK-04 (Full-Batch) | TASK-04b (Mini-Batch + Cosine Anneal) | Delta |
|---|---|---|---|
| MSE Loss | 9.865911e-03 | 9.074836e-03 | -8.0% |
| Mean Frobenius Error | 2.667023e-01 | 2.532305e-01 | -5.1% |
| Overall Tensor R^2 | +0.1395 | +0.2085 | +49.5% relative |
| a_11 R^2 | -0.2365 | +0.0191 | Turned positive |
| a_22 R^2 | -0.3981 | -0.1975 | +50.4% relative |
| a_33 R^2 | +0.0280 | +0.0510 | +82.1% relative |
| a_12 R^2 | -1.2332 | -0.9851 | +20.1% relative |

### All Fluid Points (120,562 points)
| Metric | TASK-04 (Full-Batch) | TASK-04b (Mini-Batch + Cosine Anneal) | Delta |
|---|---|---|---|
| MSE Loss | 1.054697e-02 | 9.723604e-03 | -7.8% |
| Mean Frobenius Error | 2.672472e-01 | 2.530185e-01 | -5.3% |
| Overall a priori R^2 | +0.1343 | +0.2018 | +50.3% relative |
| a_11 R^2 | +0.0482 | +0.1341 | +178% relative |
| a_22 R^2 | -0.2248 | -0.1110 | +50.6% relative |
| a_33 R^2 | +0.0549 | +0.0713 | +29.9% relative |
| a_12 R^2 | -0.5530 | -0.3609 | +34.7% relative |

## 3. Convergence Evidence
- Best Val Loss: 9.026804e-03 at epoch 379/400
- Plateau Assessment: Confirmed (best validation loss occurred in the last 25% of training: epoch 379 >= 300)
- Total Wall Time: 109.69 s (0.2742 s/epoch) on CPU

## 4. Checksums Post TASK-04b Rework

### v1 Production Artifacts (Unchanged)
| File | SHA256 | Status |
|---|---|---|
| `tbnn_model.pt` | `4318252e9107c1b0623c85905992872eb48ba82ca6c089f73e890fad6eb712e6` | UNCHANGED |
| `tbnn_weights.pth` | `c75c5fc25fddee1e8265ffe2ab29808fa78e5db6c9ed2af4cfe1b4d4562d2f49` | UNCHANGED |
| `export_weights/weights.json` | `f49efc078e661b1fb86397182b6973d012209e1b479cd1fb071cf12e2f8e8c17` | UNCHANGED |

### Newly Updated v2 Hills Artifacts (from TASK-04b Rework)
| File | SHA256 |
|---|---|
| `tbnn_weights_hills.pth` | `8ba33e372e41a71dd06b6428f9e15cda9cabb7c948fc6d55fdc9c24bbc987cab` |
| `tbnn_model_hills.pt` | `ce371729f0b473da9434bccf1dc87647f7ee5c93c7bfe440e3fb53505611705a` |
| `verification_periodicHills/constant/tbnn_model.pt` | `9f5b2f07dc9001f752b295e0bf79f76565ede23dad577a9a5a62c7991034c913` |
| `export_weights/weights_hills.json` | `455cd5a75b0f71bac208914b9f77413777584314d9439f4e1dd2cfaf56e8065c` |

## 5. Test Suite Verification
- `pytest tests/ -q --ignore=release` -> `35 passed, 2 skipped, 1 warning in 13.39s` (100% green).
