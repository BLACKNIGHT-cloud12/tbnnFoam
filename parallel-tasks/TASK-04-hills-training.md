# TASK-04 — Hills training pipeline (`tbnn_train_hills.py`)

Fully-specified new module. Do not redesign. Do NOT modify any existing file — in
particular do NOT overwrite `tbnn_model.pt`, `tbnn_weights.pth`, or
`export_weights/weights.json` (those are the v1 production artifacts).

## Context

tbnnFoam v1 trained a 5 → [64, 64, 64] → 10 tensor-basis network on square-duct DNS.
v2 retrains the IDENTICAL architecture on periodic-hills DNS (canonical Breuer geometry,
Re = 5600). The training data is the frozen-schema CSV
`data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv`
(131,584 rows = 257×512 plane; regenerated with two-sided wall distance — trust this file).
A distribution-shift probe (`scripts/validation/hills_invariant_probe.py`) already
verified the invariants are well-behaved (only 17/120,562 fluid cells outside the duct
envelope) — your job is the trainer, not the data.

## File to create (ONLY this one)

`tbnn_train_hills.py` (repository root, mirroring the structure of `tbnn_train.py` —
READ that file first; reuse `TensorBasisNeuralNetwork` by importing it, do not copy it).

## Data pipeline (exact)

1. Read the CSV with pandas; reshape to planes (NY=257, NX=512) — rows are x-fastest
   within y (verify: x strictly increasing along axis 1 after reshape).
2. Fluid mask: `k > 1e-10` (excludes the ~11k zero-masked hill-interior points).
   Report the counts.
3. Velocity-gradient tensor (2D mean flow: ∂/∂z = 0):
   `dUdx, dUdy = grad_xy(U, x, y)` and same for V, importing `grad_xy` from
   `scripts/convert_hills_dns.py` (sys.path insert of the repo root, same pattern as
   `scripts/validation/hills_invariant_probe.py` — READ that file: it implements the
   exact conversion you need, verified end-to-end).
   Physical scaling: `SCALE = U_b / h` with **U_b = 2.9714, h = 0.028**;
   `grad_U[:, 0, 0] = dUdx*SCALE; [:,0,1] = dUdy*SCALE; [:,1,0] = dVdx*SCALE;
   [:,1,1] = dVdy*SCALE` (rows 3 and column 3 zero). `omega_phys = omega_nd * SCALE`.
4. Invariants and bases: `inv, T = comp_inv_and_bases(grad_U, omega_phys)` (import from
   `tbnn_tensor_math`). The 5 s timescale cap inside is fine — with the corrected
   surrogate the probe showed τ·S stays bounded (I1 max 0.606).
5. Targets from the DNS Reynolds stresses (normalized by U_b², already in the CSV):
   `a_11 = uu/(2k) - 1/3`, `a_22 = vv/(2k) - 1/3`, `a_33 = ww/(2k) - 1/3`,
   `a_12 = uv/(2k)`, `a_13 = uw/(2k)`, `a_23 = vw/(2k)` — computed ONLY on fluid points
   (k ≥ 1e-10 guarantees no division blowup; the CSV k column is exact by construction).

## Split (spatial, honest)

- **Validation bands:** x/h ∈ [2.0, 3.0] ∪ [6.0, 7.0] (one inside the bubble, one in the
  recovery region). **Train:** everything else. Report point counts for both.
- This tests spatial extrapolation — say so in the printed header.

## Training (identical hyperparameters to v1 — deliberate, isolates the dataset change)

- Seeds 42 (numpy + torch), Adam lr=0.002, MSELoss on the 3×3 anisotropy, 400 epochs,
  full batch, device auto (cuda if available).
- Print train/val loss every 50 epochs.

## Final evaluation (print)

- Validation-band loss (MSE) and mean Frobenius norm per point.
- Per-component R² on the validation bands for a_11, a_22, a_33, a_12 (define
  R² = 1 − SSE/SST against the DNS targets).
- Overall a priori R² on ALL fluid points.

## Outputs (distinct names — never touch v1 artifacts)

- `tbnn_weights_hills.pth` (state dict)
- `tbnn_model_hills.pt` (TorchScript trace — copy the dummy-input tracing block from
  `tbnn_train.py`)
- `verification_periodicHills/constant/tbnn_model.pt` (case copy)
- C++ JSON weights: READ `tbnn_export.py` first. If `export_cpp_weights` accepts an
  output path, use `export_weights/weights_hills.json`; otherwise write the JSON
  directly, mirroring its exact format (architecture 5→[64,64,64]→10, keys
  0.weight/2.weight/4.weight/6.weight + biases + inv_mean/inv_std if present).

## Constraints

- Dependencies: torch + numpy + pandas only (repo pins). No network. LF endings, no emoji.
- No modifications outside `tbnn_train_hills.py`.

## Acceptance criteria

1. `python tbnn_train_hills.py` runs end-to-end on CPU (~10 min for 400 epochs at 120k
   points full-batch is acceptable; print progress).
2. All four output files created; v1 artifacts untouched (verify by hash before/after).
3. Final printed table includes validation-band R² per component.
4. `pytest tests/ -q --ignore=release` still green (35 passed, 2 skipped).

## When done

Write `parallel-tasks/TASK-04-RESULT.md`: files created, the final metrics table,
v1-artifact hash confirmation, deviations (none expected).
