# tbnnFoam v2 — Separated-Flow Extension: Research Plan

**Status:** active (2026-11-01) · **Branch of record:** main · **Workflow:** ZCode = brain, Antigravity = hands (see `parallel-tasks/`)

## 1. Objective

Extend tbnnFoam from its validated domain (square-duct secondary flow of the second kind,
Re_τ = 150/220) to **turbulent separation with recirculation** — the single largest known
failure class of linear eddy-viscosity RANS models, and the entry ticket to aero/mech
CFD-ML research groups. Deliverable: **THE figure** — RANS-SST vs tbnnFoam vs DNS on a
separated case — plus a v2.0 software release.

## 2. Case decision: periodic hills, Re_H = 5600 — two-dataset architecture

Canonical separation benchmark: flow detaches at the hill crest, reattaches on the valley
floor, recirculates, recovers — all mechanisms linear closures cannot represent.

**Dataset roles (confirmed 2026-11-01 from the repository records themselves):**

- **VALIDATION (THE figure):** Krank, Kronbichler & Wall (TUM) DNS reference data — DOI
  `10.14459/2018mp1415670`, **CC BY 4.0**. Contains mean-velocity / Reynolds-shear-stress /
  TKE **profiles at 10 streamwise stations** plus **skin-friction and pressure coefficients
  at both walls** (Re_H = 5600, 10,595). Validation-grade only — NOT full-plane fields.
- **TRAINING (a priori):** Voet & Laizet full-field DNS — Mendeley Data DOI
  `10.17632/by72n3s9bn.2`: **7 DNS (+ 30 RANS)** of flow over periodically arranged hills
  with geometry parameter variation (steepness angle, streamwise separation). The
  canonical-geometry DNS is the v2 training case; the parameter variation is a future
  generalization study (explicitly out of v2 scope).
- **Why not the TMR 2D bump:** attached-flow *verification* case (grid convergence), not
  separated validation — ruled out 2026-11-01.
- Provenance, licenses, and download commands: `data/benchmark_dns/periodic_hills/README.md`.

## 3. What stays fixed (the point of v2)

- Solver architecture: `tbnnFoam`/`tbnnSimpleFoam`, in-situ momentum coupling, Cardano
  realizability limiter — unchanged.
- Network: 5 invariants → [64, 64, 64] → 10 basis coefficients, Pope's basis — unchanged.
- The v1 lesson learned: a priori scores do not guarantee a posteriori success; the figure is
  an a posteriori result or it does not exist.

## 4. What changes (training pipeline adaptation)

1. **Data plane:** streamwise x–y planes (hills) instead of the y–z cross-section (duct);
   no symmetry mirroring; wall-normal finite-difference gradients on the DNS grid with the
   **curved wall contour** y_wall(x) (duct used flat walls).
2. **Wall distance:** d_w to the hill/valley contour for the ω surrogate
   (same SST relation: ω ≈ √k / (C_μ^{1/4} ℓ), ℓ = min(κ d_w, 0.07 D_h)).
3. **Envelope machinery reused:** 1D marginal invariant bounding + 5D k-NN distribution-shift
   diagnostic (`evaluate_timescale_distribution.py`) re-run for the hills training/live pair.
4. **Case:** new `verification_periodicHills/` OpenFOAM case (streamwise-periodic, hill
   contour mesh, Re_H = 5600 matching the DNS).

## 5. Phases

| Phase | Work | Output | Lane |
| :--- | :--- | :--- | :--- |
| 0 | Metrics module (sep./reattachment/bubble) — `TASK-01` | `scripts/validation/separated_flow_metrics.py` + tests | Antigravity |
| 0 | DNS ingest converter — `TASK-02` | `scripts/convert_hills_dns.py` + tests (parser format confirmed when data lands) | Antigravity |
| 1 | Download both datasets (TUM validation profiles/Cf/Cp + Voet/Laizet full-field training DNS); adapt converter parser to actual raw formats; envelope probe of duct-trained weights on hill invariants | `data/benchmark_dns/periodic_hills/` + shift report | me + user download |
| 2 | Training-pipeline adaptation (`tbnn_train.py` → streamwise-plane loader, contour wall distance) | v2 training entry point | Antigravity, after Phase 1 |
| 3 | Retrain on hills DNS; a priori evaluation (held-out x-planes) | v2 weights + parity export | me |
| 4 | `verification_periodicHills/` case (hill contour, periodic streamwise); SST baseline run; tbnnFoam run | converged fields | me (case), WSL compute |
| 5 | THE figure: Cp / iso-velocity / τ_w / reattachment-length comparison; v2.0 release | figure + release | me |

## 6. Risks (named, watched)

- **A priori → a posteriori gap** (v1's central lesson): mitigated by realizing that hills
  reattachment length is highly sensitive; report both a priori and a posteriori honestly.
- **Curved-wall gradient quality** from FD on the DNS grid: mitigate with contour-aware
  stencil; compare u_τ against DNS-published Cf if provided.
- **Solver stability under strong reverse flow:** τ limiter and realizability limiter already
  in place; monitor live-state distribution shift (tooling exists from v1).

## 7. Guardrails

- WISE host emails and Prof. Sarkar conversation proceed in parallel — v2 never blocks the
  campaign (v1.0.1 + DOI is already the email asset).
- All DNS-derived data stays out of the repo tree (v1 licensing posture).
