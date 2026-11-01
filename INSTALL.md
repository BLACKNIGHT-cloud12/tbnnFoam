# tbnnFoam — Installation & Usage Guide

Step-by-step instructions for building, running, validating, and (optionally) retraining
**tbnnFoam** — an in-situ Tensor Basis Neural Network (TBNN) turbulence closure for OpenFOAM.

For the physics, validation metrics, and honest limitations discussion, see [`README.md`](README.md).
For build provenance and archive integrity, see `release/RELEASE_AUDIT.md` in the software deposit.

---

## 1. Prerequisites

| Component | Requirement | Notes |
| :--- | :--- | :--- |
| OpenFOAM | **v2606 (tested)**; v2012+ expected compatible | [openfoam.com](https://www.openfoam.com) release; provides `wmake`, `blockMesh`, `postProcess` |
| C++ toolchain | The one bundled with your OpenFOAM installation | Third-party solver is C++11-compatible; LibTorch engine uses C++17 |
| Python | **3.12 (tested)** | Only needed for tests, training, validation, and post-processing — **not** for running the CFD cases |
| Python packages | Pinned in `requirements.txt` | `torch==2.14.1`, `numpy==2.5.3`, `scipy==1.18.1`, etc. — pins reproduce the shipped weights bit-for-bit |
| LibTorch | **Optional** | ONLY for the `tbnnFoam` (LibTorch) engine. The recommended `tbnnSimpleFoam` solver has zero third-party dependencies |
| Hardware | Modest | Verification cases are 20k–80k cell steady RANS; a laptop is sufficient |

> **Windows users:** use WSL2 with an Ubuntu distribution and install OpenFOAM v2606 inside WSL
> (follow the openfoam.com Linux installation instructions). All shell commands below run inside WSL.

---

## 2. Get the Code

Either extract the Zenodo release archive:

```bash
unzip tbnnFoam-1.0.1-source.zip          # creates the tbnnFoam-1.0.1/ tree
cd tbnnFoam-1.0.1
```

or clone the repository:

```bash
git clone https://github.com/BLACKNIGHT-cloud12/tbnnFoam.git
cd tbnnFoam
```

**Verify archive integrity** (from the archive root, after extracting):

```bash
sha256sum -c CHECKSUMS.sha256            # must report every line OK
```

The outer zip's own digest is in the accompanying `tbnnFoam-1.0.1-source.zip.sha256` sidecar.

---

## 3. Python Environment

```bash
python3 -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

On machines where you only need CPU inference/testing, the CPU-only PyTorch wheel avoids
large CUDA downloads:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The pins in `requirements.txt` are intentional: they reproduce the shipped model weights and
all committed validation metrics. Changing a pin invalidates bit-level training reproducibility.

---

## 4. Build the Solver

Activate the OpenFOAM environment first (every new shell):

```bash
source /opt/openfoam/OpenFOAM-v2606/etc/bashrc    # adjust path to your installation
```

### Path A (recommended): `tbnnSimpleFoam` — zero third-party dependencies

The standalone solver embeds a header-only C++ evaluator (`src/tbnnEvaluator.H`) that parses
model weights from a plain JSON file. No LibTorch, no CUDA, no ABI headaches — compiles
anywhere `wmake` works, including shared HPC clusters.

```bash
cd src/tbnnSimpleFoam
wmake                       # installs tbnnSimpleFoam into $FOAM_USER_APPBIN
cd ../..
```

### Path B (optional): `tbnnFoam` — LibTorch / PyTorch C++ engine

Useful for research prototyping with TorchScript models. Requires LibTorch (CPU build is
sufficient — the shipped model runs on CPU):

```bash
# 1. Download LibTorch (CPU, C++11 ABI variant) from https://pytorch.org/get-started/locally/
# 2. Unpack it to ~/libtorch (the default expected by Make/options), or edit
#    LIBTORCH_DIR in Make/options to point at your installation.
# 3. Build from the repository root:
wmake                       # installs tbnnFoam into $FOAM_USER_APPBIN
```

Note: `Make/options` sets `-D_GLIBCXX_USE_CXX11_ABI=1`; use the matching (cxx11-ABI) LibTorch
build for your GCC. If linking LibTorch fails on your cluster (mismatched CUDA/GCC/ABI — the
usual culprits), simply use Path A; both engines are numerically identical (verified by
`tests/test_parity.py` to parameter deviation 0.0 and momentum-assembly consistency to
3.6 × 10⁻¹⁷ relative in the Boussinesq limit).

---

## 5. Run the Verification Case (square duct, Re_b = 4410, Re_τ = 150)

```bash
cd verification_squareDuct
./Allrun
```

`Allrun` performs: mesh generation (`blockMesh`, 20,000 cells) → copies the pre-trained
TorchScript model (`tbnn_model.pt`) from the repository root into `constant/` → runs the
solver to steady state. The committed runs satisfy `residualControl` in 287–294 SIMPLE
iterations (`< 10⁻⁴` initial residuals).

> **Solver choice:** `Allrun` invokes `tbnnFoam` (the LibTorch engine) by default. If you built
> only `tbnnSimpleFoam` (Path A), replace the run line in `Allrun` with
> `tbnnSimpleFoam > log.tbnnSimpleFoam 2>&1` — the case ships both model formats
> (`constant/tbnn_model.pt` for LibTorch, `constant/tbnn/weights.json` for the standalone evaluator).

A converged reference solution snapshot is committed in `verification_squareDuct/300/`, and
baseline/comparison snapshots in `verification_squareDuct/baseline_sst/` (linear k–ω SST) and
`verification_squareDuct/fine_mesh/` (4× cross-stream refined, 80,000 cells).

---

## 6. Post-Process and Validate

All validation scripts run from the **repository/archive root** (they resolve case paths
internally) and use the cell-centre field `verification_squareDuct/0/C`, which ships with the
release. If you re-mesh, regenerate it first:

```bash
cd verification_squareDuct
blockMesh
postProcess -func writeCellCentres
cd ..
```

### Without DNS data (no downloads needed)

```bash
python scripts/validation/evaluate_validation_plan.py    # 8-vortex topology, core locations, peak speeds
python scripts/validation/evaluate_mesh_sensitivity.py   # coarse-vs-fine grid sensitivity trends
```

These evaluate the secondary-flow topology against the stored benchmark metadata and print the
quantitative comparison. The expected outcome: **8 secondary recirculation cores** recovered
(which linear k–ω SST cannot produce), Q1 core near the DNS reference (0.577, 0.847).

### With DNS data (for RMSEs and contour overlays)

The third-party DNS benchmark is **not redistributed** (licensing boundary — see
[`data/benchmark_dns/README.md`](data/benchmark_dns/README.md)). To obtain it:

1. Download `plotyz_Retau150.dat` and `plotyz_Retau220.dat` from the Sapienza University of Rome
   CFD Group database: [http://newton.dma.uniroma1.it/square_duct/](http://newton.dma.uniroma1.it/square_duct/)
2. Place them in `data/benchmark_dns/raw_dns/`
3. Convert: `python scripts/convert_real_dns.py`  → produces the standardized CSVs
4. Then:

```bash
python verification_squareDuct/validate_and_plot.py      # 4-panel DNS-overlaid figures
python scripts/validation/evaluate_validation_plan.py    # adds full-field U, (V,W) RMSEs
python scripts/validation/evaluate_timescale_distribution.py  # 5D invariant distribution-shift diagnostic
```

Any academic use of the DNS datasets must cite Pirozzoli et al. (*J. Fluid Mech.*, 2018).

---

## 7. High-Reynolds-Number Case (Re_b = 17,800, Re_τ ≈ 519)

```bash
cd verification_squareDuct_Re17800
./Allrun                                  # 80,000 cells; converged in 294 iterations
cd ..
python verification_squareDuct_Re17800/validate_and_plot_Re17800.py
```

Demonstrates scalability to the asymptotic DNS benchmark regime; downstream stations
(x/D_h ≥ 6.1) recover the 8-vortex topology with core-distance errors of 0.028–0.059 h.

---

## 8. Run the Test Suite

```bash
pytest tests/ -v
```

Expected results depend on the environment:

| Environment | Result |
| :--- | :--- |
| Plain Python (no OpenFOAM, no DNS — e.g. CI) | 18 passed, 3 skipped |
| + OpenFOAM toolchain (C++ parity test compiles via `tests/cpp_parity`) | 19 passed, 2 skipped |
| + external DNS data supplied | 21 passed, 0 skipped |

The compiled-binary test (`test_standalone_cpp_numerical_parity`) self-skips when the OpenFOAM
toolchain is unavailable; the 2 DNS-dependent tests self-skip without external DNS data. Since
the v1.0.1 verification-script correction, `test_momentum_coupling_parity` runs in pure Python
and no longer requires the toolchain. All skips are clean (no failures).

---

## 9. Retrain the Model (optional)

The shipped production model (5 → [64, 64, 64] → 10; 9,354 parameters) is bundled, so this is
only needed to reproduce training or experiment:

```bash
# DNS data required (Section 6, steps 1–3), then either:
python tbnn_train.py --dns-dir /path/to/directory_with_csv
# or:
export TBNN_DNS_DIR="/path/to/directory_with_csv"
python tbnn_train.py
python tbnn_export.py      # re-exports tbnn_model.pt + export_weights/weights.json
```

Training uses fixed seeds and the pinned package versions for bit-level reproducibility.
`python/` contains legacy prototype trainers (documented in `python/README.md`); the
authoritative pipeline is the root-level `tbnn_train.py` / `tbnn_export.py` / `tbnn_tensor_math.py`.

---

## 10. What Success Looks Like

* `log.tbnnFoam` (or `log.tbnnSimpleFoam`) ends with residuals meeting `residualControl` (~287–294 iterations).
* `evaluate_validation_plan.py` reports **8 secondary-flow vortex cores** (the linear baseline detects 0).
* Q1 anti-clockwise core detected near (0.577, 0.847) in h-units (exact values in the README tables).
* All checksum and parity tests pass (Section 8).

## 11. Operating Envelope & Honest Limitations

Read [`README.md`](README.md) §"Honest Technical Discussion" in full before using the model
beyond the verification cases. Key points:

* Trained on square-duct DNS at Re_τ = 150 with an empirical ω surrogate — **not** a general
  turbulence model. Retrain and revalidate before applying to other geometries/Reynolds numbers.
* The validation cases are spatially developing ducts vs. a streamwise-periodic DNS reference;
  reported RMSEs combine model-form and discretization error (first-order upwinding, uniform
  cross-stream mesh, y⁺ ≈ 7.5–13 under wall functions).
* Live-coupling distribution shift is quantified (median standardized k-NN distance 0.21σ);
  scripts are included to re-measure it on your own cases.

## 12. Citation & License

* Cite: [`CITATION.cff`](CITATION.cff) — Zenodo **concept DOI** `10.5281/zenodo.23799523`
  (resolves to the latest release; the version-specific DOI for v1.0.1 is assigned at publish time).
* Software: MIT License ([`LICENSE`](LICENSE)). Third-party DNS data remains under its
  upstream terms and must be obtained separately (Section 6).
