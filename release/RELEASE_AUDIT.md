# tbnnFoam Software Release Audit & Archival Report (v1.0.0)

**Software Name:** tbnnFoam (Tensor Basis Neural Network Turbulence Closure for OpenFOAM)  
**Version:** 1.0.0  
**Author / Principal Investigator:** Soumik Ghosh  
**Authorized Institution:** Department of Civil Engineering, Jadavpur University  
**Contact Email:** soumikghosh30.03.07@gmail.com  
**Release Date:** October 2026  
**Primary Archive:** `tbnnFoam-1.0.0-source.zip` (SHA-256 Checksummed)  
**Deposit Type:** Scientific Software Release (Zenodo `upload_type: software`)  

---

## 1. Executive Summary & Readiness Verdict

This document serves as the formal release audit and verification report for the **tbnnFoam v1.0.0** software deposit on Zenodo. 

**Readiness Verdict: READY FOR ZENODO DEPOSIT (APPROVED)**  
The software package has been fully audited, reorganized into an archival-grade structure, verified against internal numerical parity and external DNS benchmarks, stripped of publication-only duplicates and build caches, and packaged into a self-contained archive:
* **Release Archive:** `release/tbnnFoam-1.0.0-source.zip` (38.04 MB compressed, 109.35 MB uncompressed, 227 files)
* **Clean Staging Tree:** `release/tbnnFoam-1.0.0-source/`
* **File Checksums:** `release/CHECKSUMS.sha256` (227 verified SHA-256 digests)
* **Release Manifest:** `release/MANIFEST.txt`

---

## 2. Included Software Components

The archive contains the complete, reproducible computational workflow across seven foundational pillars:

### A. Core OpenFOAM C++ Solvers & Evaluators
1. **`tbnnFoam.C` & `createFields.H`**: The primary in-situ coupled RANS solver targeting OpenFOAM (v2012–v2606), featuring an embedded LibTorch C++ inference engine (`torch::jit::load`), runtime invariant calculations, and analytical Cardano cubic realizability limiter.
2. **`src/tbnnEvaluator.H`**: A zero-dependency, header-only C++ matrix evaluator capable of parsing standard JSON neural weights (`weights.json`) and evaluating Pope's (1975) ten-term integrity basis without requiring external deep-learning runtime libraries.
3. **`src/tbnnSimpleFoam/`**: A standalone steady-state SIMPLE solver utilizing the native C++ matrix evaluator, demonstrating seamless portability to supercomputing clusters without LibTorch dependencies.
4. **`Make/files` & `Make/options`**: Production build configurations linking OpenFOAM finite volume libraries and optional LibTorch C++ headers.

### B. Machine Learning Architecture, Training & Export Pipeline
1. **`tbnn_train.py`**: The authoritative PyTorch training pipeline. Implements the 9,354-parameter Multi-Layer Perceptron (5 invariants $\to$ [64, 64, 64] $\to$ 10 basis coefficients $g^{(1 \dots 10)}$), fixed random seeds (42), and MSE loss on the full 9-component Reynolds stress anisotropy tensor $\mathbf{a}$.
2. **`tbnn_export.py`**: Model serialization pipeline generating both TorchScript JIT models (`tbnn_model.pt`) and plain-JSON weights (`weights.json`) for the C++ evaluator.
3. **`tbnn_tensor_math.py`**: Mathematical module providing Pope's complete 10-tensor integrity basis $\mathbf{T}^{(1 \dots 10)}$, Cayley–Hamilton scalar invariants $I_{1 \dots 5}$, closed-form Cardano cubic eigensolver, and projection onto the Lumley realizability triangle.
4. **Trained Production Checkpoints**:
   * `tbnn_model.pt`: Serialized TorchScript model (9,354 parameters, SHA-256: `4318252E9107C1B0623C85905992872EB48BA82CA6C089F73E890FAD6EB712E6`).
   * `tbnn_weights.pth`: PyTorch model state dict (9,354 parameters, SHA-256: `C75C5FC25FDDEE1E8265FFE2AB29808FA78E5DB6C9ED2AF4CFE1B4D4562D2F49`).
   * `export_weights/weights.json`: C++ plain-JSON weights (9,354 parameters, SHA-256: `F49EFC078E661B1FB86397182B6973D012209E1B479CD1FB071CF12E2F8E8C17`).

### C. Benchmark DNS Datasets & Metadata
1. **`data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv`**: Canonical training and primary validation dataset ($Re_b = 4,410$, $Re_\tau = 150$, 4,225 grid points).
2. **`data/benchmark_dns/square_duct_dns_Reb7000_Retau220.csv`**: Generalization benchmark dataset ($Re_b = 7,000$, $Re_\tau = 220$, 4,225 grid points).
3. **`data/benchmark_dns/raw_dns/`**: Raw exported ASCII profiles (`plotyz_Retau150.dat`, `plotyz_Retau220.dat`) from the Sapienza University CFD Group database.
4. **`data/benchmark_dns/benchmark_metadata.json`**: Rigorous documentation of physical dimensions, Reynolds numbers ($Re_b$, $Re_h$, $Re_\tau$), boundary conditions, and scaling conventions.
5. **`data/benchmark_dns/README.md`**: Provenance, citation guidelines, and academic fair-use redistribution notice.

### D. Automated 21-Test Verification Suite
* **`tests/test_parity.py`**: Verifies numerical parity between PyTorch, TorchScript, and standalone C++ evaluators ($\mathrm{atol} \le 10^{-5}$), and basis symmetry/tracelessness ($\mathrm{atol} \le 10^{-12}$).
* **`tests/test_mesh_reconstruction_verification.py`**: Verifies structured cell indexing and coordinate mapping against OpenFOAM `0/C` field outputs.
* **`tests/test_mesh_sensitivity.py`**: Verifies quantitative error trends and topological core stability under cross-stream grid refinement ($20 \times 20$ vs. $40 \times 40$).
* **`tests/test_timescale_distribution.py`**: Verifies 1D invariant bounding and calculates joint 5D standardized $k$-NN distribution shift.
* **`tests/test_coordinate_convention.py`**: Enforces consistent coordinate and vorticity sign conventions across all evaluation routines.
* **`tests/test_vortex_detection.py`**: Verifies connected-component labeling and 8-vortex core identification.
* **`tests/cpp_parity/`**: C++ unit test source code for standalone matrix evaluator and momentum-equation assembled matrix consistency.

### E. Reproducibility Simulation Cases
1. **`verification_squareDuct/`**: Primary developing square duct flow ($Re_b = 4,410$, $50 \times 20 \times 20$ grid). Includes full `0/`, `constant/`, `system/` configurations, `Allrun` execution script, automated plotting routines, baseline linear $k\text{--}\omega$ SST case (`baseline_sst/`), fine-mesh sensitivity case (`fine_mesh/`), and converged solution fields at $t=300$.
2. **`verification_squareDuct_Re17800/`**: High-Reynolds-number case study ($Re_b = 17,800$, $Re_\tau \approx 519$, $50 \times 40 \times 40$ grid, 80,000 cells). Includes full case dictionaries, `Allrun`, and converged solution fields at iteration 294.

### F. Validation & Post-Processing Utilities
* **`scripts/validation/vortex_detector.py`**: Algorithmic connected-component axial vorticity detector with boundary-layer margin masking and area thresholding.
* **`scripts/validation/foam_field_io.py`**: Native binary and ASCII OpenFOAM field parser for Python.
* **`scripts/validation/evaluate_mesh_sensitivity.py`**: Discretization sensitivity metrics calculator.
* **`scripts/validation/evaluate_timescale_distribution.py`**: 5D $k$-NN distance and invariant envelope calculator.
* **`scripts/convert_real_dns.py`**: Raw DNS `.dat` to structured CSV converter.
* **`scripts/generate_all_showcase_assets.py` & `scripts/generate_showcase_animations.py`**: Computational figure and animation generation pipelines.

### G. Visual Assets & Documentation
* **`docs/figures/`**: Canonical high-resolution figures (`architecture_pipeline.png`, `tbnn_network_architecture.png`, `barycentric_anisotropy_map.png`, `reynolds_stress_mechanism.png`, `tbnn_hero_showcase.png`, `mesh_sensitivity_study.png`, `tbnn_Re17800_showcase.png`, `timescale_and_invariant_distribution.png`) and simulation animations (`animation_1_secondary_vortices.gif`, `animation_2_3d_streamlines.gif`, `animation_4_solver_convergence.gif`).
* **`README.md`**: Complete, software-oriented release documentation.
* **`LICENSE`**: MIT open-source license.
* **`CITATION.cff`**: Software citation metadata conforming to CFF 1.2.0.
* **`.zenodo.json`**: Software deposit metadata with proper creators, affiliations, keywords, and release version.

---

## 3. Excluded Items & Rationale

| Excluded Item | Location in Repository | Rationale for Exclusion |
| :--- | :--- | :--- |
| **Compiled Preprint PDF** | `docs/tbnnFoam_preprint.pdf` | Belongs exclusively to the separate publication/preprint deposit; should not bloat software archive. |
| **Preprint Manuscript Source** | `docs/PREPRINT_MANUSCRIPT.md` | Research paper manuscript draft, separate from software source code. |
| **LaTeX Compilation Source Tree** | `docs/latex/` (`main.tex`, `references.bib`, `*.bbl`, etc.) | Preprint document source, not part of software build or runtime. |
| **Duplicate LaTeX Figures** | `docs/latex/figures/` | Exact duplicate copies of the figures stored canonically in `docs/figures/`. |
| **Interim Preprint Audit Script** | `scripts/validation/audit_preprint_claims.py` | Audits LaTeX text claims against simulation data; preprint-specific. |
| **Environment-Specific Script** | `build_and_audit.sh` | Contains hardcoded local filesystem paths (`/mnt/c/Users/soumi/...`) and executes LaTeX compiler. |
| **Publication Package Scripts** | `scripts/package_publication.py` | Packaging script for the separate preprint archive. |
| **Duplicate PNGs in Cases** | `verification_squareDuct/*.png`, `verification_squareDuct_Re17800/*.png` | Exact copies of figures preserved in `docs/figures/`. |
| **Compiled C++ Binaries & Objects** | `Make/linux*`, `test_evaluator_parity`, `test_momentum_parity` | Architecture-specific Linux ELF binaries; must be built locally with `wmake`. |
| **Python Bytecode & Caches** | `__pycache__/`, `*.pyc`, `.pytest_cache/` | Ephemeral compiler caches. |

---

## 4. Reorganizations & Fixes Applied in the Release Staging

1. **Software-Specific Zenodo Metadata (`.zenodo.json`)**:
   - Switched `upload_type` from `"publication"` (with `publication_type: "preprint"`) back to `"software"`.
   - Updated record title to software title: `"tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure"`.
   - Set author affiliation to `"Department of Civil Engineering, Jadavpur University"`.
   - Set software license to `"MIT"`.
2. **Fixed Hardcoded Local Path in `run_baseline.sh`**:
   - Replaced `ROOT_DIR="/mnt/c/Users/soumi/tbnnFoam/verification_squareDuct"` with dynamic detection:
     ```bash
     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
     ROOT_DIR="${SCRIPT_DIR}"
     ```
   - Enables executing the baseline simulation on any POSIX/Linux filesystem.
3. **Model Artifact Disambiguation**:
   - Added `export_weights/README.md` clarifying that `export_weights/weights.json` is the active 64-width model (9,354 parameters) exactly matching `tbnn_model.pt`, and documenting that `export_weights/tbnn_model.pt` is a legacy 30-width prototype retained for provenance.
   - Added `python/README.md` explicitly distinguishing legacy experiments (`train_tbnn.py`, `train_numpy_fallback.py`) from the production pipeline (`tbnn_train.py`).
4. **Third-Party Data Licensing Clarification**:
   - Added `data/benchmark_dns/README.md` explicitly documenting the academic research data provenance from Pirozzoli et al. (2018) and Vinuesa et al. (2014), clarifying that the MIT software license applies to code/parsers, while the benchmark DNS datasets require scholarly citation.

---

## 5. Verification Checks & Actual Outcomes

| Verification Step | Command / Method | Observed Outcome | Status |
| :--- | :--- | :--- | :---: |
| **PyTorch TorchScript Model** | `torch.jit.load('tbnn_model.pt')` | 9,354 parameters; layer widths [64, 64, 64], 10 outputs. | **PASS** |
| **PyTorch State Dict** | `torch.load('tbnn_weights.pth')` | 9,354 parameters; exactly matches TorchScript weights (diff = 0.0). | **PASS** |
| **C++ JSON Weights** | `json.load('export_weights/weights.json')` | [64x5, 64x64, 64x64, 10x64]; exactly matches PyTorch weights (diff = 0.0). | **PASS** |
| **Python Test Suite (Staging)** | `pytest tests/` (executed inside staging) | 20 passed, 1 skipped (OpenFOAM C++ binary test requires live WSL build). | **PASS** |
| **Basis Tracelessness & Symmetry** | `test_basis_trace`, `test_basis_symmetry` | Analytical tolerance $\le 10^{-12}$. | **PASS** |
| **Realizability Enforcement** | `test_realizability_bounds` | Lumley triangle bounds verified on $N=50$ perturbation samples. | **PASS** |
| **Momentum Assembly Parity** | `test_momentum_coupling_parity` | Boussinesq limit reduces to baseline model within $3.6 \times 10^{-17}$. | **PASS** |
| **Algorithmic Vortex Detection** | `test_vortex_detection` | Exactly 8 counter-rotating secondary flow cores identified. | **PASS** |
| **Archive Integrity** | ZIP round-trip test (extract & verify) | 227 files extracted cleanly; zero checksum discrepancies. | **PASS** |

---

## 6. Known Reproducibility Considerations & Scope Boundaries

1. **Secondary Flow Magnitude Discretization Sensitivity**:
   As documented in the verification studies, peak secondary circulation magnitude is strongly coupled to cross-stream grid refinement, decreasing from $7.99\%$ ($20 \times 20$ coarse grid) to $2.62\%$ ($40 \times 40$ fine grid) toward the raw-field DNS benchmark of $1.93\%$. Users should use the refined grid when quantitative secondary velocity accuracy is required.
2. **Developing Duct Flow vs. Streamwise-Periodic DNS**:
   The included OpenFOAM verification cases model a spatially developing laboratory duct ($L = 20h = 1.0\text{ m}$) with entrance boundary layers, whereas the reference DNS of Pirozzoli et al. (2018) models an asymptotically fully developed flow in a streamwise-periodic domain. The reported $9.34\%$ streamwise velocity discrepancy combines model-form approximation and entrance development length.
3. **Hardware Determinism**:
   The neural network training pipeline (`tbnn_train.py`) uses fixed random seeds (42) and is bit-deterministic on CPU under the pinned dependencies in `requirements.txt`. Bit-level reproducibility across different CUDA hardware or PyTorch major versions cannot be guaranteed due to non-associative floating-point operations.

---

## 7. Instructions for Publishing on Zenodo

1. **Connect GitHub to Zenodo**: Sign in to [zenodo.org](https://zenodo.org/) with GitHub account `BLACKNIGHT-cloud12`, navigate to settings, and enable the switch for `tbnnFoam`.
2. **Alternative Direct Upload**: Upload `tbnnFoam-1.0.0-source.zip` directly to Zenodo via New Deposit, selecting **Software** as the resource type. Zenodo will automatically populate metadata from `.zenodo.json` and `CITATION.cff`.
3. **Cross-Referencing**: Link the software release to the forthcoming preprint/paper deposit using Zenodo's `isSupplementTo` relationship.
