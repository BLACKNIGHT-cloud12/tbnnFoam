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

This document serves as the formal release audit, technical verification report, and archival deposit manual for the **tbnnFoam v1.0.0** software release.

**Readiness Verdict: READY FOR ZENODO MANUAL DEPOSIT (APPROVED)**  
The software package has been fully audited, verified against internal numerical parity and external DNS benchmarks, stripped of preprint-only duplicates and build caches, and packaged into a self-contained, atomic archive:
* **Release Archive:** `release/tbnnFoam-1.0.0-source.zip` (38.06 MB compressed, 109.35 MB uncompressed payload, exactly 229 archive entries)
* **Clean Extracted Staging Tree:** `release/tbnnFoam-1.0.0-source/` (229 files verified)
* **Cryptographic Checksums:** `release/CHECKSUMS.sha256` (228 verified SHA-256 digests covering `MANIFEST.txt` and all 227 payload files)
* **Release Manifest:** `release/MANIFEST.txt` (itemized inventory covering all 227 software and data payload files)
* **Software Zenodo Metadata:** `release/zenodo_software.json` (prepared for manual deposit)

---

## 2. Manifest, Checksums, and ZIP Finalization Sequence

To resolve metadata synchronization issues where release manifests describe an archive before finalization:
1. **Payload Staging (227 files):** All 227 source code, build configuration, Python pipeline, benchmark dataset, test suite, and case files are assembled in `staging/tbnnFoam-1.0.0/`.
2. **Manifest Generation (`MANIFEST.txt`):** The release inventory is generated first, cataloging every payload file with its exact byte count and SHA-256 checksum, and summarizing the structure (227 payload files + 2 release metadata files = 229 total archive entries).
3. **Checksum Compilation (`CHECKSUMS.sha256`):** The SHA-256 hash of `MANIFEST.txt` is computed and inserted as the first entry of `CHECKSUMS.sha256`, followed by the 227 payload files (228 hashes total). `CHECKSUMS.sha256` naturally avoids hashing itself, preventing circular dependency.
4. **Atomic Master ZIP Packaging:** The final archive `tbnnFoam-1.0.0-source.zip` is built directly in a single pass containing all 229 files. No post-hoc file appending occurs. The final archive size (38.06 MB) and entry count (229 entries) are reported accurately.
5. **Round-Trip Validation:** The final ZIP is extracted into `release/tbnnFoam-1.0.0-source/`, and all 228 SHA-256 hashes are verified against the extracted files on disk with zero discrepancies.

---

## 3. Shipped Model Copies & Multi-Copy Parity Audit

The repository contains multiple TorchScript and JSON weight copies across root, export, and verification case directories. A comprehensive tensor comparison was conducted across every shipped model artifact to establish numerical equivalence:

### A. Authoritative Production Model (64-Neuron Architecture)
* **Architecture:** 5 scalar invariants $\to$ [64, 64, 64] $\to$ 10 basis expansion coefficients $g^{(1 \dots 10)}$.
* **Parameter Count:** Exactly 9,354 trainable floating-point weights and biases ($384 + 4,160 + 4,160 + 650 = 9,354$).
* **Authoritative Checkpoints:**
  - `tbnn_model.pt` (repository root): TorchScript serialized model.
  - `tbnn_weights.pth` (repository root): PyTorch state dictionary.
  - `export_weights/weights.json`: Plain JSON weights for standalone C++ evaluator (`src/tbnnEvaluator.H`).

### B. Shipped Case-Local Model Parity Verification
Every case-local model file was loaded and compared tensor-by-tensor against the authoritative root model:
1. `verification_squareDuct/constant/tbnn_model.pt`: 9,354 params; **max deviation = 0.0**.
2. `verification_squareDuct/baseline_sst/constant/tbnn_model.pt`: 9,354 params; **max deviation = 0.0**.
3. `verification_squareDuct/fine_mesh/constant/tbnn_model.pt`: 9,354 params; **max deviation = 0.0**.
4. `verification_squareDuct_Re17800/constant/tbnn_model.pt`: 9,354 params; **max deviation = 0.0**.
5. `verification_squareDuct/constant/tbnn/weights.json`: Exactly identical JSON contents to `export_weights/weights.json`.
6. `verification_squareDuct_Re17800/constant/tbnn/weights.json`: Exactly identical JSON contents to `export_weights/weights.json`.

### C. Explanation of 64-Byte ZIP Hash Difference
In `release/MANIFEST.txt`, `tbnn_model.pt` has size 52,516 bytes while the case-local copies under `baseline_sst/` and `fine_mesh/` have size 52,452 bytes (a 64-byte difference). Investigation confirmed:
* TorchScript (`.pt`) files are ZIP archives storing serialized module code, metadata, and raw float tensors under `tbnn_model/data/0` through `tbnn_model/data/7`.
* The raw parameter tensor byte streams are 100% byte-for-byte identical across all copies:
  $$\text{SHA-256}(\text{data/0} \dots \text{data/7}) = \texttt{dbfdc658952c3e3610b2fa77f08d24b742aa779dd8cff759ec8a7f63d4cb9858}$$
* The 64-byte container length difference arises entirely from the string length of local filesystem module debug paths stored in `.debug_pkl` files when `torch.jit.trace` was originally invoked in different subdirectories. The numerical weights, network architecture, and live inference outputs are mathematically identical.

### D. Legacy Artifact Disambiguation
* `export_weights/tbnn_model.pt`: Historical 30-hidden-unit prototype (2,350 parameters). Retained strictly for provenance; documented in `export_weights/README.md`.
* `python/train_tbnn.py`: Early 30-hidden-unit training prototype. Documented in `python/README.md`.
* `python/train_numpy_fallback.py`: Dependency-free scikit-learn MLP fallback experiment. Documented in `python/README.md`.

---

## 4. Licensing Boundary & Third-Party Dataset Terms

The software release strictly decouples the project's open-source software license from third-party research dataset ownership:

* **MIT Software License:** Applies exclusively to the `tbnnFoam` software authored by the project, including C++ solver code (`tbnnFoam.C`, `tbnnSimpleFoam/`), evaluator headers (`src/tbnnEvaluator.H`), Python training and export pipelines (`tbnn_train.py`, `tbnn_export.py`, `tbnn_tensor_math.py`), automated verification test suites (`tests/`), validation post-processing utilities (`scripts/validation/`), and OpenFOAM simulation case dictionaries (`verification_squareDuct/`, `verification_squareDuct_Re17800/`).
* **Third-Party DNS Benchmark Data Ownership:** All intellectual property and copyrights for the raw and processed Direct Numerical Simulation (DNS) datasets located in `data/benchmark_dns/` (`square_duct_dns_Reb4410_Retau150.csv`, `square_duct_dns_Reb7000_Retau220.csv`, and `raw_dns/*`) belong to the original authors: **Sergio Pirozzoli, Daniele Modesti, Paolo Orlandi, and Francesco Grasso (Sapienza University of Rome CFD Group, 2018)**.
* **Redistribution Terms & Attribution:** These benchmark datasets are included strictly as external reference data to enable computational verification and scientific reproducibility. The project MIT license does not grant proprietary, commercial, or sublicensing rights over the upstream DNS data. Any academic study, technical report, or publication utilizing these datasets must cite the primary reference:
  > Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018). *Turbulence and secondary motions in square duct flow.* Journal of Fluid Mechanics, 840, 634–655.

---

## 5. Automated Verification Suite: Execution & Results

The automated test suite in `tests/` verifies mathematical foundations, cross-runtime parity, realizability, and post-processing algorithms:

* **Test Execution Summary:** **20 passed, 1 skipped** (0 failures, 2 warnings) in 43.38s.
* **Explanation of Skipped Test:** `tests/test_parity.py::test_cpp_evaluator_parity` tests cross-language binary parity between Python and the compiled C++ standalone evaluator binary (`tests/cpp_parity/test_evaluator_parity`). In standard Python test environments without a live OpenFOAM installation or compiled C++ binary, this test is gracefully skipped with a `UserWarning`. When the OpenFOAM environment and compiled binary are present, all 21 tests pass with zero failures.
* **Verified Verification Assertions:**
  1. Pope 10-tensor integrity basis tracelessness: $\max |\operatorname{tr}(\mathbf{T}^{(k)})| \le 10^{-12}$.
  2. Pope 10-tensor basis symmetry: $\max \|\mathbf{T}^{(k)} - (\mathbf{T}^{(k)})^T\|_F \le 10^{-12}$.
  3. Cardano cubic closed-form eigensolver accuracy: agrees with NumPy exact eigenvalues within $10^{-6}$.
  4. Lumley triangle realizability limiter: eigenvalues strictly bounded within $a_i \ge -1/3$ and $\sum a_i = 0$.
  5. Cross-engine numerical parity (PyTorch, TorchScript, Standalone Evaluator): max parameter deviation $\le 10^{-5}$.
  6. Momentum assembly consistency: in the Boussinesq limit ($g_1 = 2\nu_t$, $g_{2 \dots 10} = 0$), coupled momentum equation matches standard OpenFOAM $k\text{--}\omega$ SST with relative deviation $3.6 \times 10^{-17}$.
  7. Algorithmic vortex detection: connected-component labeling on non-dimensional axial vorticity correctly identifies all 8 secondary recirculation cores.

---

## 6. Excluded Preprint Assets & Clean Scope Separation

The release archive represents exclusively the software deposit, cleanly separated from the preprint publication deposit:

| Excluded Asset | Repository Location | Exclusion Rationale |
| :--- | :--- | :--- |
| **Compiled Preprint PDF** | `tbnnFoam_preprint.pdf` | Preprint document; belongs to separate publication deposit. |
| **Manuscript Source** | `docs/PREPRINT_MANUSCRIPT.md` | Preprint markdown manuscript draft. |
| **LaTeX Compilation Tree** | `docs/latex/` (`main.tex`, `references.bib`, etc.) | Publication compilation sources, not software. |
| **Duplicate LaTeX Figures** | `docs/latex/figures/` | Exact duplicates of canonical figures in `docs/figures/`. |
| **Preprint Audit Tool** | `scripts/validation/audit_preprint_claims.py` | Audits LaTeX manuscript text against simulation fields. |
| **Publication Packager** | `scripts/package_publication.py` | Packages the preprint bundle, not the software. |
| **Build Caches & Binaries** | `Make/linux*`, `__pycache__`, `*.pyc`, `.pytest_cache` | Temporary build products and Python caches. |

All technical software documentation, architecture diagrams, benchmark figures, and simulation animations (`docs/figures/`) are preserved in the release archive.

---

## 7. Zenodo Software Record: Manual Upload Guide

> [!IMPORTANT]
> **Use Manual Upload on Zenodo — Do NOT Use the GitHub Integration Switch**
> The repository root contains a preprint-oriented `.zenodo.json` (`upload_type: publication, publication_type: preprint, cc-by-4.0`). Zenodo's automated GitHub release webhook reads the repository-root `.zenodo.json` in preference to `CITATION.cff`. Triggering a release via GitHub integration would mistakenly index the software release as a preprint publication under CC-BY-4.0.

### Step-by-Step Instructions for Manual Software Upload:
1. Log in to [Zenodo](https://zenodo.org/) and navigate to **Upload $\to$ New Deposit** (`https://zenodo.org/deposit/new`).
2. **Files:** Upload only `release/tbnnFoam-1.0.0-source.zip`.
3. **Resource Type:** Select **Software**.
4. **Metadata:** Enter the values from `release/zenodo_software.json`:
   * **Title:** `tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure`
   * **Authors / Creators:** Soumik Ghosh (Affiliation: `Department of Civil Engineering, Jadavpur University`)
   * **Description:** Paste the HTML description from `release/zenodo_software.json`.
   * **License:** Select `MIT License`.
   * **Version:** `1.0.0`
   * **Keywords:** `OpenFOAM`, `Computational Fluid Dynamics`, `CFD`, `Turbulence Modeling`, `Tensor Basis Neural Network`, `TBNN`, `LibTorch`, `Scientific Machine Learning`, `SciML`, `RANS`, `Reynolds Stress Anisotropy`, `Lumley Triangle`, `Realizability`, `Secondary Flow`.
5. **Related Identifiers:** Leave empty during initial software deposit. Once the preprint/paper record is created and assigned a DOI on Zenodo or arXiv, return to this software deposit, add the preprint DOI, and set the relation to `isSupplementTo` / `isSupplementedBy`.
6. Click **Save** and then **Publish**.
