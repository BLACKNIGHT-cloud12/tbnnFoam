# tbnnFoam Software Release Audit & Archival Report (v1.0.1)

**Software Name:** tbnnFoam (Tensor Basis Neural Network Turbulence Closure for OpenFOAM)  
**Version:** 1.0.1  
**Author / Principal Investigator:** Soumik Ghosh  
**Authorized Institution:** Department of Civil Engineering, Jadavpur University  
**Contact Email:** soumikghosh30.03.07@gmail.com  
**Release Date:** 2026-11-01  
**Primary Archive:** `release/tbnnFoam-1.0.1-source.zip` (SHA-256 Checksummed)  
**Deposit Type:** Scientific Software Release (Zenodo `upload_type: software`, published as a new version of record DOI `10.5281/zenodo.23799524`)  

---

## 1. Executive Summary & Readiness Verdict

This document is the formal release audit, technical verification report, and archival deposit
manual for the **tbnnFoam v1.0.1** software release. It supersedes the v1.0.0 audit report.

**Readiness Verdict: READY FOR ZENODO MANUAL DEPOSIT (APPROVED)**  
The software package has been fully audited, re-verified against internal numerical parity and
external DNS benchmarks, stripped of preprint-only duplicates and build caches, and packaged
into a self-contained, single-pass release archive:

* **Release Archive:** `release/tbnnFoam-1.0.1-source.zip` (36.46 MB compressed, 104.35 MB uncompressed payload, exactly 227 archive entries)
* **External Archive Digest:** `release/tbnnFoam-1.0.1-source.zip.sha256` — `638c2593570c9aa084e45a48336518376730dcb56d0ed63a0b4b80d259089e1a`
* **Clean Extracted Staging Tree:** `release/tbnnFoam-1.0.1-source/` (227 files verified)
* **Cryptographic Checksums:** `release/CHECKSUMS.sha256` (226 verified SHA-256 digests covering `MANIFEST.txt` and all 225 payload files) — **LF line endings**
* **Release Manifest:** `release/MANIFEST.txt` (itemized inventory of all 225 software, benchmark metadata, documentation, and case payload files; header corrected to v1.0.1)
* **Software Zenodo Metadata:** `release/zenodo_software.json` (version 1.0.1, prepared for manual deposit)
* **Installation & Usage Guide:** `INSTALL.md` (new in v1.0.1 — ships inside the archive)

---

## 2. What Changed in v1.0.1 (Relative to the Published v1.0.0 Record)

### A. Scientific/metadata changes (already present in the first v1.0.1 build)
1. **Benchmark metadata reconciliation:** `data/benchmark_dns/benchmark_metadata.json` and
   `data/benchmark_dns/README.md` updated; the raw-field-verified DNS reference (Q1 ACW core
   at (0.577, 0.847); peak secondary speed 1.93% of U_b) is authoritative.
2. **Corrected verification scripts:** `scripts/validation/evaluate_mesh_sensitivity.py`,
   `evaluate_validation_plan.py`, and `verify_dns_reference.py` updated to match the reconciled
   references. As part of this correction, `test_momentum_coupling_parity` now runs in pure
   Python (no OpenFOAM binary required).
3. Version files: `CITATION.cff`, `.zenodo.json`, `README.md`, one regenerated figure
   (`docs/figures/tbnn_Re17800_showcase.png`). Payload path set otherwise identical to v1.0.0.

### B. Packaging-engineering changes (this rebuild, 2026-11-01)
1. **`INSTALL.md` added** (payload file 225 of 225): complete install/usage guide —
   prerequisites, both solver build paths (zero-dependency `tbnnSimpleFoam` and LibTorch
   `tbnnFoam`), case execution, validation workflow with and without DNS, retraining,
   troubleshooting, operating envelope, citation.
2. **Checksum files normalized to LF:** the v1.0.0-era `CHECKSUMS.sha256` and
   `tbnnFoam-1.0.1-source.zip.sha256` were written with CRLF line endings, which breaks
   `sha256sum -c` for Linux/macOS users. The packager (`scripts/prepare_software_release.py`)
   now writes all generated text artifacts with `newline="\n"`.
3. **`MANIFEST.txt` header corrected** ("v1.0.0" → "v1.0.1"; audited date → 2026-11-01) and
   payload paths normalized to forward slashes for cross-platform readability.
4. **`README.md` updated:** Quick Start links to `INSTALL.md`; test-suite counts corrected to
   the post-correction matrix (see §6); CI badge updated to 18 passed / 3 skipped.
5. **Packager hardened:** hardcoded file-count assertions raised to 225 payload / 227 archive
   entries; version and date strings centralized in the manifest header.

---

## 3. Manifest, Checksums, and ZIP Finalization Sequence

1. **Payload Staging (225 files):** all source, build configuration, Python pipeline, benchmark
   metadata, synthetic test fixture, test suite, documentation, and case files assembled in
   `release/staging/tbnnFoam-1.0.1/`.
2. **Manifest Generation (`MANIFEST.txt`):** generated first, cataloging every payload file
   with byte count and SHA-256 (225 payload files + 2 release metadata files = 227 entries).
3. **Checksum Compilation (`CHECKSUMS.sha256`):** the hash of `MANIFEST.txt` is inserted as the
   first entry, followed by the 225 payload files (226 hashes total). `CHECKSUMS.sha256` does
   not hash itself (no circular dependency). **LF line endings.**
4. **Single-Pass Master ZIP Packaging:** `tbnnFoam-1.0.1-source.zip` built in one pass from
   staging (arcname prefix `tbnnFoam-1.0.1/`); no post-hoc appends. Final size 36.46 MB,
   227 entries.
5. **Round-Trip Validation:** archive extracted to `release/tbnnFoam-1.0.1-source/`; all 226
   SHA-256 digests verified against the extracted files — zero discrepancies.
6. **External Archive Digest:** independent SHA-256 of the outer ZIP recorded in
   `release/tbnnFoam-1.0.1-source.zip.sha256` (`638c2593570c9aa084e45a48336518376730dcb56d0ed63a0b4b80d259089e1a`).

---

## 4. Shipped Model Copies & Multi-Copy Parity Audit (re-verified this build)

The packager re-verified every shipped model artifact against the authoritative production model:

### A. Authoritative Production Model (64-Neuron Architecture)
* **Architecture:** 5 scalar invariants → [64, 64, 64] → 10 basis expansion coefficients.
* **Parameter Count:** exactly 9,354 trainable weights and biases (384 + 4,160 + 4,160 + 650).
* **Authoritative Checkpoints:** `tbnn_model.pt` (TorchScript), `tbnn_weights.pth` (state dict),
  `export_weights/weights.json` (plain JSON for `src/tbnnEvaluator.H`).

### B. Verification Results (this build)
* Root TorchScript model: 9,354 parameters ✓
* Root state dict: 9,354 parameters; **max deviation 0.0** vs TorchScript ✓
* Case-local TorchScript copies (`verification_squareDuct/constant`,
  `baseline_sst/constant`, `fine_mesh/constant`, `verification_squareDuct_Re17800/constant`):
  9,354 parameters, **0.0 tensor deviation**, raw weight-stream
  SHA-256 `dbfdc658952c3e3610b2fa77f08d24b742aa779dd8cff759ec8a7f63d4cb9858` ✓
* Case-local `weights.json` copies: byte-identical JSON contents to `export_weights/weights.json` ✓

### C. TorchScript Container Note (unchanged from v1.0.0)
Case-local `tbnn_model.pt` copies differ from the root copy by 64 bytes of TorchScript
container metadata (`.debug_pkl` filesystem paths from the original `torch.jit.trace`
invocation directories). The raw parameter tensor byte streams are 100% identical (hash above);
numerics, architecture, and inference outputs are mathematically identical.

### D. Legacy Artifact Disambiguation (unchanged)
`export_weights/tbnn_model.pt` (30-neuron prototype, 2,350 parameters),
`python/train_tbnn.py`, and `python/train_numpy_fallback.py` are retained for provenance and
documented in their respective READMEs.

---

## 5. Licensing Boundary, Git History Purge & Third-Party Dataset Decoupling

*(Unchanged in substance from v1.0.0; re-verified by leak scan on this build.)*

* **MIT Software License** applies exclusively to the tbnnFoam software: C++ solver code
  (`tbnnFoam.C`, `tbnnSimpleFoam/`), evaluator header (`src/tbnnEvaluator.H`), Python training
  and export pipelines (`tbnn_train.py`, `tbnn_export.py`, `tbnn_tensor_math.py`), synthetic
  test fixtures (`tests/fixtures/synthetic_dns_grid.csv`), automated verification suites
  (`tests/`), validation post-processing utilities (`scripts/validation/`), and OpenFOAM case
  setups (`verification_squareDuct/`, `verification_squareDuct_Re17800/`).
* **Upstream DNS decoupling:** raw and converted DNS datasets are not redistributed; historical
  copies were removed from branch history via `git-filter-repo`. Provenance, source URLs
  (Sapienza University of Rome CFD Group; Pirozzoli et al., *J. Fluid Mech.*, 2018;
  Vinuesa et al., 2014), scaling, and conversion instructions are documented in
  `data/benchmark_dns/README.md` with the conversion utility `scripts/convert_real_dns.py`
  (local processing only — performs no network downloads).
* **This build's leak scan:** no `*.dat`, no `*.csv` (other than the project-authored synthetic
  fixture), and no `raw_dns/` content anywhere in the archive. ✓
* **Attribution:** any academic use of the upstream DNS benchmarks must cite Pirozzoli et al.
  (2018), *Turbulence and secondary motions in square duct flow*, J. Fluid Mech., 840, 634–655.

---

## 6. Automated Verification Suite: Execution & Results (this build)

The 21-test `pytest` suite was executed **twice** on the rebuilt package:

1. **Inside the staging tree** (Windows Python 3.12, no OpenFOAM binaries, no DNS):
   **18 passed, 3 skipped, 0 failures** in 59.62 s.
2. **From a fresh independent extraction of the final ZIP** (clean temp directory outside the
   release tree): **18 passed, 3 skipped, 0 failures** in 52.46 s — confirming the shipped
   payload itself is functional.

**Environment matrix:**

| Environment | Result |
| :--- | :--- |
| Plain Python (no OpenFOAM binaries, no DNS — CI profile) | 18 passed, 3 skipped |
| + OpenFOAM toolchain (compiled C++ parity test runs) | 19 passed, 2 skipped |
| + external DNS data supplied | 21 passed, 0 skipped |

**Skips enumerated (CI profile):**
* `test_standalone_cpp_numerical_parity` — requires the compiled C++ evaluator binary/OpenFOAM
  toolchain; self-skips cleanly.
* `test_cfd_invariants_within_training_envelope`, `test_basis_norms_bounded_by_training_distribution`
  — require the external DNS dataset; self-skip cleanly.

**Correction note:** since the v1.0.1 verification-script update,
`test_momentum_coupling_parity` runs in **pure Python** and no longer requires the OpenFOAM
toolchain (previously 17 passed / 4 skipped on CI; the CI badge now reflects 18/3).
Full integration with OpenFOAM + DNS remains 21 passed / 0 skipped.

**Verified verification assertions (unchanged):**
1. Pope 10-tensor integrity basis tracelessness: max |tr(T^(k))| ≤ 1e-12.
2. Pope basis symmetry: max ‖T − Tᵀ‖_F ≤ 1e-12.
3. Cardano cubic closed-form eigensolver accuracy ≤ 1e-6 vs NumPy eigenvalues.
4. Lumley triangle realizability limiter bounds eigenvalues (a_i ≥ −1/3, Σa_i = 0).
5. Cross-engine numerical parity (PyTorch, TorchScript, standalone evaluator) ≤ 1e-5.
6. Boussinesq-limit momentum assembly consistency: relative deviation 3.6e-17.
7. Algorithmic vortex detection recovers all 8 secondary recirculation cores.

---

## 7. Independent Third-Party-Style Verification (2026-11-01)

Beyond the packager's self-verification, the final artifact was audited independently:

* Fresh `unzip` of `tbnnFoam-1.0.1-source.zip` into a clean temp directory outside `release/`.
* **Plain `sha256sum -c CHECKSUMS.sha256`** (no line-ending preprocessing — exactly what a
  Linux/macOS user runs): **226/226 OK, zero failures**. ✓
* Line-ending audit of all shipped checksum/manifest files: **LF**. ✓
* `unzip -l`: 227 entries. Outer ZIP hash matches sidecar. ✓
* DNS leakage scan: clean (§5). ✓
* `INSTALL.md` present in payload with current test matrix. ✓

---

## 8. Excluded Preprint Assets & Clean Scope Separation

The release archive represents exclusively the software deposit:

| Excluded Asset | Repository Location | Exclusion Rationale |
| :--- | :--- | :--- |
| Compiled Preprint PDF | `tbnnFoam_preprint.pdf`, `zenodo_preprint_package/` | Preprint document; separate publication deposit. |
| Manuscript Source | `docs/PREPRINT_MANUSCRIPT.md` | Preprint markdown manuscript draft. |
| LaTeX Compilation Tree | `docs/latex/` | Publication compilation sources, not software. |
| Preprint Audit Tool | `scripts/validation/audit_preprint_claims.py` | Audits the LaTeX manuscript, not the software. |
| Publication Packager | `scripts/package_publication.py`, `build_zenodo_package.py` | Package the preprint bundle, not the software. |
| Legacy Mixed Package | `publication_package/`, `tbnnFoam_Publication_Package.zip` | **Never publish** — contains MIT-licensed DNS CSVs mixed with software. |
| Build Caches & Binaries | `Make/linux*`, `__pycache__`, `*.pyc`, `.pytest_cache` | Temporary build products and caches. |

All technical software documentation, architecture diagrams, benchmark figures, and simulation
animations (`docs/figures/`) are preserved in the release archive.

---

## 9. Zenodo Software Record: Manual Upload Guide (v1.0.1 — New Version)

> [!IMPORTANT]
> **Use Manual Upload on Zenodo — Do NOT Use the GitHub Integration Switch**
> The repository root contains a preprint-oriented `.zenodo.json` (`upload_type: publication`,
> `publication_type: preprint`, `cc-by-4.0`). Zenodo's automated GitHub release webhook reads
> the repository-root `.zenodo.json` in preference to `CITATION.cff`. Triggering a release via
> GitHub integration would mistakenly index the software release as a preprint publication
> under CC-BY-4.0.

### Step-by-Step Instructions (publishing v1.0.1 as a new version of the existing record):
1. Log in to [Zenodo](https://zenodo.org/) and open the existing software record
   **DOI `10.5281/zenodo.23799524`** (currently holding v1.0.0).
2. Click **"New version"**.
3. **Files:** remove nothing (v1.0.0 files remain in their own version); upload only
   `release/tbnnFoam-1.0.1-source.zip` to the new version.
4. **Resource Type:** Software. **License:** MIT. **Version:** `1.0.1`.
5. **Metadata:** carry over the values from `release/zenodo_software.json` (title, creators,
   HTML description, keywords) — they are already version-agnostic; confirm the version field
   reads 1.0.1.
6. Optionally append a one-line version note: *"v1.0.1: reconciled benchmark metadata, corrected
   verification scripts, added INSTALL.md installation and usage guide, LF-normalized checksum
   files."*
7. **Related Identifiers:** once the preprint record is published, link it here with relation
   `isSupplementedBy` (the preprint record already links this software DOI).
8. **Save → Publish.**

---

*End of audit report — generated 2026-11-01 following the full rebuild and independent verification.*
