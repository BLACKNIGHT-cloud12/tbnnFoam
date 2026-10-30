#!/usr/bin/env python3
"""
tbnnFoam Software Release Manager
Packages the complete, self-contained, archival-ready software release of tbnnFoam for Zenodo.
Produces:
- release/tbnnFoam-1.0.0-source.zip
- release/tbnnFoam-1.0.0-source/ (clean extracted staging directory)
- release/MANIFEST.txt
- release/CHECKSUMS.sha256
- release/RELEASE_AUDIT.md
"""

import os
import shutil
import hashlib
import json
import zipfile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = REPO_ROOT / "release"
STAGING_DIR = RELEASE_DIR / "staging" / "tbnnFoam-1.0.0"
OUTPUT_ZIP = RELEASE_DIR / "tbnnFoam-1.0.0-source.zip"
EXTRACT_DIR = RELEASE_DIR / "tbnnFoam-1.0.0-source"

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def create_export_weights_readme(dst: Path):
    content = """# Exported Model Weights (C++ Standalone Evaluator)

This directory contains exported model weight artifacts consumed by the zero-dependency C++ standalone evaluator (`src/tbnnEvaluator.H`).

## Active Production Weights
* **`weights.json`**:
  - **Architecture:** 5 invariants -> [64, 64, 64] -> 10 basis expansion coefficients g^(1...10).
  - **Parameters:** 9,354 trainable parameters (layer counts: 384 + 4160 + 4160 + 650 = 9,354).
  - **Precision:** Single-precision float arrays.
  - **Normalization:** `inv_mean` = [0, 0, 0, 0, 0], `inv_std` = [1, 1, 1, 1, 1] (the production model trains directly on physical invariants).
  - **Parity:** Numerically identical (max parameter deviation = 0.0) to the TorchScript model `tbnn_model.pt` and PyTorch state dict `tbnn_weights.pth`.

## Legacy Artifacts
* **`tbnn_model.pt`** (in this directory):
  - **Note:** Historical 30-neuron prototype model (2,350 parameters). Retained strictly for provenance and historical reference.
  - **Active Model:** The authoritative 64-neuron production model is located at the repository root (`tbnn_model.pt`) and in the verification cases (`verification_squareDuct/constant/tbnn_model.pt`).
"""
    (dst / "README.md").write_text(content, encoding="utf-8")

def create_python_readme(dst: Path):
    content = """# Python Training & Experimental Pipelines

This directory contains experimental and fallback training implementations:

* **`train_tbnn.py`**:
  - **Status:** Legacy 30-neuron prototype experiment.
  - Trains an earlier 30-hidden-unit network writing to `legacy_exports/`.
* **`train_numpy_fallback.py`**:
  - **Status:** Dependency-free scikit-learn fallback experiment.
  - Trains an `MLPRegressor` using pseudo-inverse tensor basis projection without requiring PyTorch.

## Authoritative Shipped Training Pipeline
The authoritative, reproducible training pipeline for the shipped 64-width production model (9,354 parameters) is located in the repository root:
* **`tbnn_train.py`**: Trains the production model on canonical DNS data (`data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv`) with fixed random seeds (42).
* **`tbnn_export.py`**: Exports weights to TorchScript (`tbnn_model.pt`) and JSON (`export_weights/weights.json`).
* **`tbnn_tensor_math.py`**: Canonical implementation of Pope's 10-tensor integrity basis, Cayley-Hamilton invariants, and Cardano cubic realizability limiter.
"""
    (dst / "README.md").write_text(content, encoding="utf-8")

def create_dns_readme(dst: Path):
    content = """# Benchmark DNS Datasets

Authoritative Direct Numerical Simulation (DNS) benchmark datasets for turbulent square duct flow from the Sapienza University CFD Group.

## Primary Citations & Provenance
* **Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018).** *Turbulence and secondary motions in square duct flow.* Journal of Fluid Mechanics, 840, 634-655.
* **Vinuesa, R., Noorani, A., Lozano-Durán, A., El Khoury, G. K., Schlatter, P., Fischer, P. F., & Nagib, H. M. (2014).** *Aspect ratio effects in turbulent duct flows studied through direct numerical simulation.* Journal of Turbulence, 15(10), 677-706.
* **Database Repository:** Sapienza University of Rome CFD Group DNS Database (http://newton.dma.uniroma1.it/square_duct/).

## Dataset Files
1. **`square_duct_dns_Reb4410_Retau150.csv`**:
   - Reb = 4,410 (Reh = 2,205, Retau = 150).
   - 4,225 grid cells (65 x 65 cross-section).
   - Contains: (y, z) coordinates, mean velocity gradients dUi/dxj, Reynolds stress anisotropy tensor components a_ij, and specific dissipation surrogate omega.
2. **`square_duct_dns_Reb7000_Retau220.csv`**:
   - Reb = 7,000 (Reh = 3,500, Retau = 220).
   - 4,225 grid cells (65 x 65 cross-section).
3. **`raw_dns/plotyz_Retau150.dat`** & **`raw_dns/plotyz_Retau220.dat`**:
   - Raw exported data files from the Pirozzoli et al. database.
4. **`benchmark_metadata.json`**:
   - Comprehensive machine-readable specification of physical scaling, Reynolds number definitions, coordinate systems, and baseline flow parameters.

## License and Terms of Use
The benchmark DNS datasets and raw files are redistribution copies for open scientific validation and peer review under academic fair use. All intellectual property for the DNS simulation data belongs to Pirozzoli et al. The software license (MIT) applies to the software, scripts, and parsing tools, not to the underlying DNS simulation data. Any publication using these datasets must cite Pirozzoli et al. (2018).
"""
    (dst / "README.md").write_text(content, encoding="utf-8")

def create_zenodo_software_json(dst: Path):
    zenodo_dict = {
        "title": "tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure",
        "upload_type": "software",
        "description": "<p><b>tbnnFoam</b> is an open-source in-situ coupled Reynolds-Averaged Navier–Stokes (RANS) solver within OpenFOAM featuring a dual-runtime inference architecture: an embedded LibTorch C++ engine and a zero-dependency standalone C++ matrix evaluator with an analytical Cardano cubic eigensolver that enforces physical realizability on the Lumley triangle in \\(\\mathcal{O}(1)\\) operations per cell.</p>\n<p>This software release archives the complete source code, trained neural network weights, LibTorch and native C++ inference backends, automated 21-test verification suite, canonical square duct DNS benchmark datasets (Pirozzoli et al., <i>J. Fluid Mech.</i> 2018), and full OpenFOAM verification cases for developing square duct flow at <i>Re<sub>b</sub></i> = 4,410 and <i>Re<sub>b</sub></i> = 17,800.</p>\n<p>Features cross-engine numerical parity across PyTorch, TorchScript, and standalone C++ (within 10<sup>-5</sup>), exact Boussinesq-limit momentum assembly consistency (relative deviation 3.6 &times; 10<sup>-17</sup>), and independent algorithmic vortex detection recovering Prandtl's eight secondary recirculation cells.</p>",
        "creators": [
            {
                "name": "Ghosh, Soumik",
                "affiliation": "Department of Civil Engineering, Jadavpur University"
            }
        ],
        "version": "1.0.0",
        "access_right": "open",
        "license": "MIT",
        "keywords": [
            "OpenFOAM",
            "Computational Fluid Dynamics",
            "CFD",
            "Turbulence Modeling",
            "Tensor Basis Neural Network",
            "TBNN",
            "LibTorch",
            "Scientific Machine Learning",
            "SciML",
            "RANS",
            "Reynolds Stress Anisotropy",
            "Lumley Triangle",
            "Realizability",
            "Secondary Flow"
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/BLACKNIGHT-cloud12/tbnnFoam",
                "relation": "isSupplementTo",
                "scheme": "url"
            }
        ]
    }
    (dst / ".zenodo.json").write_text(json.dumps(zenodo_dict, indent=2), encoding="utf-8")

def create_citation_cff(dst: Path):
    cff = """cff-version: 1.2.0
message: "If you use tbnnFoam in your scientific research or engineering simulations, please cite it as below."
authors:
  - family-names: "Ghosh"
    given-names: "Soumik"
    affiliation: "Department of Civil Engineering, Jadavpur University"
title: "tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure"
version: 1.0.0
date-released: 2026-10-19
url: "https://github.com/BLACKNIGHT-cloud12/tbnnFoam"
repository-code: "https://github.com/BLACKNIGHT-cloud12/tbnnFoam"
keywords:
  - "OpenFOAM"
  - "CFD"
  - "Turbulence Modeling"
  - "Tensor Basis Neural Network"
  - "TBNN"
  - "LibTorch"
  - "Scientific Machine Learning"
  - "SciML"
  - "RANS"
license: MIT
"""
    (dst / "CITATION.cff").write_text(cff, encoding="utf-8")

def clean_caches(base_dir: Path):
    for p in list(base_dir.rglob("__pycache__")):
        if p.exists():
            shutil.rmtree(p)
    for p in list(base_dir.rglob(".pytest_cache")):
        if p.exists():
            shutil.rmtree(p)
    for p in list(base_dir.rglob("linux64Gcc*")):
        if p.exists():
            shutil.rmtree(p)
    for f in list(base_dir.rglob("*.pyc")):
        if f.exists():
            f.unlink()
    for bin_name in ["test_evaluator_parity", "test_momentum_parity"]:
        for f in list(base_dir.rglob(bin_name)):
            if f.is_file():
                f.unlink()

def stage_release():
    print("=== [1/7] Initializing Staging Directory ===")
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if (RELEASE_DIR / 'staging').exists():
        shutil.rmtree(RELEASE_DIR / 'staging')
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Staging to: {STAGING_DIR}")

    print("=== [2/7] Copying Core Software Components ===")
    # Root source and python files
    root_files = [
        "tbnnFoam.C", "createFields.H",
        "tbnn_train.py", "tbnn_export.py", "tbnn_tensor_math.py",
        "tbnn_model.pt", "tbnn_weights.pth",
        "requirements.txt", "LICENSE",
        "CODE_OF_CONDUCT.md", "CONTRIBUTING.md"
    ]
    for rf in root_files:
        src = REPO_ROOT / rf
        if src.exists():
            shutil.copy2(src, STAGING_DIR / rf)

    # Make/
    shutil.copytree(REPO_ROOT / "Make", STAGING_DIR / "Make")

    # src/
    shutil.copytree(REPO_ROOT / "src", STAGING_DIR / "src")

    # export_weights/
    shutil.copytree(REPO_ROOT / "export_weights", STAGING_DIR / "export_weights")
    create_export_weights_readme(STAGING_DIR / "export_weights")

    # python/
    shutil.copytree(REPO_ROOT / "python", STAGING_DIR / "python")
    create_python_readme(STAGING_DIR / "python")

    # data/
    shutil.copytree(REPO_ROOT / "data", STAGING_DIR / "data")
    create_dns_readme(STAGING_DIR / "data" / "benchmark_dns")

    # tests/
    shutil.copytree(REPO_ROOT / "tests", STAGING_DIR / "tests")

    # scripts/
    scripts_dst = STAGING_DIR / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "convert_real_dns.py", scripts_dst / "convert_real_dns.py")
    shutil.copy2(REPO_ROOT / "scripts" / "generate_all_showcase_assets.py", scripts_dst / "generate_all_showcase_assets.py")
    shutil.copy2(REPO_ROOT / "scripts" / "generate_showcase_animations.py", scripts_dst / "generate_showcase_animations.py")

    # validation scripts (exclude preprint audit script)
    val_dst = scripts_dst / "validation"
    val_dst.mkdir(parents=True, exist_ok=True)
    for vf in (REPO_ROOT / "scripts" / "validation").iterdir():
        if vf.is_file() and vf.name != "audit_preprint_claims.py":
            shutil.copy2(vf, val_dst / vf.name)

    # verification cases
    shutil.copytree(REPO_ROOT / "verification_squareDuct", STAGING_DIR / "verification_squareDuct")
    shutil.copytree(REPO_ROOT / "verification_squareDuct_Re17800", STAGING_DIR / "verification_squareDuct_Re17800")

    # Make run_baseline.sh portable
    baseline_sh = STAGING_DIR / "verification_squareDuct" / "run_baseline.sh"
    if baseline_sh.exists():
        b_content = baseline_sh.read_text(encoding="utf-8")
        b_content = b_content.replace(
            'ROOT_DIR="/mnt/c/Users/soumi/tbnnFoam/verification_squareDuct"',
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nROOT_DIR="${SCRIPT_DIR}"'
        )
        baseline_sh.write_text(b_content, encoding="utf-8")

    # docs/figures
    (STAGING_DIR / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    for fig in (REPO_ROOT / "docs" / "figures").iterdir():
        if fig.is_file():
            shutil.copy2(fig, STAGING_DIR / "docs" / "figures" / fig.name)

    # .github
    if (REPO_ROOT / ".github").exists():
        shutil.copytree(REPO_ROOT / ".github", STAGING_DIR / ".github")

    # .gitignore
    shutil.copy2(REPO_ROOT / ".gitignore", STAGING_DIR / ".gitignore")

    # Create metadata files
    create_zenodo_software_json(STAGING_DIR)
    create_citation_cff(STAGING_DIR)
    shutil.copy2(REPO_ROOT / "README.md", STAGING_DIR / "README.md")

    clean_caches(STAGING_DIR)

def verify_staged_models():
    print("=== [3/7] Verifying Model Artifacts & Integrity ===")
    import torch

    pt_model = STAGING_DIR / "tbnn_model.pt"
    assert pt_model.exists(), "tbnn_model.pt missing"
    m_jit = torch.jit.load(str(pt_model))
    params = list(m_jit.parameters())
    num_params = sum(p.numel() for p in params)
    print(f"Verified TorchScript model: {num_params} parameters (expected 9,354)")
    assert num_params == 9354, f"Unexpected parameter count {num_params}"

    pth_model = STAGING_DIR / "tbnn_weights.pth"
    assert pth_model.exists(), "tbnn_weights.pth missing"
    sd = torch.load(str(pth_model), map_location="cpu")
    pth_params = sum(p.numel() for p in sd.values())
    print(f"Verified PyTorch state dict: {pth_params} parameters (expected 9,354)")
    assert pth_params == 9354

    wj_file = STAGING_DIR / "export_weights" / "weights.json"
    assert wj_file.exists(), "weights.json missing"
    with open(wj_file) as f:
        wj = json.load(f)
    assert len(wj["0.weight"]) == 64 and len(wj["0.weight"][0]) == 5
    assert len(wj["2.weight"]) == 64 and len(wj["2.weight"][0]) == 64
    assert len(wj["4.weight"]) == 64 and len(wj["4.weight"][0]) == 64
    assert len(wj["6.weight"]) == 10 and len(wj["6.weight"][0]) == 64
    print("Verified export_weights/weights.json: [64x5, 64x64, 64x64, 10x64] matching production model")

def run_staged_tests():
    print("=== [4/7] Executing Verification Test Suite in Staging ===")
    res = subprocess.run(["pytest", "tests/"], cwd=str(STAGING_DIR), capture_output=True, text=True)
    print("Pytest stdout:\n" + "\n".join(res.stdout.splitlines()[-5:]))
    assert res.returncode == 0, f"Tests failed in staging directory:\n{res.stdout}\n{res.stderr}"
    print("All verification tests passed in staging directory!")

    # Clean caches generated by pytest immediately
    clean_caches(STAGING_DIR)

def package_and_extract():
    print("=== [5/7] Building Master ZIP & Validating Clean Extraction ===")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)

    # Ensure no cache files in staging before zipping
    clean_caches(STAGING_DIR)

    # Create zip
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(STAGING_DIR):
            for file in files:
                full_p = Path(root) / file
                rel_p = full_p.relative_to(STAGING_DIR)
                arcname = Path("tbnnFoam-1.0.0") / rel_p
                z.write(full_p, arcname)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Created: {OUTPUT_ZIP} ({zip_size_mb:.2f} MB)")

    # Extract to EXTRACT_DIR
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as z:
        z.extractall(RELEASE_DIR / "temp_extract")

    # Move content of tbnnFoam-1.0.0 directly into EXTRACT_DIR
    extracted_inner = RELEASE_DIR / "temp_extract" / "tbnnFoam-1.0.0"
    for item in extracted_inner.iterdir():
        shutil.move(str(item), str(EXTRACT_DIR / item.name))
    shutil.rmtree(RELEASE_DIR / "temp_extract")
    print(f"Extracted clean staging directory to: {EXTRACT_DIR}")

def generate_manifest_and_checksums():
    print("=== [6/7] Generating MANIFEST.txt & CHECKSUMS.sha256 ===")
    manifest_lines = [
        "tbnnFoam v1.0.0 Software Release Manifest",
        "========================================",
        "Archive: tbnnFoam-1.0.0-source.zip",
        "Release Date: 2026-10-19 / Audited: 2026-10-30",
        "Author: Soumik Ghosh (Department of Civil Engineering, Jadavpur University)",
        "License: MIT (Software) / Academic Citation Required (DNS Benchmark Data)",
        "",
        "File & Directory Inventory:",
        "---------------------------"
    ]

    checksum_lines = []

    all_files = sorted(list(EXTRACT_DIR.rglob("*")))
    file_count = 0
    total_bytes = 0

    for f in all_files:
        if f.is_file():
            file_count += 1
            rel_p = f.relative_to(EXTRACT_DIR)
            sha = compute_sha256(f)
            size = f.stat().st_size
            total_bytes += size
            manifest_lines.append(f"{str(rel_p):<75} {size:>10} bytes  SHA256: {sha}")
            checksum_lines.append(f"{sha}  {str(rel_p).replace(chr(92), '/')}")

    manifest_lines.extend([
        "",
        "Summary:",
        f"  Total Files: {file_count}",
        f"  Total Uncompressed Size: {total_bytes / (1024*1024):.2f} MB",
        f"  Archive Size: {OUTPUT_ZIP.stat().st_size / (1024*1024):.2f} MB"
    ])

    (RELEASE_DIR / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (RELEASE_DIR / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    shutil.copy2(RELEASE_DIR / "MANIFEST.txt", EXTRACT_DIR / "MANIFEST.txt")
    shutil.copy2(RELEASE_DIR / "CHECKSUMS.sha256", EXTRACT_DIR / "CHECKSUMS.sha256")

    # Update zip to include MANIFEST.txt and CHECKSUMS.sha256
    with zipfile.ZipFile(OUTPUT_ZIP, "a", zipfile.ZIP_DEFLATED) as z:
        z.write(RELEASE_DIR / "MANIFEST.txt", Path("tbnnFoam-1.0.0") / "MANIFEST.txt")
        z.write(RELEASE_DIR / "CHECKSUMS.sha256", Path("tbnnFoam-1.0.0") / "CHECKSUMS.sha256")

    print(f"Generated MANIFEST.txt ({file_count} files, {total_bytes / (1024*1024):.2f} MB total)")
    print(f"Generated CHECKSUMS.sha256")

    # Clean intermediate staging folder
    if (RELEASE_DIR / "staging").exists():
        shutil.rmtree(RELEASE_DIR / "staging")
    print("Cleaned intermediate staging folder.")

if __name__ == "__main__":
    stage_release()
    verify_staged_models()
    run_staged_tests()
    package_and_extract()
    generate_manifest_and_checksums()
    print("\n[SUCCESS] Software release generated successfully in release/")
