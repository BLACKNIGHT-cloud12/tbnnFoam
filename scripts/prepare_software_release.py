#!/usr/bin/env python3
"""
tbnnFoam Software Release Manager
Packages the complete, self-contained, archival-ready software release of tbnnFoam for Zenodo.
Produces:
- release/tbnnFoam-1.0.1-source.zip
- release/tbnnFoam-1.0.1-source/ (clean extracted staging directory)
- release/MANIFEST.txt
- release/CHECKSUMS.sha256
- release/zenodo_software.json
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
STAGING_DIR = RELEASE_DIR / "staging" / "tbnnFoam-1.0.1"
OUTPUT_ZIP = RELEASE_DIR / "tbnnFoam-1.0.1-source.zip"
EXTRACT_DIR = RELEASE_DIR / "tbnnFoam-1.0.1-source"

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
    (dst / "README.md").write_text(content, encoding="utf-8", newline="\n")

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
    (dst / "README.md").write_text(content, encoding="utf-8", newline="\n")

def create_dns_readme(dst: Path):
    src = REPO_ROOT / "data" / "benchmark_dns" / "README.md"
    shutil.copy2(src, dst / "README.md")

def create_zenodo_software_json(dst: Path):
    zenodo_dict = {
        "title": "tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure",
        "upload_type": "software",
        "description": "<p><b>tbnnFoam</b> is an open-source in-situ coupled Reynolds-Averaged Navier–Stokes (RANS) solver within OpenFOAM featuring a dual-runtime inference architecture: an embedded LibTorch C++ engine and a zero-dependency standalone C++ matrix evaluator with an analytical Cardano cubic eigensolver that enforces physical realizability on the Lumley triangle in \\(\\mathcal{O}(1)\\) operations per cell.</p>\n<p>This software release archives the complete source code, pre-trained neural network weights, LibTorch and native C++ inference backends, automated verification test suite with synthetic fixtures, canonical square duct benchmark specifications (Pirozzoli et al., <i>J. Fluid Mech.</i> 2018), and full OpenFOAM verification cases for developing square duct flow at <i>Re<sub>b</sub></i> = 4,410 and <i>Re<sub>b</sub></i> = 17,800.</p>\n<p>Features cross-engine numerical parity across PyTorch, TorchScript, and standalone C++ (within 10<sup>-5</sup>), exact Boussinesq-limit momentum assembly consistency (relative deviation 3.6 &times; 10<sup>-17</sup>), and independent algorithmic vortex detection recovering Prandtl's eight secondary recirculation cells.</p>",
        "creators": [
            {
                "name": "Ghosh, Soumik",
                "affiliation": "Department of Civil Engineering, Jadavpur University"
            }
        ],
        "version": "1.0.1",
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
        ]
    }
    (dst / ".zenodo.json").write_text(json.dumps(zenodo_dict, indent=2), encoding="utf-8", newline="\n")

def create_citation_cff(dst: Path):
    cff = """cff-version: 1.2.0
message: "If you use tbnnFoam in your scientific research or engineering simulations, please cite it as below."
authors:
  - family-names: "Ghosh"
    given-names: "Soumik"
    affiliation: "Department of Civil Engineering, Jadavpur University"
title: "tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure"
version: 1.0.1
date-released: 2026-11-01
doi: 10.5281/zenodo.23799524
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
    (dst / "CITATION.cff").write_text(cff, encoding="utf-8", newline="\n")

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
    if (RELEASE_DIR / "staging").exists():
        shutil.rmtree(RELEASE_DIR / "staging")
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
        "requirements.txt", "INSTALL.md", "LICENSE",
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

    # tests/
    shutil.copytree(REPO_ROOT / "tests", STAGING_DIR / "tests")

    # data/
    shutil.copytree(REPO_ROOT / "data", STAGING_DIR / "data")
    create_dns_readme(STAGING_DIR / "data" / "benchmark_dns")

    # verification_squareDuct/
    shutil.copytree(REPO_ROOT / "verification_squareDuct", STAGING_DIR / "verification_squareDuct")

    # verification_squareDuct_Re17800/
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

    # docs/figures/
    (STAGING_DIR / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "docs" / "figures", STAGING_DIR / "docs" / "figures")

    # .github/
    shutil.copytree(REPO_ROOT / ".github", STAGING_DIR / ".github")

    # .gitignore
    shutil.copy2(REPO_ROOT / ".gitignore", STAGING_DIR / ".gitignore")

    # Create metadata files inside staging
    create_zenodo_software_json(STAGING_DIR)
    create_citation_cff(STAGING_DIR)
    shutil.copy2(REPO_ROOT / "README.md", STAGING_DIR / "README.md")

    clean_caches(STAGING_DIR)

def verify_staged_models():
    print("=== [3/7] Verifying Model Artifacts & Cross-Copy Parity ===")
    import torch

    # 1. Authoritative root TorchScript model
    pt_model = STAGING_DIR / "tbnn_model.pt"
    assert pt_model.exists(), "tbnn_model.pt missing in staging"
    m_jit = torch.jit.load(str(pt_model))
    root_params = dict(m_jit.named_parameters())
    num_params = sum(p.numel() for p in root_params.values())
    print(f"Verified Root TorchScript model: {num_params} parameters (expected 9,354)")
    assert num_params == 9354, f"Unexpected parameter count {num_params}"

    # 2. Authoritative root PyTorch state dict
    pth_model = STAGING_DIR / "tbnn_weights.pth"
    assert pth_model.exists(), "tbnn_weights.pth missing in staging"
    sd = torch.load(str(pth_model), map_location="cpu")
    pth_params = sum(p.numel() for p in sd.values())
    print(f"Verified Root PyTorch state dict: {pth_params} parameters (expected 9,354)")
    assert pth_params == 9354

    # Cross-check PyTorch state dict against TorchScript root
    for k, v in sd.items():
        assert k in root_params, f"Key {k} missing in root TorchScript model"
        dev = (v - root_params[k]).abs().max().item()
        assert dev == 0.0, f"State dict deviation from TorchScript for {k}: {dev}"
    print("Verified exact 0.0 parameter deviation between tbnn_weights.pth and tbnn_model.pt")

    # 3. Export weights JSON
    wj_file = STAGING_DIR / "export_weights" / "weights.json"
    assert wj_file.exists(), "weights.json missing in staging"
    with open(wj_file) as f:
        wj_root = json.load(f)
    assert len(wj_root["0.weight"]) == 64 and len(wj_root["0.weight"][0]) == 5
    assert len(wj_root["2.weight"]) == 64 and len(wj_root["2.weight"][0]) == 64
    assert len(wj_root["4.weight"]) == 64 and len(wj_root["4.weight"][0]) == 64
    assert len(wj_root["6.weight"]) == 10 and len(wj_root["6.weight"][0]) == 64
    print("Verified export_weights/weights.json: [64x5, 64x64, 64x64, 10x64] matching production model")

    # 4. Verify EVERY shipped case-local TorchScript copy
    case_pt_models = [
        ("verification_squareDuct/constant", STAGING_DIR / "verification_squareDuct" / "constant" / "tbnn_model.pt"),
        ("verification_squareDuct/baseline_sst/constant", STAGING_DIR / "verification_squareDuct" / "baseline_sst" / "constant" / "tbnn_model.pt"),
        ("verification_squareDuct/fine_mesh/constant", STAGING_DIR / "verification_squareDuct" / "fine_mesh" / "constant" / "tbnn_model.pt"),
        ("verification_squareDuct_Re17800/constant", STAGING_DIR / "verification_squareDuct_Re17800" / "constant" / "tbnn_model.pt"),
    ]

    EXPECTED_WEIGHTS_STREAM_SHA256 = "dbfdc658952c3e3610b2fa77f08d24b742aa779dd8cff759ec8a7f63d4cb9858"

    for label, model_path in case_pt_models:
        assert model_path.exists(), f"Case model missing: {label}"
        case_jit = torch.jit.load(str(model_path))
        case_params = dict(case_jit.named_parameters())
        assert set(case_params.keys()) == set(root_params.keys()), f"Parameter key mismatch in {label}"
        max_dev = 0.0
        for k in root_params:
            dev = (root_params[k] - case_params[k]).abs().max().item()
            if dev > max_dev:
                max_dev = dev
        assert max_dev == 0.0, f"Non-zero tensor deviation in {label}: {max_dev}"

        # Verify raw parameter byte stream hash inside TorchScript zip container
        with zipfile.ZipFile(model_path) as z:
            h = hashlib.sha256()
            for i in range(8):
                h.update(z.read(f"tbnn_model/data/{i}"))
            stream_sha = h.hexdigest()
            assert stream_sha == EXPECTED_WEIGHTS_STREAM_SHA256, f"Raw tensor stream hash mismatch in {label}"

        print(f"Verified case TorchScript model {label}: 9,354 params, 0.0 tensor deviation, weight stream SHA256 verified")

    # 5. Verify case-local weights.json files
    case_json_models = [
        ("verification_squareDuct/constant/tbnn", STAGING_DIR / "verification_squareDuct" / "constant" / "tbnn" / "weights.json"),
        ("verification_squareDuct_Re17800/constant/tbnn", STAGING_DIR / "verification_squareDuct_Re17800" / "constant" / "tbnn" / "weights.json"),
    ]
    for label, json_path in case_json_models:
        assert json_path.exists(), f"Case weights.json missing: {label}"
        with open(json_path) as f:
            wj_case = json.load(f)
        assert wj_case == wj_root, f"Case weights.json contents differ from root in {label}"
        print(f"Verified case JSON weights {label}: exact match with export_weights/weights.json")

def run_staged_tests():
    print("=== [4/7] Executing Verification Test Suite in Staging ===")
    res = subprocess.run(["pytest", "tests/"], cwd=str(STAGING_DIR), capture_output=True, text=True)
    print("Pytest stdout:\n" + "\n".join(res.stdout.splitlines()[-5:]))
    assert res.returncode == 0, f"Tests failed in staging directory:\n{res.stdout}\n{res.stderr}"
    summary_match = ""
    for line in reversed(res.stdout.splitlines()):
        if "passed" in line and ("=" in line or "in " in line):
            summary_match = line.strip("= ")
            break
    if summary_match:
        print(f"All verification tests passed in staging directory ({summary_match})!")
    else:
        print("All verification tests passed in staging directory!")

    # Clean caches generated by pytest immediately
    clean_caches(STAGING_DIR)

def generate_metadata_and_package():
    print("=== [5/7] Generating MANIFEST.txt & CHECKSUMS.sha256 ===")
    
    clean_caches(STAGING_DIR)
    payload_files = sorted([f for f in STAGING_DIR.rglob("*") if f.is_file()])
    assert len(payload_files) == 225, f"Expected 225 payload files in staging, found {len(payload_files)}"

    manifest_lines = [
        "tbnnFoam v1.0.1 Software Release Manifest",
        "========================================",
        "Archive: tbnnFoam-1.0.1-source.zip",
        "Release Date: 2026-11-01 / Audited: 2026-11-01",
        "Author: Soumik Ghosh (Department of Civil Engineering, Jadavpur University)",
        "License: MIT",
        "",
        "Archive Inventory & Structure:",
        "------------------------------",
        f"This release archive contains {len(payload_files)} software source, case, benchmark, and test",
        "files, accompanied by 2 release verification metadata files (MANIFEST.txt and",
        f"CHECKSUMS.sha256), totaling {len(payload_files) + 2} archive entries.",
        "",
        f"Payload File Inventory ({len(payload_files)} files):",
        "------------------------------------"
    ]

    checksum_lines = []
    total_payload_bytes = 0

    for f in payload_files:
        rel_p = f.relative_to(STAGING_DIR)
        rel_p_str = str(rel_p).replace(chr(92), "/")
        sha = compute_sha256(f)
        size = f.stat().st_size
        total_payload_bytes += size
        manifest_lines.append(f"{rel_p_str:<75} {size:>10} bytes  SHA256: {sha}")
        checksum_lines.append(f"{sha}  {rel_p_str}")

    manifest_lines.extend([
        "",
        "Summary:",
        f"  Payload Files: {len(payload_files)}",
        f"  Release Metadata Files: 2 (MANIFEST.txt, CHECKSUMS.sha256)",
        f"  Total Archive Entries: {len(payload_files) + 2}",
        f"  Total Uncompressed Payload Size: {total_payload_bytes / (1024*1024):.2f} MB",
        ""
    ])

    # Write MANIFEST.txt into staging and release directory
    manifest_text = "\n".join(manifest_lines) + "\n"
    (STAGING_DIR / "MANIFEST.txt").write_text(manifest_text, encoding="utf-8", newline="\n")
    (RELEASE_DIR / "MANIFEST.txt").write_text(manifest_text, encoding="utf-8", newline="\n")
    print(f"Generated MANIFEST.txt covering {len(payload_files)} payload files ({total_payload_bytes / (1024*1024):.2f} MB)")

    # Compute checksum of MANIFEST.txt itself and add to CHECKSUMS.sha256
    manifest_sha = compute_sha256(STAGING_DIR / "MANIFEST.txt")
    checksum_lines.insert(0, f"{manifest_sha}  MANIFEST.txt")

    # Write CHECKSUMS.sha256 into staging and release directory
    # CHECKSUMS.sha256 does not hash itself (226 entries: MANIFEST.txt + 225 payload files)
    checksum_text = "\n".join(checksum_lines) + "\n"
    (STAGING_DIR / "CHECKSUMS.sha256").write_text(checksum_text, encoding="utf-8", newline="\n")
    (RELEASE_DIR / "CHECKSUMS.sha256").write_text(checksum_text, encoding="utf-8", newline="\n")
    print(f"Generated CHECKSUMS.sha256 with {len(checksum_lines)} hashes ({len(payload_files)} payload files + MANIFEST.txt)")

    # Write release/zenodo_software.json for manual upload convenience
    create_zenodo_software_json(RELEASE_DIR)
    if (RELEASE_DIR / ".zenodo.json").exists():
        shutil.move(str(RELEASE_DIR / ".zenodo.json"), str(RELEASE_DIR / "zenodo_software.json"))
    print("Generated release/zenodo_software.json for manual Zenodo software deposit")

    # Build Master ZIP directly from STAGING_DIR (single-pass archive generation with all 226 files)
    print("=== [6/7] Building Master ZIP (Single-Pass Archive Generation) ===")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    clean_caches(STAGING_DIR)

    staging_all_files = sorted([f for f in STAGING_DIR.rglob("*") if f.is_file()])
    assert len(staging_all_files) == 227, f"Expected 227 total files in staging, found {len(staging_all_files)}"

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in staging_all_files:
            rel_p = f.relative_to(STAGING_DIR)
            arcname = Path("tbnnFoam-1.0.1") / rel_p
            z.write(f, arcname)

    final_zip_size = OUTPUT_ZIP.stat().st_size
    final_zip_size_mb = final_zip_size / (1024 * 1024)
    print(f"Created: {OUTPUT_ZIP} ({final_zip_size_mb:.2f} MB, {len(staging_all_files)} entries)")

    # Calculate and save external SHA-256 digest for the outer ZIP itself
    zip_sha = compute_sha256(OUTPUT_ZIP)
    zip_sha_file = RELEASE_DIR / f"{OUTPUT_ZIP.name}.sha256"
    zip_sha_file.write_text(f"{zip_sha}  {OUTPUT_ZIP.name}\n", encoding="utf-8", newline="\n")
    print(f"Generated external ZIP checksum: {zip_sha_file} (SHA256: {zip_sha})")

    # Extract and validate round-trip extraction & checksums
    print("=== [7/7] Validating Clean Extraction & Checksums ===")
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "r") as z:
        zip_entries = z.namelist()
        assert len(zip_entries) == 227, f"Expected 227 zip entries, found {len(zip_entries)}"
        z.extractall(RELEASE_DIR / "temp_extract")

    extracted_inner = RELEASE_DIR / "temp_extract" / "tbnnFoam-1.0.1"
    for item in extracted_inner.iterdir():
        shutil.move(str(item), str(EXTRACT_DIR / item.name))
    shutil.rmtree(RELEASE_DIR / "temp_extract")

    # Verify all 225 checksums on extracted files
    with open(EXTRACT_DIR / "CHECKSUMS.sha256", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exp_sha, rel_name = line.split("  ", 1)
            target_file = EXTRACT_DIR / Path(rel_name)
            assert target_file.exists(), f"Extracted file missing: {rel_name}"
            calc_sha = compute_sha256(target_file)
            assert calc_sha == exp_sha, f"Checksum mismatch for {rel_name}: expected {exp_sha}, got {calc_sha}"

    print(f"Extracted {len(zip_entries)} entries to {EXTRACT_DIR}")
    print(f"Verified all {len(checksum_lines)} cryptographic checksums on extracted files successfully!")

    # Clean intermediate staging directory
    if (RELEASE_DIR / "staging").exists():
        shutil.rmtree(RELEASE_DIR / "staging")
    print("Cleaned intermediate staging folder.")

if __name__ == "__main__":
    stage_release()
    verify_staged_models()
    run_staged_tests()
    generate_metadata_and_package()
    print("\n[SUCCESS] Software release generated successfully in release/")
