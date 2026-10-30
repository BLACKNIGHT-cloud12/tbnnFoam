#!/usr/bin/env python3


import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT_DIR / "publication_package"
OUTPUT_ZIP = ROOT_DIR / "tbnnFoam_Publication_Package.zip"

def create_table_csvs(tables_dir: Path):
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Pope Integrity Basis
    t1 = """Index,Tensor Basis T^(n),Algebraic Symmetry and Trace Property
1,"s","Symmetric, tr(T^(1)) = 0 (incompressible)"
2,"s*w - w*s","Symmetric, tr(T^(2)) = 0"
3,"s^2 - 1/3*tr(s^2)*I","Symmetric, trace-free"
4,"w^2 - 1/3*tr(w^2)*I","Symmetric, trace-free"
5,"w*s^2 - s^2*w","Symmetric, trace-free"
6,"w^2*s + s*w^2 - 2/3*tr(s*w^2)*I","Symmetric, trace-free"
7,"w*s*w^2 - w^2*s*w","Symmetric, trace-free"
8,"s*w*s^2 - s^2*w*s","Symmetric, trace-free"
9,"w^2*s^2 + s^2*w^2 - 2/3*tr(s^2*w^2)*I","Symmetric, trace-free"
10,"w*s^2*w^2 - w^2*s^2*w","Symmetric, trace-free"
"""
    (tables_dir / "Table1_Pope_Integrity_Basis.csv").write_text(t1, encoding="utf-8")

    # Table 2: Network Topology
    t2 = """Parameter,Value / Configuration,Physical / Numerical Purpose
Input Dimension,"5 features (lambda_1 ... lambda_5)","Cayley-Hamilton scalar invariants"
Hidden Layers,"3 layers x 64 neurons","Balanced capacity without overfitting"
Activation Function,"Rectified Linear Unit (ReLU)","phi(z) = max(0, z), standard piecewise linear activation"
Output Dimension,"10 coefficients (g^(1) ... g^(10))","Expansion weights for Pope bases T^(1 ... 10)"
Total Parameters,"9,354 trainable parameters","Exact layer count: 384 + 4160 + 4160 + 650 = 9,354"
Training Loss,"Mean Squared Error on a (all 9 tensor components)","L = (1 / 9N) * sum ||a_pred - a_DNS||_F^2"
"""
    (tables_dir / "Table2_Neural_Network_Topology.csv").write_text(t2, encoding="utf-8")

    # Table 3: Numerical Parity Verification
    t3 = """Comparison Component,Test Function,Tolerance,Condition / Environment,Status
PyTorch vs. TorchScript,"test_torchscript_parity","atol <= 1e-5","CPU via LibTorch runtime","Passed (10/10)"
PyTorch vs. C++ (g^(n)),"test_standalone_cpp","atol <= 1e-5","OpenFOAM / WSL environment","Passed (3/3)"
PyTorch vs. C++ (a),"test_standalone_cpp","atol <= 1e-5","OpenFOAM / WSL environment","Passed (3/3)"
Basis Trace Freedom,"test_basis_trace","atol <= 1e-12","Analytical traceless check","Passed"
Basis Symmetry,"test_basis_symmetry","atol <= 1e-12","Analytical symmetry check","Passed"
Cardano Realizability,"test_realizability","lambda_min >= -1/3","Perturbation matrices (N=50)","Guaranteed"
Coupling Boussinesq Limit,"test_momentum_coupling_parity","<= 1e-8 relative","Assembled fvMatrix, 2^3 k-omega SST mesh","Passed (3.6e-17)"
"""
    (tables_dir / "Table3_Numerical_Parity_Verification.csv").write_text(t3, encoding="utf-8")

    # Table 4: Canonical Square Duct Benchmarks
    t4 = """Regime / Case,Bulk Re_b,Re_h,Friction Re_tau,Fully Dev. Uc / Ub,Peak V_sec / Ub,Canonical Q1 Centroid,Benchmark Literature Source
Low-Re Primary Baseline,"4,410","2,205",150,1.32,1.93%,"(0.577, 0.847)","Pirozzoli et al. (2018) (raw export, verified)"
Intermediate Export Baseline,"7,000","3,500",220,1.30,2.03%,"(0.737, 0.896)","Pirozzoli et al. (2018) (raw export, verified)"
High-Re Asymptotic Benchmark,"17,800","8,900","~ 519",1.33,"2.03% - 2.30%","not exported","Literature"
"""
    (tables_dir / "Table4_Canonical_SquareDuct_Benchmarks.csv").write_text(t4, encoding="utf-8")

    # Table 5: Quantitative Comparison
    t5 = """Validation Metric,Standard OpenFOAM (k-omega SST),tbnnFoam (coarse native / DNS-grid interpolated),Reference DNS (Pirozzoli et al. 2018),Physical Interpretation
Secondary Cores,"0 cores (Failed)","8 cores (Detected)","8 cores (Canonical)","Recovers Prandtl's 8 cells"
Peak V_sec / Ub,"0.45% (small residual)","7.99% / 6.38%","1.93%","Non-zero secondary flow"
Q1 Core (y/h, z/h),"Undefined","(0.850, 0.350) / (0.91, 0.78)","(0.577, 0.847)","Cell center vs. interp. peak"
Core Distance Error,"N/A","0.567 h / 0.336 h","0.000 h (Ref)","Distance to raw-field-verified core"
Streamwise U RMSE,"0.0616 m/s (6.16%)","0.0934 m/s (9.34%)","0.0000 m/s","Full cross-section NRMSE"
Secondary (V, W) RMSE,"0.0087 m/s","0.0261 m/s","0.0000 m/s","In-plane cross-stream velocity"
Continuity Residual,"< 1e-6","< 1e-6","N/A","Steady-state convergence"
"""
    (tables_dir / "Table5_Validation_Comparison_Linear_vs_tbnnFoam_vs_DNS.csv").write_text(t5, encoding="utf-8")

    # Table 6: Discretization Sensitivity
    t6 = """Parameter / Metric,Coarse Mesh (M1),Refined Mesh (M2),Relative Delta,DNS Target (Pirozzoli et al. 2018)
Cross-Sectional Grid,"20 x 20 (400 cells)","40 x 40 (1600 cells)","4x Refinement","65 x 65 (DNS)"
Total Mesh Cells,"20,000 cells","80,000 cells","4x Refinement","N/A"
Peak V_sec / Ub,7.99%,2.62%,-67.2%,1.93%
Q1 Core (y/h, z/h),"(0.850, 0.350)","(0.825, 0.475)","Error reduction: -21.2%","(0.577, 0.847)"
Core Localization Error,0.567 h,0.447 h,"-0.120 h (-21.2%)","0.000 h (Ref)"
Vortex Cores Detected,8 cores,8 cores,Stable (8 cores),8 cores
Streamwise U RMSE,0.0934 m/s,0.0857 m/s,-8.2%,0.0000 m/s
"""
    (tables_dir / "Table6_Discretization_Sensitivity_Metrics.csv").write_text(t6, encoding="utf-8")

    # Table 7: High Reynolds Number Spatial Development
    t7 = """Station,x/Dh,Uc/Ub,Raw Max V_sec/Ub,Interior Max (|y|,|z| <= 0.85h),Cores,Q1 Dist. to low-Re DNS Ref.
#1,1.1,1.019,9.63%,1.91%,0 (Rollup),N/A
#2,2.1,1.037,10.43%,2.80%,0 (Rollup),N/A
#3,3.1,1.054,10.43%,2.39%,0 (Rollup),N/A
#4,4.1,1.069,10.20%,1.95%,0 (Rollup),N/A
#5,5.1,1.083,9.92%,1.87%,0 (Rollup),N/A
#6,6.1,1.096,9.69%,1.84%,8 cores,0.056 h
#7,7.1,1.108,9.55%,1.78%,8 cores,0.028 h
#8,8.1,1.121,9.34%,1.70%,8 cores,0.028 h
#9,9.1,1.132,9.39%,1.67%,8 cores,0.059 h (1.2 cells)
DNS Ref (low-Re),infinity,1.33,"2.03% - 2.30%","2.03% - 2.30%",8 cores,"(0.577, 0.847)"
"""
    (tables_dir / "Table7_Re17800_Streamwise_Development.csv").write_text(t7, encoding="utf-8")

    # Table 8: Marginal Bounds
    t8 = """Feature / Invariant,DNS Training Envelope,CFD Operating Range (20k Cells),Marginal Coverage
"I_1 = tr(s^2)","[0.000, 0.933]","[2.927e-4, 0.0690]","100.0% In-Bounds"
"I_2 = tr(w^2)","[-0.932, 0.000]","[-0.0682, -1.118e-10]","100.0% In-Bounds"
"I_3 = tr(s^3)","[-0.0349, 0.00426]","[-4.853e-3, 4.505e-3]","99.0% In-Bounds"
"I_4 = tr(w^2*s)","[-0.00142, 0.0116]","[-1.453e-3, 1.546e-3]","99.8% In-Bounds"
"I_5 = tr(w^2*s^2)","[-0.435, 0.000]","[-2.350e-3, -4.317e-14]","100.0% In-Bounds"
"Basis Norms ||T^(n)||_F","[0.0, 1.318]","[0.0, 0.243]","100.0% Bounded"
"""
    (tables_dir / "Table8_Cayley_Hamilton_Marginal_Bounds.csv").write_text(t8, encoding="utf-8")

    # Markdown Summary
    md_content = """# Supplementary Tables: tbnnFoam Manuscript

This document compiles the quantitative tables accompanying the preprint:
*In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM: A Critical Verification, Validation, and Reproducibility Study*
(Soumik Ghosh, October 2026).

---

### Table 1: Pope's (1975) Ten-Tensor Integrity Basis
| Index $n$ | Tensor Basis $\\mathbf{T}^{(n)}$ | Algebraic Symmetry & Trace Property |
| :---: | :--- | :--- |
| 1 | $\\mathbf{s}$ | Symmetric, $\\mathrm{tr}(\\mathbf{T}^{(1)}) = 0$ (incompressible) |
| 2 | $\\mathbf{s}\\mathbf{w} - \\mathbf{w}\\mathbf{s}$ | Symmetric, $\\mathrm{tr}(\\mathbf{T}^{(2)}) = 0$ |
| 3 | $\\mathbf{s}^2 - \\frac{1}{3}\\mathrm{tr}(\\mathbf{s}^2)\\mathbf{I}$ | Symmetric, trace-free |
| 4 | $\\mathbf{w}^2 - \\frac{1}{3}\\mathrm{tr}(\\mathbf{w}^2)\\mathbf{I}$ | Symmetric, trace-free |
| 5 | $\\mathbf{w}\\mathbf{s}^2 - \\mathbf{s}^2\\mathbf{w}$ | Symmetric, trace-free |
| 6 | $\\mathbf{w}^2\\mathbf{s} + \\mathbf{s}\\mathbf{w}^2 - \\frac{2}{3}\\mathrm{tr}(\\mathbf{s}\\mathbf{w}^2)\\mathbf{I}$ | Symmetric, trace-free |
| 7 | $\\mathbf{w}\\mathbf{s}\\mathbf{w}^2 - \\mathbf{w}^2\\mathbf{s}\\mathbf{w}$ | Symmetric, trace-free |
| 8 | $\\mathbf{s}\\mathbf{w}\\mathbf{s}^2 - \\mathbf{s}^2\\mathbf{w}\\mathbf{s}$ | Symmetric, trace-free |
| 9 | $\\mathbf{w}^2\\mathbf{s}^2 + \\mathbf{s}^2\\mathbf{w}^2 - \\frac{2}{3}\\mathrm{tr}(\\mathbf{s}^2\\mathbf{w}^2)\\mathbf{I}$ | Symmetric, trace-free |
| 10 | $\\mathbf{w}\\mathbf{s}^2\\mathbf{w}^2 - \\mathbf{w}^2\\mathbf{s}^2\\mathbf{w}$ | Symmetric, trace-free |

---

### Table 2: Neural Network Topology & Hyperparameters
| Parameter | Value / Configuration | Physical / Numerical Purpose |
| :--- | :--- | :--- |
| **Input Dimension** | 5 features ($\\lambda_1 \\dots \\lambda_5$) | Cayley-Hamilton scalar invariants |
| **Hidden Layers** | 3 layers $\\times$ 64 neurons | Balanced capacity without overfitting |
| **Activation Function** | Rectified Linear Unit (ReLU) | $\\phi(z) = \\max(0, z)$, standard piecewise linear activation |
| **Output Dimension** | 10 coefficients ($g^{(1)} \\dots g^{(10)}$) | Expansion weights for Pope bases $\\mathbf{T}^{(1 \\dots 10)}$ |
| **Total Parameters** | 9,354 trainable parameters | Exact layer count: $384 + 4160 + 4160 + 650 = 9,354$ |
| **Training Loss** | Mean Squared Error on $\\mathbf{a}$ | $\\mathcal{L} = \\frac{1}{9N}\\sum_{i=1}^N \\|\\mathbf{a}_{\\text{pred}, i} - \\mathbf{a}_{\\text{DNS}, i}\\|_F^2$ |

---

### Table 3: Numerical Parity Verification Across Engines
| Comparison Component | Test Function | Tolerance | Condition / Environment | Status |
| :--- | :--- | :---: | :--- | :---: |
| **PyTorch vs. TorchScript** | `test_torchscript_parity` | $\\mathrm{atol} \\le 10^{-5}$ | CPU via LibTorch runtime | **Passed (10/10)** |
| **PyTorch vs. C++ ($g^{(n)}$)** | `test_standalone_cpp` | $\\mathrm{atol} \\le 10^{-5}$ | OpenFOAM / WSL environment | **Passed (3/3)** |
| **PyTorch vs. C++ ($\\mathbf{a}$)** | `test_standalone_cpp` | $\\mathrm{atol} \\le 10^{-5}$ | OpenFOAM / WSL environment | **Passed (3/3)** |
| **Basis Trace Freedom** | `test_basis_trace` | $\\mathrm{atol} \\le 10^{-12}$ | Analytical traceless check | **Passed** |
| **Basis Symmetry** | `test_basis_symmetry` | $\\mathrm{atol} \\le 10^{-12}$ | Analytical symmetry check | **Passed** |
| **Cardano Realizability** | `test_realizability` | $\\lambda_{\\min} \\ge -1/3$ | Perturbation matrices ($N=50$) | **Guaranteed** |
| **Coupling Boussinesq Limit** | `test_momentum_coupling_parity` | $\\le 10^{-8}$ relative | Assembled fvMatrix, $2^3$ $k\\text{--}\\omega$ SST | **Passed ($3.6 \\times 10^{-17}$)** |

---

### Table 4: Canonical Benchmark DNS Reference Parameters
| Regime / Case | Bulk $Re_b$ | $Re_h$ | Friction $Re_\\tau$ | Fully Dev. $U_c / U_b$ | Peak $V_{\\text{sec}}/U_b$ | Canonical Q1 Centroid | Benchmark Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low-$Re$ Primary Baseline** | 4,410 | 2,205 | 150 | 1.32 | 1.93% | (0.577, 0.847) | Pirozzoli et al. (2018) raw export |
| **Intermediate Export Baseline** | 7,000 | 3,500 | 220 | 1.30 | 2.03% | (0.737, 0.896) | Pirozzoli et al. (2018) raw export |
| **High-$Re$ Asymptotic Benchmark** | 17,800 | 8,900 | $\\approx 519$ | 1.33 | 2.03% - 2.30% | not exported | Literature |

---

### Table 5: Comprehensive Quantitative Comparison
| Validation Metric | Standard OpenFOAM ($k\\text{--}\\omega$ SST) | `tbnnFoam` (coarse / interp) | Reference DNS | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Cores** | 0 cores (Failed) | **8 cores (Detected)** | 8 cores (Canonical) | Recovers Prandtl's 8 cells |
| **Peak $V_{\\text{sec}} / U_b$** | 0.45% (residual) | **7.99% / 6.38%** | 1.93% | Non-zero secondary flow |
| **Q1 Core $(y/h, z/h)$** | Undefined | **(0.850, 0.350) / (0.91, 0.78)** | (0.577, 0.847) | Cell center vs. interp. peak |
| **Core Distance Error** | N/A | **0.567 h / 0.336 h** | 0.000 h (Ref) | Distance to verified core |
| **Streamwise $U$ RMSE** | 0.0616 m/s (6.16%) | **0.0934 m/s (9.34%)** | 0.0000 m/s | Full cross-section NRMSE |
| **Secondary $(V, W)$ RMSE** | 0.0087 m/s | **0.0261 m/s** | 0.0000 m/s | In-plane secondary velocity |
| **Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Steady-state convergence |

---

### Table 6: Cross-Stream Discretization Sensitivity ($20 \\times 20$ vs. $40 \\times 40$)
| Parameter / Metric | Coarse Mesh (M1) | Refined Mesh (M2) | Relative Delta | DNS Target |
| :--- | :---: | :---: | :---: | :---: |
| **Cross-Sectional Grid** | $20 \\times 20$ (400 cells) | $40 \\times 40$ (1600 cells) | 4$\\times$ Refinement | $65 \\times 65$ (DNS) |
| **Total Mesh Cells** | 20,000 cells | 80,000 cells | 4$\\times$ Refinement | N/A |
| **Peak $V_{\\text{sec}} / U_b$** | 7.99% | 2.62% | -67.2% | **1.93%** |
| **Q1 Core $(y/h, z/h)$** | (0.850, 0.350) | (0.825, 0.475) | Error reduction: -21.2% | **(0.577, 0.847)** |
| **Core Localization Error** | 0.567 h | 0.447 h | -0.120 h (-21.2%) | **0.000 h (Ref)** |
| **Vortex Cores Detected** | 8 cores | 8 cores | Stable (8 cores) | **8 cores** |
| **Streamwise $U$ RMSE** | 0.0934 m/s | 0.0857 m/s | -8.2% | 0.0000 m/s |

---

### Table 7: Streamwise Spatial Development at $Re_b = 17,800$
| Station | $x/D_h$ | $U_c/U_b$ | Raw Max $V_{\\text{sec}}/U_b$ | Interior Max ($|y|,|z| \\le 0.85h$) | Cores | Q1 Dist. to low-$Re$ DNS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | 1.1 | 1.019 | 9.63% | 1.91% | 0 (Rollup) | N/A |
| #2 | 2.1 | 1.037 | 10.43% | 2.80% | 0 (Rollup) | N/A |
| #3 | 3.1 | 1.054 | 10.43% | 2.39% | 0 (Rollup) | N/A |
| #4 | 4.1 | 1.069 | 10.20% | 1.95% | 0 (Rollup) | N/A |
| #5 | 5.1 | 1.083 | 9.92% | 1.87% | 0 (Rollup) | N/A |
| #6 | 6.1 | 1.096 | 9.69% | 1.84% | 8 cores | $0.056\\,h$ |
| #7 | 7.1 | 1.108 | 9.55% | 1.78% | 8 cores | $0.028\\,h$ |
| #8 | 8.1 | 1.121 | 9.34% | 1.70% | 8 cores | $0.028\\,h$ |
| #9 | 9.1 | 1.132 | 9.39% | 1.67% | 8 cores | $0.059\\,h$ (1.2 cells) |
| **DNS Ref (low-$Re$)** | $\\infty$ | 1.33 | 2.03% - 2.30% | 2.03% - 2.30% | 8 cores | **(0.577, 0.847)** |

---

### Table 8: Empirical Bounds and Marginal Coverage of Invariants
| Feature / Invariant | DNS Training Envelope | CFD Operating Range (20k Cells) | Marginal Coverage |
| :--- | :---: | :---: | :---: |
| $I_1 = \\mathrm{tr}(\\mathbf{s}^2)$ | $[0.000, 0.933]$ | $[2.927 \\times 10^{-4}, 0.0690]$ | **100.0% In-Bounds** |
| $I_2 = \\mathrm{tr}(\\mathbf{w}^2)$ | $[-0.932, 0.000]$ | $[-0.0682, -1.118 \\times 10^{-10}]$ | **100.0% In-Bounds** |
| $I_3 = \\mathrm{tr}(\\mathbf{s}^3)$ | $[-0.0349, 0.00426]$ | $[-4.853 \\times 10^{-3}, 4.505 \\times 10^{-3}]$ | **99.0% In-Bounds** |
| $I_4 = \\mathrm{tr}(\\mathbf{w}^2\\mathbf{s})$ | $[-0.00142, 0.0116]$ | $[-1.453 \\times 10^{-3}, 1.546 \\times 10^{-3}]$ | **99.8% In-Bounds** |
| $I_5 = \\mathrm{tr}(\\mathbf{w}^2\\mathbf{s}^2)$ | $[-0.435, 0.000]$ | $[-2.350 \\times 10^{-3}, -4.317 \\times 10^{-14}]$ | **100.0% In-Bounds** |
| **Basis Norms** $\\|\\mathbf{T}^{(n)}\\|_F$ | $[0.0, 1.318]$ | $[0.0, 0.243]$ | **100.0% Bounded** |
"""
    (tables_dir / "supplementary_tables.md").write_text(md_content, encoding="utf-8")


def build_package():
    print("[1/6] Cleaning existing publication package directory...")
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("[2/6] Copying primary publication items (PDF, README, LICENSE)...")
    # Main PDF
    preprint_pdf = ROOT_DIR / "docs" / "tbnnFoam_preprint.pdf"
    if not preprint_pdf.exists():
        preprint_pdf = ROOT_DIR / "docs" / "latex" / "main.pdf"
    shutil.copy2(preprint_pdf, PACKAGE_DIR / "tbnnFoam_preprint.pdf")

    # README and LICENSE
    shutil.copy2(ROOT_DIR / "README.md", PACKAGE_DIR / "README.md")
    shutil.copy2(ROOT_DIR / "LICENSE", PACKAGE_DIR / "LICENSE")
    if (ROOT_DIR / ".zenodo.json").exists():
        shutil.copy2(ROOT_DIR / ".zenodo.json", PACKAGE_DIR / ".zenodo.json")
    if (ROOT_DIR / "CITATION.cff").exists():
        shutil.copy2(ROOT_DIR / "CITATION.cff", PACKAGE_DIR / "CITATION.cff")

    print("[3/6] Generating clean source code ZIP via git archive...")
    source_zip_path = PACKAGE_DIR / "tbnnFoam_source_code.zip"
    cmd = ["git", "archive", "--format=zip", "--prefix=tbnnFoam/", "HEAD", "-o", str(source_zip_path)]
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    print(f"       -> Generated source ZIP: {source_zip_path.stat().st_size / (1024*1024):.2f} MB")

    print("[4/6] Assembling supplementary materials (figures, tables, reports, benchmarks)...")
    supp_dir = PACKAGE_DIR / "supplementary"
    supp_figures = supp_dir / "figures"
    supp_tables = supp_dir / "tables"
    supp_reports = supp_dir / "reports"
    supp_benchmarks = supp_dir / "benchmarks"

    supp_figures.mkdir(parents=True, exist_ok=True)
    supp_reports.mkdir(parents=True, exist_ok=True)
    supp_benchmarks.mkdir(parents=True, exist_ok=True)

    # Figures
    src_figures = ROOT_DIR / "docs" / "figures"
    if src_figures.exists():
        for item in src_figures.iterdir():
            if item.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".pdf"]:
                shutil.copy2(item, supp_figures / item.name)

    # Tables
    create_table_csvs(supp_tables)

    # Reports
    src_reports = ROOT_DIR / "docs" / "report"
    if src_reports.exists():
        for rep in ["tbnnFoam_Technical_Report.pdf", "tbnnFoam_Project_Summary.pdf"]:
            rep_path = src_reports / rep
            if rep_path.exists():
                shutil.copy2(rep_path, supp_reports / rep)

    # Benchmarks
    src_benchmarks = ROOT_DIR / "data" / "benchmark_dns"
    if src_benchmarks.exists():
        for b_item in ["square_duct_dns_Reb4410_Retau150.csv", "square_duct_dns_Reb7000_Retau220.csv", "benchmark_metadata.json"]:
            b_path = src_benchmarks / b_item
            if b_path.exists():
                shutil.copy2(b_path, supp_benchmarks / b_item)

    print("[5/6] Writing package manifest...")
    manifest = """tbnnFoam Publication Package Manifest
======================================
This package contains all assets required for scientific publication, archival (Zenodo/arXiv/institutional), and independent reproducibility:

1. tbnnFoam_preprint.pdf:
   Authoritative manuscript draft: "In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM: A Critical Verification, Validation, and Reproducibility Study" (October 2026).

2. README.md & LICENSE:
   Complete project documentation, installation instructions, quick-start guide, and MIT open-source license.

3. tbnnFoam_source_code.zip:
   Full, clean snapshot of the solver source code, test suites, Python training & export pipelines, pre-trained neural network weights, and verification case setups.

4. supplementary/:
   - figures/: High-resolution PNG figures (architecture, barycentric map, mesh sensitivity, showcase, timescale distribution) and animated GIFs (secondary vortex development, 3D streamlines, solver convergence).
   - tables/: Machine-readable CSV files (Tables 1 through 8) and a comprehensive supplementary_tables.md summary.
   - reports/: High-fidelity Technical Report and Project Summary PDFs.
   - benchmarks/: Authoritative DNS benchmark datasets (Pirozzoli et al. 2018) at Reb = 4,410 (Retau = 150) and Reb = 7,000 (Retau = 220) plus metadata.
"""
    (PACKAGE_DIR / "MANIFEST.txt").write_text(manifest, encoding="utf-8")

    print(f"[6/6] Creating final master archive: {OUTPUT_ZIP.name}...")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PACKAGE_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = Path("tbnnFoam_Publication_Package") / file_path.relative_to(PACKAGE_DIR)
                zipf.write(file_path, arcname)

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"SUCCESS! Publication Package generated successfully.")
    print(f"Local Folder : {PACKAGE_DIR}")
    print(f"Master ZIP   : {OUTPUT_ZIP} ({size_mb:.2f} MB)")
    print(f"=======================================================")

if __name__ == "__main__":
    build_package()
