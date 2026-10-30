# Benchmark DNS Data Reference Guide

This directory contains benchmark configuration metadata for the turbulent square duct flow studies associated with **tbnnFoam**.

To maintain clean open-source licensing boundaries for the software deposit, proprietary third-party Direct Numerical Simulation (DNS) raw and converted data files are not redistributed in the release package. All solver implementations, trained neural network weights, validation algorithms, and OpenFOAM simulation cases remain 100% self-contained and reproducible.

---

## 1. Provenance & Citations

The benchmark DNS data for turbulent square duct flow was produced by the Sapienza University of Rome CFD Group:

* **Primary Citation (Re_tau = 150 & Re_tau = 220):**
  > Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018). *Turbulence and secondary motions in square duct flow.* **Journal of Fluid Mechanics**, 840, 634–655.
* **Secondary Citation (Aspect Ratio DNS):**
  > Vinuesa, R., Noorani, A., Lozano-Durán, A., El Khoury, G. K., Schlatter, P., Fischer, P. F., & Nagib, H. M. (2014). *Aspect ratio effects in turbulent duct flows studied through direct numerical simulation.* **Journal of Turbulence**, 15(10), 677–706.
* **Upstream Online Repository:**
  Sapienza University of Rome CFD Group DNS Database:  
  [http://newton.dma.uniroma1.it/square_duct/](http://newton.dma.uniroma1.it/square_duct/)

---

## 2. When Do You Need the DNS Data?

* **NOT needed for running OpenFOAM simulations:** The shipped solvers (`tbnnFoam`, `tbnnSimpleFoam`) and verification cases (`verification_squareDuct`, `verification_squareDuct_Re17800`) run out-of-the-box using the pre-trained production model weights (`tbnn_model.pt`, `export_weights/weights.json`).
* **NOT needed for automated test suites:** The test suite (`pytest tests/`) validates tensor mathematical properties, realizability bounds, cross-engine numerical parity, and vortex topological core recovery using project-authored synthetic fixtures and `benchmark_metadata.json`.
* **ONLY needed if you want to:**
  1. Re-train the Tensor Basis Neural Network from scratch (`tbnn_train.py`).
  2. Compute direct point-by-point velocity field RMSEs ($L_2$ differences against DNS fields).
  3. Re-run raw discretization verification checks (`scripts/validation/verify_dns_reference.py`).

---

## 3. How to Obtain and Prepare the Benchmark Datasets

### Step A: Download the Raw DNS Exports
Download the raw ASCII slice exports from the Sapienza University CFD Group repository:
1. `plotyz_Retau150.dat` ($Re_b = 4,410$, $Re_\tau = 150$)
2. `plotyz_Retau220.dat` ($Re_b = 7,000$, $Re_\tau = 220$)

Place the downloaded files into:
```
data/benchmark_dns/raw_dns/
```

### Step B: Run the Conversion Pipeline
Run the included preprocessing script to symmetry-mirror, bilinearly resample to a $65 \times 65$ cross-section, compute mean strain and rotation rate tensors, and generate standardized CSV files:
```bash
python scripts/convert_real_dns.py
```
This automatically produces:
* `data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv`
* `data/benchmark_dns/square_duct_dns_Reb7000_Retau220.csv`

---

## 4. Supplying an External DNS Directory

All validation, testing, and training scripts accept an externally supplied DNS directory or file path without requiring files to be placed inside the repository tree.

### Using Command-Line Arguments:
```bash
# Quantitative physics validation report:
python scripts/validation/evaluate_validation_plan.py --dns-dir /path/to/external_dns_data

# Cross-sectional mesh sensitivity study:
python scripts/validation/evaluate_mesh_sensitivity.py --dns-dir /path/to/external_dns_data

# Time-scale and invariant distribution shift diagnostic:
python scripts/validation/evaluate_timescale_distribution.py --dns-dir /path/to/external_dns_data

# Raw DNS reference discretization check:
python scripts/validation/verify_dns_reference.py --raw-dir /path/to/raw_dns_data

# Verification case validation and plotting:
python verification_squareDuct/validate_and_plot.py --dns-dir /path/to/external_dns_data
```

### Using an Environment Variable:
Alternatively, set the `TBNN_DNS_DIR` environment variable:
```bash
# Linux / macOS / WSL:
export TBNN_DNS_DIR="/path/to/external_dns_data"

# Windows PowerShell:
$env:TBNN_DNS_DIR = "C:\path\to\external_dns_data"
```
When `TBNN_DNS_DIR` is set, all validation scripts and test suites automatically detect and load the benchmark files.

---

## 5. Machine-Readable Metadata

The canonical physical constants, scaling relationships, Reynolds number definitions, and literature target vortex coordinates are formally defined in:
* [`benchmark_metadata.json`](benchmark_metadata.json)

Validation scripts use `benchmark_metadata.json` to verify topological secondary recirculation and corner vortex core locations even when external DNS raw CSVs are not present.
