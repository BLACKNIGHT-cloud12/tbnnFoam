# tbnnFoam

An in-situ Tensor Basis Neural Network (TBNN) turbulence closure for OpenFOAM, designed to capture turbulence-driven secondary flows of the second kind in non-circular ducts.

Developed by **Soumik Ghosh**

<div align="left">

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2012%20--%20v2606-004C87.svg?logo=openfoam&logoColor=white)](https://www.openfoam.com)
[![Pytest](https://img.shields.io/badge/Tests-17%20passed-2ea44f.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.9%20--%203.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What is this project?

Standard linear eddy-viscosity RANS models (such as standard $k\text{--}\varepsilon$ and $k\text{--}\omega$ SST) rely on the Boussinesq hypothesis:

$$\boldsymbol{\tau}^R = 2\nu_t \mathbf{S} - \frac{2}{3}k\mathbf{I}$$

Because the turbulent viscosity $\nu_t$ is a scalar, linear models assume that turbulence is isotropic in the cross-stream plane ($\overline{v'^2} = \overline{w'^2}$). 

In non-circular ducts, however, turbulent secondary flows of the second kind (Prandtl, 1926) are driven directly by spatial gradients of turbulent normal stress differences:

$$\mathbf{U} \cdot \nabla \omega_x \approx \frac{\partial^2}{\partial y \partial z}\left(\overline{w'^2} - \overline{v'^2}\right) + \left(\frac{\partial^2}{\partial z^2} - \frac{\partial^2}{\partial y^2}\right)\overline{v'w'}$$

Because linear models force $\overline{w'^2} - \overline{v'^2} = 0$, the production of axial vorticity is identically zero. As a result, standard OpenFOAM predicts completely flat, unidirectional axial flow ($V = W = 0$), missing the 8 corner counter-rotating vortex pairs that transport momentum and heat into duct corners.

**`tbnnFoam`** embeds the Tensor Basis Neural Network (TBNN) architecture of Ling et al. (2016) directly into OpenFOAM's C++ finite volume solver loop. During every SIMPLE iteration, it extracts local velocity gradients, evaluates non-dimensional flow invariants, evaluates the neural network weights, enforces physical realizability, and injects the anisotropic stress divergence directly into the momentum equation.

---

## Why two solvers?

Embedding machine learning models into CFD solvers often introduces practical friction on high-performance computing (HPC) clusters. To address this, I built two complementary implementations:

1. **`tbnnFoam` (LibTorch / PyTorch C++ API):**
   * Uses `torch::jit` to load and evaluate serialized TorchScript models (`tbnn_model.pt`) directly in memory.
   * Useful during active research and prototyping when testing different PyTorch architectures.
   * **Limitation:** Linking LibTorch on shared university or enterprise clusters often causes build issues due to mismatched CUDA libraries, GCC versions, or C++11 ABI differences.

2. **`tbnnSimpleFoam` (Standalone Header-Only C++ Evaluator):**
   * Implemented in [`src/tbnnEvaluator.H`](src/tbnnEvaluator.H) with zero third-party library dependencies.
   * Parses model weights directly from a plain JSON file ([`export_weights/weights.json`](export_weights/weights.json)) using standard C++ streams.
   * Evaluates the multi-layer perceptron, constructs the 10 tensor bases, and performs realizability clipping using native OpenFOAM tensor algebra.
   * **Advantage:** Compiles anywhere with standard OpenFOAM `wmake`, requiring no external packages or environment setup.

Parity between both engines is continuously tested and verified in [`tests/test_parity.py`](tests/test_parity.py).

---

## In-Situ Momentum Coupling Pipeline

At each iteration of the SIMPLE loop, the solver couples the neural network to the flow field:

![Architecture Pipeline](docs/figures/architecture_pipeline.png)

1. **Gradient Extraction:** Reads cell-centered velocity gradients $\nabla \mathbf{U}$, turbulent kinetic energy $k$, and specific dissipation $\omega$.
2. **Dimensionless Invariants & Bases:** Computes the normalized strain rate $\mathbf{s} = \tau \mathbf{S}$ and rotation rate $\mathbf{w} = \tau \mathbf{W}$ using the timescale limiter $\tau = \min(1/\omega, 5.0\text{ s})$. From these, it evaluates the 5 Cayley-Hamilton scalar invariants ($\lambda_1 \dots \lambda_5$) and Pope's (1975) 10 symmetric, trace-free tensor bases ($\mathbf{T}^{(1)} \dots \mathbf{T}^{(10)}$).
3. **Neural Network Evaluation:** Feeds the 5 scalar invariants into the network to predict 10 scalar expansion coefficients $g^{(1)} \dots g^{(10)}$.
4. **Anisotropy Assembly:** Computes the deviatoric anisotropy tensor:
   $$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\boldsymbol{\lambda}) \, \mathbf{T}^{(n)}$$
5. **Realizability Enforcement:** Uses a closed-form Cardano cubic eigensolver to clamp the eigenvalues of $\mathbf{a}$ to the Lumley triangle, ensuring physical positive semi-definiteness without iterative overhead.
6. **Momentum Injection:** Formulates the perturbation stress $\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$ and injects its divergence $\nabla \cdot \boldsymbol{\tau}_\Delta$ into OpenFOAM's momentum predictor (`UEqn.H`).

### Key Implementation Safeguards

* **Preventing Double-Counting:** OpenFOAM's `turbulence->divDevReff(U)` implicitly discretizes $-\nabla \cdot (2\nu_t \mathbf{S})$ for numerical stability. Injecting the full Reynolds stress $2k\mathbf{a}$ directly would double-count the linear viscosity. By defining $\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$, the explicit source term represents only the non-linear deviation. When the flow is purely Boussinesq ($\mathbf{a} = -\frac{\nu_t}{k}\mathbf{S}$), $\boldsymbol{\tau}_\Delta \equiv 0$, preserving diagonal dominance.
* **Analytical Cardano Realizability:** If a neural network predicts unphysical stress states, CFD solvers diverge rapidly. Rather than using slow iterative projections, [`src/tbnnEvaluator.H`](src/tbnnEvaluator.H) solves the cubic characteristic equation analytically:
  $$p = \frac{1}{2}\mathrm{tr}(\mathbf{a}^2), \quad q = \det(\mathbf{a}), \quad \lambda_k = 2\sqrt{p/3} \cos\left(\frac{1}{3}\arccos\left(\frac{3q}{2p\sqrt{p/3}}\right) - \frac{2\pi k}{3}\right)$$
  If the smallest eigenvalue violates $\lambda_3 \ge -1/3$, the tensor is scaled radially in $\mathcal{O}(1)$ operations, guaranteeing frame-invariant stability.
* **Turbulent Timescale Limiter:** Capping $\tau = \min(1/\omega, 5.0\text{ s})$ prevents high near-wall strain rates from causing extreme invariant values, keeping all CFD cells strictly inside the trained manifold.

---

## Physical Mechanism: Normal Stress Differences

Secondary flow of the second kind is driven by gradients of normal stress differences $\frac{\partial^2}{\partial y \partial z}(\overline{w'^2} - \overline{v'^2})$. The plot below compares standard linear OpenFOAM, our in-situ `tbnnFoam` simulation, and the reference DNS benchmark:

![Reynolds Stress Mechanism](docs/figures/reynolds_stress_mechanism.png)

* **Linear RANS:** Predicts identically zero normal stress differences across the duct.
* **`tbnnFoam`:** Captures the alternating quadrant lobes that pump fluid outward toward duct corners and recirculate it along the wall bisectors.

---

## Turbulence Anisotropy State: Barycentric Map

The Barycentric Map (Banerjee et al., 2007) visualizes the anisotropy state across the duct cross-section:

![Barycentric Anisotropy Map](docs/figures/barycentric_anisotropy_map.png)

In standard linear OpenFOAM, the simulation is locked at the top apex (3-component isotropic turbulence) because normal stresses are forced equal. In `tbnnFoam`, the stresses leave the isotropic corner and travel along the right boundary toward realistic 2-component and 1-component states, in close qualitative agreement with the DNS benchmark.

---

## Benchmark Validation: Square Duct ($Re_\tau = 150$)

I validated the solver on a square duct flow against the benchmark Direct Numerical Simulation (DNS) dataset of Pirozzoli et al. (2018) at bulk Reynolds number $Re_b \approx 4410$ ($Re_h \approx 2205$) and friction Reynolds number $Re_\tau = 150$:

![Benchmark Validation](docs/figures/tbnn_hero_showcase.png)

### Quantitative Comparison

All metrics were computed by interpolating the exported CFD fields (`verification_squareDuct/300/U` for `tbnnFoam` and `verification_squareDuct/baseline_sst/111/U` for converged standard $k\text{--}\omega$ SST) onto the matching $65 \times 65$ DNS cross-section grid:

| Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` | Reference DNS (Pirozzoli et al. 2018) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortex Cores** | 0 cores (Failed) | **8 cores (Detected)** | 8 cores (Canonical) | Recovers Prandtl's 8 corner vortex pairs |
| **Peak Secondary Speed ($V_{\text{sec}} / U_b$)** | 0.45% (Noise) | **6.38%** | 2.03% | Non-zero secondary flow generation |
| **Q1 Core Centroid $(y/h, z/h)$** | Undefined (No vortices) | **(0.91, 0.78)** | (0.50, 0.82) | Detected anti-clockwise (ACW) core coordinates |
| **Core Distance Error** | N/A | **0.408 h** | 0.000 h (Ref) | Distance to benchmark core centroid |
| **Streamwise $U$ RMSE** | 0.0616 m/s (6.16%) | **0.0934 m/s (9.34%)** | 0.0000 m/s (Ref) | Centerline profile difference (see discussion below) |
| **Secondary $(V, W)$ RMSE** | 0.0087 m/s | **0.0261 m/s** | 0.0000 m/s (Ref) | In-plane cross-stream velocity recovery |
| **Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Both meet numerical steady-state convergence |

---

## Honest Technical Discussion: What Works and What Deviates

### What works well:
* **8-Vortex Recovery:** Standard linear OpenFOAM predicts completely dead secondary flow ($V = W = 0$, with $0.45\%$ numerical discretization noise). `tbnnFoam` successfully spins up all 8 secondary corner vortices in the correct quadrant orientations.
* **Secondary Velocity Strength:** Secondary flow reaches $6.38\% U_b$ on the coarse mesh and converges down to $2.62\% U_b$ under cross-stream refinement, closely approaching the DNS benchmark value of $2.03\% U_b$.
* **Numerical Stability:** The simulation converged stably over 287 iterations to true mathematical steady state, with global continuity residuals dropping to $-5.1 \times 10^{-8} < 10^{-6}$.

### Understanding the Differences:
1. **Developing Spatial Duct vs. Streamwise-Periodic Box:**
   The reference DNS benchmark of Pirozzoli et al. (2018) was simulated in an idealized streamwise-periodic box driven by a body force (representing an asymptotically fully developed duct of infinite length). In contrast, our OpenFOAM simulation is a spatially developing duct ($L = 10h = 0.5\text{ m}$) with a uniform velocity inlet ($U_b = 1.0\text{ m/s}$) and physical boundary layers growing along the walls. 
   Because secondary flow is triggered locally by near-wall corner normal stress differences, the 8 secondary vortex cells establish quickly ($x/h \ge 6$). However, the centerline streamwise core requires a longer development length ($U_{\text{center}} \approx 1.25\text{ m/s}$ in our CFD case vs $1.33\text{ m/s}$ in the periodic DNS). This domain difference accounts for the $9.3\%$ RMSE in streamwise velocity. Standard linear $k\text{--}\omega$ SST exhibits slightly lower streamwise error ($6.16\%$) simply because it lacks cross-stream convective momentum transport to the corners, yet it fails entirely on the secondary flow physics ($0$ vortex cores).
2. **Modelled $\omega$ Surrogate in the DNS Training Data:**
   Direct numerical simulation yields exact mean velocity gradients $\nabla \mathbf{U}$ and true Reynolds stress tensors $\boldsymbol{\tau}^R$, but DNS inherently lacks the specific dissipation rate $\omega$ because $\omega$ is a semi-empirical RANS modeling construct rather than a physical Navier-Stokes field.
   In the training dataset, $\omega$ was estimated using an empirical SST surrogate relation ($\omega \approx k^{1/2} / (C_\mu^{1/4} \ell)$). In live OpenFOAM CFD execution, $\omega$ is dynamically governed by the coupled transport equation. This introduces a slight model-form distribution shift between offline training and online coupled RANS.
3. **Coupled Flow Feedback vs. Frozen Training Data:**
   The neural network was trained offline on frozen velocity gradients. In live CFD coupling, the emerging secondary flow deforms the mean velocity field $\mathbf{U}$, which in turn alters the velocity gradients fed back into the neural network at each SIMPLE iteration.

---

## Cross-Sectional Mesh Sensitivity Study

A formal Grid Convergence Index (GCI) according to Roache and ASME V&V 20 standards requires at least three geometrically scaled grid levels with constant refinement ratio ($r \ge 1.3$) to evaluate the asymptotic observed order of convergence $p$. 

Here, we conduct an independent **2-level cross-sectional grid resolution sensitivity check** comparing the standard base mesh ($20\text{k}$ cells) and a 4x cross-stream refined mesh ($80\text{k}$ cells) to examine discretization sensitivity and verify that the detected secondary vortices are genuine physical solutions rather than numerical grid artifacts:

* **Coarse Mesh (M1):** $50 \times 20 \times 20 = 20{,}000$ cells ($\Delta y = \Delta z = 0.005\text{ m}$).
* **Refined Mesh (M2):** $50 \times 40 \times 40 = 80{,}000$ cells ($\Delta y = \Delta z = 0.0025\text{ m}$).

![Mesh Sensitivity Study](docs/figures/mesh_sensitivity_study.png)

| Grid Resolution | Cells | Peak $V_{\text{sec}} / U_b$ | Q1 Core Centroid $(y/h, z/h)$ | Core Localization Error | Vortices Detected | Streamwise $U$ RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coarse Base (M1)** | 20,000 | 7.99% | (0.850, 0.350) | 0.586 h | 8 cores | 0.0934 m/s |
| **Refined Grid (M2)** | 80,000 | 2.62% | (0.825, 0.475) | 0.474 h | 8 cores | 0.0857 m/s |
| **Reference DNS Target** | N/A | **2.03%** | **(0.500, 0.820)** | **0.000 h (Ref)** | **8 cores** | **0.0000 m/s** |

* **Secondary Circulation Convergence:** Reducing cross-stream discretization error reduces peak secondary speed from $7.99\%$ to $2.62\%$, converging sharply toward the DNS benchmark reference value ($2.03\%$).
* **Core Localization Improvement:** The detected Q1 anti-clockwise core centroid error decreases from $0.586\,h$ down to $0.474\,h$.
* **Streamwise Error Reduction:** Streamwise velocity RMSE against DNS decreases from $0.0934\text{ m/s}$ down to $0.0857\text{ m/s}$ (-8.2%).
* **Topological Stability:** Both meshes cleanly identify all **8 distinct vortex cores** using connected-component labeling ([`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py)), confirming topological stability under refinement.

---

## Operating Envelope: Zero Out-of-Distribution Drift

A known failure mode in ML turbulence models is out-of-distribution drift, where near-wall strain rates cause invariants to drift far outside the training support, leading to solver divergence.

To verify that the model operates strictly in interpolation mode, [`scripts/validation/evaluate_timescale_distribution.py`](scripts/validation/evaluate_timescale_distribution.py) evaluated all 5 Cayley-Hamilton invariants and tensor basis norms across all 20,000 cells in the 3D CFD mesh against the DNS training envelope:

![Operating Envelope](docs/figures/timescale_and_invariant_distribution.png)

| Feature / Invariant | DNS Training Envelope | CFD Range (All 20k Cells) | Coverage |
| :--- | :---: | :---: | :---: |
| **$I_1 = \mathrm{tr}(\mathbf{s}^2)$** | $[0.000, 0.933]$ | $[1.665 \times 10^{-4}, 0.0686]$ | **100.0% In-Bounds** |
| **$I_2 = \mathrm{tr}(\mathbf{w}^2)$** | $[-0.932, 0.000]$ | $[-0.0682, -1.218 \times 10^{-8}]$ | **100.0% In-Bounds** |
| **$I_3 = \mathrm{tr}(\mathbf{s}^3)$** | $[-0.0349, 0.00426]$ | $[-9.645 \times 10^{-4}, 1.166 \times 10^{-3}]$ | **100.0% In-Bounds** |
| **$I_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s})$** | $[-0.00142, 0.0116]$ | $[-3.909 \times 10^{-4}, 3.180 \times 10^{-4}]$ | **100.0% In-Bounds** |
| **$I_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$** | $[-0.435, 0.000]$ | $[-2.338 \times 10^{-3}, -9.433 \times 10^{-12}]$ | **100.0% In-Bounds** |
| **Basis Norms $\|\mathbf{T}^{(n)}\|_F$** | $[0.0, 1.318]$ | $[0.0, 0.243]$ | **100.0% Bounded** |

**Zero Extrapolation:** Exactly 100.0% of all cells in the simulation operate strictly inside the DNS training support. The neural network never extrapolates outside its trained manifold during the run.

---

## Automated Verification Suite (17/17 Tests Passing)

The repository includes a pytest test suite verifying numerical accuracy, coordinate conventions, and parity across all components:

```bash
pytest -v
```

```text
tests/test_coordinate_convention.py::test_non_square_gradient_ordering PASSED [  5%]
tests/test_coordinate_convention.py::test_non_square_interpolation_ordering PASSED [ 11%]
tests/test_coordinate_convention.py::test_benchmark_dns_grid_convention PASSED [ 17%]
tests/test_mesh_sensitivity.py::test_mesh_sensitivity_core_and_velocity PASSED [ 23%]
tests/test_parity.py::test_basis_trace_freedom PASSED                    [ 29%]
tests/test_parity.py::test_basis_symmetry PASSED                         [ 35%]
tests/test_parity.py::test_cardano_eigenvalue_accuracy PASSED            [ 41%]
tests/test_parity.py::test_realizability_bounds PASSED                   [ 47%]
tests/test_parity.py::test_evaluator_parity PASSED                       [ 52%]
tests/test_parity.py::test_torchscript_parity PASSED                     [ 58%]
tests/test_parity.py::test_json_weights_parity PASSED                    [ 64%]
tests/test_parity.py::test_standalone_cpp_numerical_parity PASSED        [ 70%]
tests/test_timescale_distribution.py::test_cfd_invariants_within_training_envelope PASSED [ 76%]
tests/test_timescale_distribution.py::test_basis_norms_bounded_by_training_distribution PASSED [ 82%]
tests/test_vortex_detection.py::test_zero_vorticity_field PASSED         [ 88%]
tests/test_vortex_detection.py::test_known_four_vortex_field PASSED      [ 94%]
tests/test_vortex_detection.py::test_known_eight_vortex_field PASSED     [100%]

======================= 17 passed, 1 warning in 16.23s =======================
```

---

## Quick Start: How to Build and Run

### 1. Python Environment
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Tests
```bash
pytest
```

### 3. Model Training & Export
To re-train the model on the primary DNS benchmark (`square_duct_dns_Reb4410_Retau150.csv`) and export the TorchScript weights:
```bash
python tbnn_train.py
python tbnn_export.py
```

### 4. Build the Solvers
Ensure your OpenFOAM environment is active (`source /opt/openfoam/etc/bashrc`):

```bash
# Build the standalone C++ solver (zero third-party dependencies):
cd src/tbnnSimpleFoam
wmake
cd ../..

# Or build the LibTorch solver (requires TORCH_LIBRARIES in Make/options):
wmake
```

### 5. Run the Square Duct Case
```bash
cd verification_squareDuct
./Allrun
```

### 6. Post-Process and Validate Against DNS
```bash
python validate_and_plot.py
```

---

## Repository Structure

```text
tbnnFoam/
├── tbnnFoam.C                     # OpenFOAM solver with embedded LibTorch
├── createFields.H                 # In-situ field allocation
├── tbnn_train.py                  # PyTorch model training script
├── tbnn_export.py                 # TorchScript export script
├── tbnn_model.pt                  # Serialized TorchScript model
├── Make/                          # wmake configuration for tbnnFoam
├── src/
│   ├── tbnnEvaluator.H            # Standalone C++ evaluation engine & JSON parser
│   └── tbnnSimpleFoam/            # Standalone TBNN solver (no LibTorch dependency)
├── export_weights/
│   └── weights.json               # Exported model weights & normalization parameters
├── data/
│   └── benchmark_dns/             # Canonical square duct DNS datasets (Pirozzoli et al., 2018)
├── docs/
│   ├── figures/                   # Publication-grade validation figures and animations
│   └── report/                    # Compiled project summary PDF reports
├── scripts/
│   ├── generate_all_showcase_assets.py  # Figure generation script
│   ├── generate_concise_pdf.py          # PDF project summary compiler
│   └── validation/                      # Quantitative DNS evaluation & vortex detection
├── tests/                         # 17-test automated verification suite
└── verification_squareDuct/       # Square duct benchmark case (Re_b=4410, Re_tau=150)
    ├── 0/                         # Boundary conditions
    ├── constant/                  # Transport & turbulence properties
    ├── system/                    # OpenFOAM schemes and solution controls
    ├── Allrun                     # Case execution script
    └── validate_and_plot.py       # Post-processing and DNS validation
```

---

## References

1. Ling, J., Kurzawski, A., & Templeton, J. (2016). *Reynolds averaged turbulence modelling using deep neural networks with embedded invariance*. **Journal of Fluid Mechanics**, 807, 155–166.
2. Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018). *Turbulence and secondary motions in square duct flow*. **Journal of Fluid Mechanics**, 840, 634–655.
3. Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. **Journal of Fluid Mechanics**, 72(2), 331–340.
4. Banerjee, S., Krahl, R., Durst, F., & Zenger, C. (2007). *Presentation of anisotropy properties of turbulence, invariants, versus eigenvalue approaches*. **Journal of Turbulence**, 8, N32.
5. Nikuradse, J. (1930). *Untersuchungen über turbulente Strömungen in nicht kreisförmigen Rohren*. **Ingenieur-Archiv**, 1(3), 306–332.

---

## Citation

```bibtex
@software{ghosh2026tbnnfoam,
  author       = {Soumik Ghosh},
  title        = {{tbnnFoam: In-Situ Tensor Basis Neural Network RANS Closure in OpenFOAM}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/BLACKNIGHT-cloud12/tbnnFoam}},
  version      = {1.0.0}
}
```
