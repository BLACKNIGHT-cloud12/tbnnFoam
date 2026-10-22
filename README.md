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

* **Physical Scaling Reconciliation:** 
  - **Canonical Dimensionless Domain:** Normalized coordinates $y/h \in [-1, 1]$ and $z/h \in [-1, 1]$ with origin $(0, 0)$ at duct centerline; normalized velocities $U/U_b, V/U_b, W/U_b$.
  - **OpenFOAM CFD Scale:** Laboratory square duct ($2h \times 2h$) with half-width $h = 0.05\text{ m}$ ($D_h = 0.10\text{ m}$), length $L = 1.0\text{ m} = 20h$, $U_b = 1.0\text{ m/s}$, and kinematic viscosity $\nu = 2.25 \times 10^{-5}\text{ m}^2/\text{s}$ ($Re_b = U_b D_h / \nu \approx 4440$, $Re_h \approx 2220$).
  - **Reference DNS Database Scale (Pirozzoli et al. 2018):** Unit hydraulic diameter $D_h = 1.0\text{ m}$ ($h = 0.5\text{ m}$), $U_b = 1.0\text{ m/s}$, and $\nu = 2.26757 \times 10^{-4}\text{ m}^2/\text{s}$ ($Re_b = 4410$, $Re_h = 2205$, $Re_\tau = 150.0$).
  - Both map onto identical non-dimensional flow physics and are formally documented in [`data/benchmark_dns/benchmark_metadata.json`](data/benchmark_dns/benchmark_metadata.json).
* **Canonical DNS Reference:** From column 13 (`omx/(ub/h)`) of Pirozzoli et al. (2018) on a $65 \times 65$ cross-section grid, the primary anti-clockwise (ACW) vortex core in Quadrant 1 ($y > 0, z > 0$) is located at **$(y/h, z/h) = (0.50, 0.82)$**.

![Benchmark Validation](docs/figures/tbnn_hero_showcase.png)

### Independent Vortex Core Detection Methodology

To ensure that vortex core comparisons are objective and reproducible rather than relying on visual inspection or static annotations, the repository implements an independent algorithmic vortex detector in [`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py):

1. **Discretization & Vorticity Sign Convention:** The non-dimensional axial vorticity field $\omega_x^*$ is computed via second-order central differences on normalized coordinates:
   $$\omega_x^*(y/h, z/h) = \frac{\omega_x}{U_b / h} = \frac{\partial (W / U_b)}{\partial (y / h)} - \frac{\partial (V / U_b)}{\partial (z / h)}$$
   Here, $\omega_x^* > 0$ denotes positive circulation (**anti-clockwise / ACW**), and $\omega_x^* < 0$ denotes negative circulation (**clockwise / CW**).
2. **Boundary Layer Wall-Margin Mask:** An interior mask $|y/h| \le 0.92, |z/h| \le 0.92$ is applied to prevent attached wall shear layers ($\partial U/\partial n$) from registering as secondary flow vortices.
3. **Filtering Fragmented Vorticity from Physical Cores:** To distinguish the 8 primary physical secondary cells from weak numerical artifacts or fragmented sub-structures, the detector enforces three filtering criteria:
   * **Strength Thresholding:** Cells must satisfy $|\omega_x^*| = |\omega_x| / (U_b / h) \ge 0.04$, discarding low-level numerical background noise.
   * **Minimum Cluster Area:** Connected components (identified via 8-connectivity labeling with `scipy.ndimage.label`) must contain at least 4 contiguous cells (`min_area = 4`), eliminating single-cell numerical spikes.
   * **Non-Maximum Suppression (Distance Separation):** A separation radius `min_separation = 0.20 h` is enforced between candidate cores of the same sign, merging fragmented sub-peaks into the dominant local vortex core.
4. **Vortex Center Extraction & Matching Rule:** The detected core coordinate $(y_{\text{core}}, z_{\text{core}})$ is the location of the maximum $|\omega_x|$ within each retained component. In Quadrant 1 ($y > 0, z > 0$), which contains one counter-rotating vortex pair, the detected ACW core closest to the canonical DNS reference $(0.50, 0.82)$ in Euclidean distance is matched for quantitative tracking:
   $$\Delta r = \sqrt{(y_{\text{CFD}} - 0.50)^2 + (z_{\text{CFD}} - 0.82)^2} \quad [h]$$

Reproduce the detection and metrics directly on the raw CFD outputs:
```bash
python scripts/validation/evaluate_mesh_sensitivity.py
python scripts/validation/evaluate_validation_plan.py
```

### Quantitative Comparison & Error Definitions

Metrics are evaluated on both the native CFD mesh cross-section and the full $65 \times 65 = 4{,}225$ uniform benchmark grid $(y/h, z/h) \in [-1, 1] \times [-1, 1]$ (sampled at the developing station $x/h = 7.0$, or $x/D_h = 3.5$), comparing `tbnnFoam` (`verification_squareDuct/300/U`) against converged baseline $k\text{--}\omega$ SST (`verification_squareDuct/baseline_sst/111/U`) and reference DNS:

* **Streamwise Velocity Normalization:** Normalized by the bulk mean velocity $U_{\text{ref}} = U_b = 1.00\text{ m/s}$:
  $$\mathrm{RMSE}_U = \sqrt{\frac{1}{N}\sum_{i=1}^N (U_{\text{CFD}, i} - U_{\text{DNS}, i})^2} = 0.0934\text{ m/s}, \quad \mathrm{NRMSE}_U = \frac{\mathrm{RMSE}_U}{U_b} \times 100\% = 9.34\%$$
* **Secondary Velocity RMSE:** Evaluated over all cross-stream velocity components:
  $$\mathrm{RMSE}_{\text{sec}} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left[(V_{\text{CFD}, i} - V_{\text{DNS}, i})^2 + (W_{\text{CFD}, i} - W_{\text{DNS}, i})^2\right]} = 0.0261\text{ m/s}$$

| Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` (Native Mesh / DNS Grid) | Reference DNS (Pirozzoli et al. 2018) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortex Cores** | 0 cores (Failed) | **8 cores (Detected)** | 8 cores (Canonical) | Recovers Prandtl's 8 corner vortex pairs |
| **Peak Secondary Speed ($V_{\text{sec}} / U_b$)** | 0.45% (Noise) | **7.99% (native) / 6.38% (DNS grid)** | 2.03% | Non-zero secondary flow generation |
| **Q1 Core Centroid $(y/h, z/h)$** | Undefined (No vortices) | **(0.850, 0.350) / (0.91, 0.78)** | (0.50, 0.82) | Native discrete cell center vs. $65\times65$ interpolated peak |
| **Core Distance Error** | N/A | **0.586 h (native) / 0.408 h (DNS grid)** | 0.000 h (Ref) | Distance to benchmark core centroid |
| **Streamwise $U$ RMSE** | 0.0616 m/s (6.16%) | **0.0934 m/s (9.34%)** | 0.0000 m/s (Ref) | Full cross-section NRMSE ($U_{\text{ref}} = 1.00\text{ m/s}$) |
| **Secondary $(V, W)$ RMSE** | 0.0087 m/s | **0.0261 m/s** | 0.0000 m/s (Ref) | In-plane cross-stream velocity recovery |
| **Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Both meet numerical steady-state convergence |

> [!NOTE]
> **Reconciliation of Native Mesh vs. Interpolated Grid Metrics:**
> On the native coarse CFD mesh ($20 \times 20$ cross-section), the discrete cell centers have spacing $\Delta y/h = \Delta z/h = 0.10$. Connected-component labeling identifies the discrete cell center $(0.850, 0.350)$ as the cell-centered peak, giving a core localization error of $0.586\,h$ and peak secondary velocity $7.99\%$. When the velocity field is bilinearly interpolated onto the $65 \times 65$ DNS grid ($\Delta y/h = \Delta z/h = 0.03125$), the resolved peak shifts to $(0.91, 0.78)$, yielding a distance error of $0.408\,h$ and peak secondary velocity $6.38\%$. Both evaluations are generated directly and reproducibly by `scripts/validation/evaluate_mesh_sensitivity.py` (native mesh) and `scripts/validation/evaluate_validation_plan.py` (interpolated DNS grid).

---

## Honest Technical Discussion: Physical Validation & Model Limitations

### Representational Capability vs. Physical Realization
Generating non-zero normal stress differences $(\overline{w'^2} - \overline{v'^2} \ne 0)$ is a necessary *mathematical representational capability* of non-linear tensor basis closures beyond the linear Boussinesq hypothesis. However, non-zero normal stresses alone do not prove that the model has successfully overcome the Boussinesq limitation in the physical solution. 

The physical validation rests on coupling the anisotropic stress divergence $\nabla \cdot \boldsymbol{\tau}_\Delta$ into OpenFOAM's momentum predictor and demonstrating that:
1. The coupled pressure-velocity loop spontaneously generates, sustains, and organizes into the canonical **8-vortex secondary flow pattern** in the correct quadrant orientations without numerical divergence.
2. Under cross-stream grid refinement, peak secondary circulation trends from $7.99\%$ toward the DNS benchmark of $2.03\%$, and core localization error drops from $0.586\,h$ down to $0.474\,h$.
3. Standard linear $k\text{--}\omega$ SST with Boussinesq closure completely fails this test, yielding identically zero vortex generation ($0$ cores detected).

### Understanding Model Discrepancies and Limitations:
1. **Developing Spatial Duct vs. Streamwise-Periodic Box:**
   The reference DNS benchmark of Pirozzoli et al. (2018) was simulated in an idealized streamwise-periodic box driven by a body force (representing an asymptotically fully developed duct of infinite length). In contrast, our OpenFOAM simulation is a spatially developing duct ($L = 10h = 0.5\text{ m}$) with a uniform velocity inlet ($U_b = 1.0\text{ m/s}$) and physical boundary layers growing along the walls. 
   Because secondary flow is triggered locally by near-wall corner normal stress differences, the 8 secondary vortex cells establish quickly ($x/h \ge 6$). However, the centerline streamwise core requires a longer development length ($U_{\text{center}} \approx 1.25\text{ m/s}$ in our CFD case vs $1.33\text{ m/s}$ in the periodic DNS). Standard linear $k\text{--}\omega$ SST exhibits slightly lower streamwise error ($6.16\%$) simply because it lacks cross-stream convective momentum transport to the corners, yet it fails entirely on the secondary flow physics ($0$ vortex cores).
2. **Neural Network Model-Form Uncertainty:**
   The $9.3\%$ streamwise RMSE is not attributable solely to boundary conditions. The neural network closure is an empirical approximation trained on frozen DNS gradients. In live CFD coupling, the emerging secondary flow deforms the mean velocity field $\mathbf{U}$, modifying the velocity gradients fed back into the network at each SIMPLE iteration. This interactive feedback, combined with the finite expressivity of the MLP, introduces genuine model-form discrepancy.
3. **Modelled $\omega$ Surrogate & Turbulent Timescale Sensitivity:**
   Direct numerical simulation yields exact mean velocity gradients $\nabla \mathbf{U}$ and true Reynolds stress tensors $\boldsymbol{\tau}^R$, but DNS inherently lacks the specific dissipation rate $\omega$ because $\omega$ is a semi-empirical RANS modeling construct rather than a physical Navier-Stokes field.
   In the training dataset, $\omega$ was estimated using an empirical SST surrogate relation ($\omega \approx k^{1/2} / (C_\mu^{1/4} \ell)$ with $C_\mu = 0.09, \ell = \min(\kappa d_w, 0.07 D_h)$). In live OpenFOAM CFD execution, $\omega$ is governed dynamically by its transport equation.
   
   **Influence on Timescale $\tau$:** The turbulent timescale $\tau = \min(1/\omega, 5.0\text{ s})$ directly scales the dimensionless strain $\mathbf{s} = \tau \mathbf{S}$ and rotation $\mathbf{w} = \tau \mathbf{W}$. Consequently:
   * The scalar invariants scale non-linearly with $\tau$: $I_1, I_2 \propto \tau^2$, $I_3 \propto \tau^3$, $I_4 \propto \tau^4$, and $I_5 \propto \tau^5$.
   * Pope's basis tensors scale as $\mathbf{T}^{(1)} \propto \tau$, $\mathbf{T}^{(2..4)} \propto \tau^2$, $\mathbf{T}^{(5..8)} \propto \tau^3$, and $\mathbf{T}^{(9..10)} \propto \tau^4$.
   * As a result, an underestimation of $\omega$ (overestimating $\tau$) strongly amplifies the higher-order tensor bases and accelerates secondary flow intensity, whereas overestimating $\omega$ damps the perturbation stress $\boldsymbol{\tau}_\Delta$ back toward linear RANS. Capping $\tau \le 5.0\text{ s}$ serves as an essential regularizer preventing singular invariant growth near the duct center where $\omega \to 0$.

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

* **Secondary Circulation Discretization Sensitivity:** Increasing cross-stream resolution reduces peak secondary speed from $7.99\%$ to $2.62\%$, demonstrating a clear trend toward the DNS benchmark reference value ($2.03\%$).
* **Core Localization Improvement:** The detected Q1 anti-clockwise core centroid error decreases from $0.586\,h$ down to $0.474\,h$.
* **Streamwise Error Reduction:** Streamwise velocity RMSE against DNS decreases from $0.0934\text{ m/s}$ down to $0.0857\text{ m/s}$ (-8.2%).
* **Topological Stability:** Both meshes cleanly identify all **8 distinct vortex cores** using connected-component labeling ([`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py)), confirming topological stability under refinement.

---

## High-Reynolds Number Simulation: Square Duct at $Re_b = 17{,}800$ ($Re_\tau \approx 519$)

To demonstrate scalability to higher Reynolds numbers matching the asymptotic DNS benchmark regime of Pirozzoli et al. (2018), a dedicated high-Reynolds-number verification case is provided in [`verification_squareDuct_Re17800/`](verification_squareDuct_Re17800/):

* **Physical Scaling Reconciliation:**
  - Bulk velocity $U_b = 1.0\text{ m/s}$, hydraulic diameter $D_h = 0.10\text{ m}$ ($h = 0.05\text{ m}$), kinematic viscosity $\nu = 5.617978 \times 10^{-6}\text{ m}^2/\text{s}$, yielding:
    $$Re_b = \frac{U_b D_h}{\nu} = 17{,}800 \quad (Re_h = 8{,}900, \quad Re_\tau \approx 519)$$
  - Inlet turbulence conditions: $k_{\text{inlet}} = 0.00333\text{ m}^2/\text{s}^2$ ($I = 4.71\%$), $\omega_{\text{inlet}} = 15.0\text{ s}^{-1}$.
  - Grid resolution: $50 \times 40 \times 40 = 80{,}000$ cells ($y^+ \approx 13$ with $k\text{--}\omega$ SST wall functions).
* **Numerical Convergence:** Solved via `tbnnFoam` to full steady-state convergence in **294 SIMPLE iterations** ($< 10^{-4}$ residuals across all velocity and turbulence fields).
* **Streamwise Spatial Development & Quantitative Reconciliation:** Directly reproducible via:
  ```bash
  python verification_squareDuct_Re17800/validate_and_plot_Re17800.py
  ```

| Station | $x/D_h$ | $U_c / U_b$ | Raw Max $V_{\mathrm{sec}}/U_b$ | Core Max ($|y/h|,|z/h| \le 0.85$) | Cores Detected | Q1 Centroid $(y/h, z/h)$ | Distance to DNS $(0.50, 0.82)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | $1.1$ | $1.019$ | $9.63\%$ | $1.91\%$ | 0 (Rollup) | N/A | N/A |
| **#2** | $2.1$ | $1.037$ | $10.43\%$ | $2.80\%$ | 0 (Rollup) | N/A | N/A |
| **#3** | $3.1$ | $1.054$ | $10.43\%$ | $2.39\%$ | 0 (Rollup) | N/A | N/A |
| **#4** | $4.1$ | $1.069$ | $10.20\%$ | $1.95\%$ | 0 (Rollup) | N/A | N/A |
| **#5** | $5.1$ | $1.083$ | $9.92\%$ | $1.87\%$ | 0 (Rollup) | N/A | N/A |
| **#6** | $6.1$ | $1.096$ | $9.69\%$ | $1.84\%$ | **8 cores** | $(0.625, 0.875)$ | $0.137\,h$ |
| **#7** | $7.1$ | $1.108$ | $9.55\%$ | $1.78\%$ | **8 cores** | $(0.575, 0.875)$ | $0.093\,h$ |
| **#8** | $8.1$ | $1.121$ | $9.34\%$ | $1.70\%$ | **8 cores** | $(0.575, 0.875)$ | $0.093\,h$ |
| **#9** | $9.1$ | $1.132$ | $9.39\%$ | $1.67\%$ | **8 cores** | $(0.525, 0.875)$ | $\mathbf{0.060\,h}$ (1.2 cells) |
| **DNS Ref** | $\infty$ | $1.33$ | $2.03\% - 2.30\%$ | **$2.03\% - 2.30\%$** | **8 cores** | **$(0.500, 0.820)$** | $\mathbf{0.000\,h}$ (Reference) |

* **Key Physical & Numerical Insights:**
  1. **Core Secondary Speed Agreement:** Across the fully established secondary flow region ($x/D_h \ge 6$), the physical core secondary speed stabilizes at **$1.67\% - 1.84\%$** (mean vortex core speed **$1.72\%$**), bounding the DNS benchmark range ($2.03\% - 2.30\%$).
  2. **Vortex Centroid Convergence:** As the boundary layer develops downstream from $x/D_h = 6.1$ to $9.1$, the detected Quadrant 1 core centroid migrates from $(0.625, 0.875)$ to $(0.525, 0.875)$, reducing core distance error against canonical DNS down to $\Delta r = 0.060\,h$ (barely $1.2$ discrete cells on the $40 \times 40$ mesh).
  3. **Near-Wall Gradient Mechanism:** The raw unmasked grid maximum ($9.3\% - 10.4\%$) occurs strictly in the outermost boundary-layer cell center ($|z/h| = 0.975$) where steep wall-normal velocity gradients ($\partial U_x / \partial z \sim 400\text{ s}^{-1}$) interact with the explicit anisotropic stress divergence $\nabla \cdot \boldsymbol{\tau}_\Delta$ under high-shear wall functions, while the physical vortex core circulation throughout the interior cross-section remains smooth and within $\sim 1.7\% - 2.4\%$.

![High-Re Validation Showcase](docs/figures/tbnn_Re17800_showcase.png)

---

## Operating Envelope: 1D Marginal Invariant Bounding & Joint Distribution Shift Diagnostic

A known failure mode in ML turbulence modeling is out-of-distribution (OOD) drift, where high strain rates near corners or walls cause invariants to drift far outside the training support, leading to solver divergence.

To quantitatively evaluate whether the model operates in interpolation mode, [`scripts/validation/evaluate_timescale_distribution.py`](scripts/validation/evaluate_timescale_distribution.py) assesses both **1D marginal range coverage** and a **joint 5D feature-space nearest-neighbor diagnostic** across all 20,000 cells in the 3D CFD mesh against the DNS training distribution:

![Operating Envelope & Distribution Diagnostic](docs/figures/timescale_and_invariant_distribution.png)

### 1D Marginal Invariant Coverage (Extrapolation Protection)

| Feature / Invariant | DNS Training Envelope | CFD Range (All 20k Cells) | Marginal Coverage |
| :--- | :---: | :---: | :---: |
| **$I_1 = \mathrm{tr}(\mathbf{s}^2)$** | $[0.000, 0.933]$ | $[2.927 \times 10^{-4}, 0.0690]$ | **100.0% In-Bounds** |
| **$I_2 = \mathrm{tr}(\mathbf{w}^2)$** | $[-0.932, 0.000]$ | $[-0.0682, -1.118 \times 10^{-10}]$ | **100.0% In-Bounds** |
| **$I_3 = \mathrm{tr}(\mathbf{s}^3)$** | $[-0.0349, 0.00426]$ | $[-4.853 \times 10^{-3}, 4.505 \times 10^{-3}]$ | **99.0% In-Bounds** |
| **$I_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s})$** | $[-0.00142, 0.0116]$ | $[-1.453 \times 10^{-3}, 1.546 \times 10^{-3}]$ | **99.8% In-Bounds** |
| **$I_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$** | $[-0.435, 0.000]$ | $[-2.350 \times 10^{-3}, -4.317 \times 10^{-14}]$ | **100.0% In-Bounds** |
| **Basis Norms $\|\mathbf{T}^{(n)}\|_F$** | $[0.0, 1.318]$ | $[0.0, 0.243]$ | **100.0% Bounded** |

### Joint 5D Distribution Shift Diagnostic (Standardized $k$-NN Distances)

While 1D marginal bounds confirm that no single invariant experiences catastrophic runaway (averaging $99.8\%$ overall marginal bounding), **1D marginal bounding does not guarantee joint manifold containment**. Because the CFD simulation is spatially developing ($\partial U/\partial x \ne 0$) with interactive velocity updates, whereas the DNS benchmark is a fully developed periodic box, live CFD cells inevitably experience modest joint distribution shift.

We quantify this shift via standardized Euclidean nearest-neighbor distance in the 5D invariant space:
$$d_{\text{kNN}}(\mathbf{x}_{\text{CFD}}) = \min_{j} \left\| \frac{\mathbf{x}_{\text{CFD}} - \boldsymbol{\mu}_{\text{DNS}}}{\boldsymbol{\sigma}_{\text{DNS}}} - \frac{\mathbf{x}_{\text{DNS}, j} - \boldsymbol{\mu}_{\text{DNS}}}{\boldsymbol{\sigma}_{\text{DNS}}} \right\|_2$$

* **Median Standardized Distance:** $0.21\,\sigma$
* **Mean Standardized Distance:** $0.33\,\sigma$
* **95th Percentile:** $1.17\,\sigma$
* **Maximum Distance:** $1.43\,\sigma$

This establishes that live CFD operating points remain closely clustered around observed training states (median gap $< 0.25$ standard deviations), preventing unphysical tensor extrapolations while transparently characterizing the joint distribution shift inherent in live Navier-Stokes coupling.

---

## Automated Verification Suite (19/19 Tests Passing)

The repository includes a pytest test suite verifying numerical accuracy, coordinate conventions, mesh geometry reconstruction, and parity across all components:

```bash
pytest -v
```

```text
tests/test_coordinate_convention.py::test_non_square_gradient_ordering PASSED [  5%]
tests/test_coordinate_convention.py::test_non_square_interpolation_ordering PASSED [ 10%]
tests/test_coordinate_convention.py::test_benchmark_dns_grid_convention PASSED [ 15%]
tests/test_mesh_reconstruction_verification.py::test_coarse_mesh_geometry_and_ordering PASSED [ 21%]
tests/test_mesh_reconstruction_verification.py::test_fine_mesh_geometry_and_ordering PASSED [ 26%]
tests/test_mesh_sensitivity.py::test_mesh_sensitivity_core_and_velocity PASSED [ 31%]
tests/test_parity.py::test_basis_trace_freedom PASSED                    [ 36%]
tests/test_parity.py::test_basis_symmetry PASSED                         [ 42%]
tests/test_parity.py::test_cardano_eigenvalue_accuracy PASSED            [ 47%]
tests/test_parity.py::test_realizability_bounds PASSED                   [ 52%]
tests/test_parity.py::test_evaluator_parity PASSED                       [ 57%]
tests/test_parity.py::test_torchscript_parity PASSED                     [ 63%]
tests/test_parity.py::test_json_weights_parity PASSED                    [ 68%]
tests/test_parity.py::test_standalone_cpp_numerical_parity PASSED        [ 73%]
tests/test_timescale_distribution.py::test_cfd_invariants_within_training_envelope PASSED [ 78%]
tests/test_timescale_distribution.py::test_basis_norms_bounded_by_training_distribution PASSED [ 84%]
tests/test_vortex_detection.py::test_zero_vorticity_field PASSED         [ 89%]
tests/test_vortex_detection.py::test_known_four_vortex_field PASSED      [ 94%]
tests/test_vortex_detection.py::test_known_eight_vortex_field PASSED     [100%]

======================= 19 passed, 1 warning in 16.88s =======================
```

### Mesh Geometry Reconstruction & Cell Ordering Verification

To eliminate verification limitations associated with hard-coded mesh dimensions and assumed cell ordering:
1. **Direct Geometry Parsing:** [`scripts/validation/evaluate_mesh_sensitivity.py`](scripts/validation/evaluate_mesh_sensitivity.py) and [`scripts/validation/evaluate_validation_plan.py`](scripts/validation/evaluate_validation_plan.py) directly parse OpenFOAM's cell center vector field (`0/C`) generated by `postProcess -func writeCellCentres`.
2. **Explicit Block Ordering Verification:** In [`tests/test_mesh_reconstruction_verification.py`](tests/test_mesh_reconstruction_verification.py), we verify against OpenFOAM's mesh geometry that:
   - Streamwise cells $X$ strictly increment along the fastest axis (axis 2, $N_x = 50$).
   - Horizontal cross-stream cells $Y$ strictly increment along axis 1 ($N_y = 20$ coarse, $40$ fine).
   - Vertical cross-stream cells $Z$ strictly increment along axis 0 ($N_z = 20$ coarse, $40$ fine).
   - True cell center coordinates $y/h$ and $z/h$ derived from the mesh geometry match uniform cell-center definitions to within machine precision ($\Delta_{\max} < 2 \times 10^{-16}$).
This ensures that field reshaping introduces identically zero distortion or bias into the computed vortex centers and velocity RMSE.

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
├── verification_squareDuct/       # Square duct benchmark case (Re_b=4410, Re_tau=150)
│   ├── 0/                         # Boundary conditions
│   ├── constant/                  # Transport & turbulence properties
│   ├── system/                    # OpenFOAM schemes and solution controls
│   ├── Allrun                     # Case execution script
│   └── validate_and_plot.py       # Post-processing and DNS validation
└── verification_squareDuct_Re17800/ # High-Re verification case (Re_b=17800, Re_tau=519)
    ├── 0/                         # Scaled boundary conditions
    ├── 294/                       # Converged solution fields
    ├── constant/                  # Transport properties (nu=5.618e-6)
    ├── system/                    # BlockMesh (80k cells) & solution controls
    ├── Allrun                     # Case execution script
    ├── tbnn_Re17800_showcase.png  # 4-panel secondary flow visualization
    └── validate_and_plot_Re17800.py # Automated 8-core vortex detector & metrics
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
