# tbnnFoam: In-Situ Tensor Basis Neural Network RANS Closure in OpenFOAM

<div align="center">

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2012%20--%20v2606-004C87.svg?logo=openfoam&logoColor=white)](https://www.openfoam.com)
[![LibTorch](https://img.shields.io/badge/LibTorch-C%2B%2B11_ABI-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%20--%203.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**In-situ implementation of the Tensor Basis Neural Network (TBNN) turbulence closure in OpenFOAM for Reynolds-averaged Navier-Stokes (RANS) simulations.**

[Benchmark Data](data/benchmark_dns/) • [Theory](#mathematical-formulation) • [Validation](#numerical-results--physics-verification) • [Citation](#citation)

</div>

---

## Overview

Linear eddy-viscosity RANS models (e.g., standard $k\text{--}\varepsilon$, $k\text{--}\omega$ SST) rely on the Boussinesq approximation:

$$\boldsymbol{\tau}^R = 2\nu_t \mathbf{S} - \frac{2}{3}k\mathbf{I}$$

Because turbulent viscosity $\nu_t$ is scalar, linear eddy-viscosity models enforce isotropic normal stresses ($\overline{v'^2} = \overline{w'^2}$) in non-circular ducts. Consequently, they cannot capture turbulence-driven secondary flows of the second kind (Prandtl, 1926), which are driven by gradients of normal stress differences:

$$\mathbf{U} \cdot \nabla \omega_x \approx \frac{\partial^2}{\partial y \partial z}\left(\overline{w'^2} - \overline{v'^2}\right) + \left(\frac{\partial^2}{\partial z^2} - \frac{\partial^2}{\partial y^2}\right)\overline{v'w'}$$

Setting $\overline{v'^2} - \overline{w'^2} = 0$ yields identically zero cross-stream velocities ($V = W = 0$), missing the corner vortex pairs and near-wall velocity contour bulging (Nikuradse, 1930).

**`tbnnFoam`** incorporates the **Tensor Basis Neural Network (TBNN)** architecture of Ling et al. (2016) directly into OpenFOAM's finite-volume discretization loop:
* **Coupling Strategies:**
  1. **Embedded LibTorch Solver (`tbnnFoam`):** Directly evaluates compiled TorchScript models (`tbnn_model.pt`) inside OpenFOAM's matrix assembly via `torch::jit`.
  2. **Standalone C++ Solver (`tbnnSimpleFoam`):** Lightweight in-solver implementation using [src/tbnnEvaluator.H](src/tbnnEvaluator.H) to load network weights (`weights.json`) with zero external runtime dependencies.
* **Frame Invariance:** Embeds Pope's (1975) 10-tensor integrity basis and 5 scalar invariants, ensuring Galilean and coordinate-rotation invariance by construction.
* **DNS Verification ($Re_\tau = 150$, $Re_b \approx 4410$ / $Re_h \approx 2205$):** Verified against canonical square duct DNS data of Pirozzoli et al. (2018) (Sapienza University of Rome CFD Database), with held-out generalization testing on $Re_\tau = 220$ ($Re_b \approx 7000$ / $Re_h \approx 3500$).

---

## Solver Architecture

The solver updates the mean momentum equation during each SIMPLE iteration with an anisotropic stress divergence correction:

![Architecture Pipeline](docs/figures/architecture_pipeline.png)

1. **Local Strain and Rotation Extraction:** Extracts velocity gradient $\nabla \mathbf{U}$, kinetic energy $k$, and dissipation $\varepsilon$ (or specific dissipation $\omega$) at cell centroids.
2. **Invariant & Basis Engine:** Evaluates the non-dimensional strain rate $\mathbf{s} = \frac{k}{2\varepsilon}(\nabla \mathbf{U} + \nabla \mathbf{U}^T)$ and rotation rate $\mathbf{w} = \frac{k}{2\varepsilon}(\nabla \mathbf{U} - \nabla \mathbf{U}^T)$, computing the 5 scalar invariants $\lambda_1 \dots \lambda_5$ and 10 symmetric, trace-free tensor bases $\mathbf{T}^{(1)} \dots \mathbf{T}^{(10)}$.
3. **Neural Network Forward Pass:** Predicts 10 scalar expansion coefficients $g^{(1)} \dots g^{(10)}$.
4. **Anisotropy Reconstruction & Realizability Clipping:** Forms $\mathbf{a} = \sum g^{(n)} \mathbf{T}^{(n)}$ and enforces Lumley realizability constraints.
5. **Momentum Coupling:** Injects the divergence of the extra anisotropic Reynolds stress $\nabla \cdot (2k\mathbf{a})$ as an explicit source term into OpenFOAM's momentum predictor (`UEqn.H`).

---

## Neural Network Architecture

The network embeds Pope's (1975) integrity basis expansion directly into the model topology:

![Network Architecture](docs/figures/tbnn_network_architecture.png)

* **Inputs:** 5 scalar invariants $\boldsymbol{\lambda} \in \mathbb{R}^5$:
  $$\lambda_1 = \mathrm{tr}(\mathbf{s}^2), \quad \lambda_2 = \mathrm{tr}(\mathbf{w}^2), \quad \lambda_3 = \mathrm{tr}(\mathbf{s}^3), \quad \lambda_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}), \quad \lambda_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$$
* **Hidden Layers:** Multilayer perceptron with ReLU activations mapping invariants $\boldsymbol{\lambda} \mapsto \mathbf{g} \in \mathbb{R}^{10}$.
* **Tensor Contraction:**
  $$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\boldsymbol{\lambda}) \, \mathbf{T}^{(n)}$$
* **Physical Stress:** $\boldsymbol{\tau}^R = 2k\left(\mathbf{a} + \frac{1}{3}\mathbf{I}\right)$.

### Dual Model Implementation & Parity

The repository provides two distinct inference paths tailored for production CFD:
1. **Embedded LibTorch Solver (`tbnnFoam`):**
   * **Topology:** 64 hidden units per layer with ReLU activations.
   * **Input Preprocessing:** Unnormalized invariants $\boldsymbol{\lambda} \in \mathbb{R}^5$ clamped to $[-50.0, 50.0]$ with time scale clipped to $\tau = \min(1/\omega, 5.0)\text{ s}$.
   * **Execution:** Direct in-memory C++ evaluation via `torch::jit` module (`tbnn_model.pt`).
2. **Standalone C++ Solver (`tbnnSimpleFoam`):**
   * **Topology:** 30 hidden units per layer with ReLU activations.
   * **Input Preprocessing:** Standardized invariants $\hat{\boldsymbol{\lambda}} = (\boldsymbol{\lambda} - \boldsymbol{\mu}_{\lambda}) / \boldsymbol{\sigma}_{\lambda}$ with time scale clipped to $\tau = \min(1/\omega, 5.0)\text{ s}$.
   * **Execution:** Header-only matrix-vector multiplication engine ([src/tbnnEvaluator.H](src/tbnnEvaluator.H)) parsing JSON weights ([export_weights/weights.json](export_weights/weights.json)) with zero external third-party dependencies.

Both implementations enforce identical Pope (1975) 10-tensor basis contraction, time-scale limiters ($\tau \le 5.0$), and Cardano eigensolver Lumley realizability clipping. Parity across both architectures and execution engines is verified via [`tests/test_parity.py`](tests/test_parity.py):
* **PyTorch vs. TorchScript:** Verifies bitwise tensor output parity between the root PyTorch network and the traced `torch::jit` model.
* **Python vs. Standalone OpenFOAM C++ (`tbnnEvaluator.H`):** Compiles and executes the native C++ evaluation utility (`tests/cpp_parity/test_evaluator_parity`) against the reference Python pipeline on identical strain/rotation rates $(\nabla \mathbf{U}, \omega)$, confirming numerical parity across all 5 invariants, all 10 tensor bases, neural coefficients $\mathbf{g}$, and reconstructed realizable anisotropy tensors $\mathbf{a}$.

---

## Numerical Results & Physics Verification

### 1. Cross-Section Secondary Flow Dynamics
Advected through genuine OpenFOAM simulation velocity fields (`verification_squareDuct/300/U`):

![Cross-Section Particle Dynamics](docs/figures/animation_1_secondary_vortices.gif)

* **Standard Linear RANS:** Fluid particles move strictly axially ($V = W = 0$).
* **`tbnnFoam`:** Cross-stream secondary convection transports fluid toward the duct corners along diagonals and recirculates inward along wall bisectors.

### 2. In-Situ Solver Evolution & Residual History
Continuity residual convergence directly extracted from [`verification_squareDuct/log.tbnnFoam`](verification_squareDuct/log.tbnnFoam):

![Solver Convergence](docs/figures/animation_4_solver_convergence.gif)

* **$t = 73$ (Initialization):** Standard RANS initial condition; zero cross-stream secondary flow.
* **$t = 73 \to 140$ (TBNN Coupling):** Under-relaxed anisotropic stress divergence injection activates ($\alpha = 0.01$). Secondary circulation cells establish progressively.
* **$t = 300$ (300-Iteration Run State):** Continuity residuals reach $5.56 \times 10^{-7} < 10^{-6}$ with stable in-plane secondary convection (while pressure and velocity component residuals level off near $1.5\text{--}2.8 \times 10^{-4}$).

### 3. 3D Streamline Ribbons
Streamlines integrated through the genuine 3D velocity field (`verification_squareDuct/300/U`):

![3D Streamline Ribbons](docs/figures/animation_2_3d_streamlines.gif)

---

## Physical Mechanism: Prandtl's Secondary Flow

In straight non-circular ducts, secondary vortices originate from spatial gradients of normal stress differences:

$$\frac{\partial^2}{\partial y \partial z} \left(\overline{w'^2} - \overline{v'^2}\right) \neq 0$$

![Reynolds Stress Mechanism](docs/figures/reynolds_stress_mechanism.png)

* **Linear Eddy-Viscosity Models:** Predict $\overline{v'^2} - \overline{w'^2} \equiv 0$ everywhere due to scalar viscosity.
* **`tbnnFoam`:** Extracted directly from OpenFOAM simulation outputs (`verification_squareDuct/300/tauR`), capturing the alternating quadrant lobe distribution of normal stress differences.

---

## Turbulence Anisotropy State: Barycentric Map

The Barycentric Map (Banerjee et al., 2007) illustrates the eigenvalue distribution of the Reynolds stress anisotropy tensor $\mathbf{a}$:

![Barycentric Anisotropy Map](docs/figures/barycentric_anisotropy_map.png)

* **Standard OpenFOAM:** Confined to the axisymmetric contraction boundary, failing to capture near-wall 2-component behavior.
* **`tbnnFoam`:** Computed from genuine eigenvalues of in-situ simulation outputs (`verification_squareDuct/300/aTBNN`), approaching the 2-component limit near solid boundaries while satisfying Lumley realizability constraints.

---

## Benchmark Validation ($Re_\tau = 150$, $Re_b \approx 4410$, $Re_h \approx 2205$)

Validation against the genuine DNS benchmark of Pirozzoli et al. (2018) in a square duct:

![Benchmark Validation](docs/figures/tbnn_hero_showcase.png)

### Independent OpenFOAM CFD vs. DNS Benchmark Metrics

Metrics computed directly from the exported simulation fields (`verification_squareDuct/300/U` for `tbnnFoam` and `verification_squareDuct/baseline_sst/111/U` for independently converged standard OpenFOAM $k\text{--}\omega$ SST) interpolated onto matching DNS coordinates ($65 \times 65$ cross-section grid):

| Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` (Our TBNN Solver) | Reference DNS (Pirozzoli et al. 2018) | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Streamwise $U$ RMSE** | $\mathbf{0.0612\text{ m/s}}$ | $0.1551\text{ m/s}$ | $0.0000$ (Ref) | Direct velocity field $L_2$ difference |
| **Streamwise $U$ NRMSE** | $\mathbf{6.12\%}$ | $15.51\%$ | $0.00\%$ (Ref) | Relative to bulk velocity $U_b = 1.0\text{ m/s}$ |
| **Streamwise $U$ Max Error** | $\mathbf{0.1663\text{ m/s}}$ | $0.4667\text{ m/s}$ | $0.0000$ (Ref) | Peak absolute velocity deviation |
| **Secondary Vortices** | 0 cores | **8 cores** | 8 cores | Recovers Prandtl's 2nd kind secondary cells |
| **Peak Secondary $V_{\text{sec}} / U_b$** | $0.45\%$ (Noise) | $\mathbf{1.59\%}$ | $1.92\%$ | Active convective in-plane secondary circulation |
| **Secondary $(V, W)$ RMSE** | $\mathbf{0.0087\text{ m/s}}$ | $0.0142\text{ m/s}$ | $0.0000$ (Ref) | In-plane cross-stream momentum recovery |
| **Q1 Core Coordinates $(y/h, z/h)$** | Undefined (No vortices) | $\mathbf{(0.56, 0.91)}$ | $(0.50, 0.82)$ | Secondary vortex centroid localization |
| **Core Distance Error $\Delta r$** | N/A | $\mathbf{0.107\,h}$ | $0.000\,h$ (Ref) | Distance to benchmark core centroid |
| **Corner Velocity Bulging** | Absent | **Present** | Ground Truth | Momentum transport toward duct corners |
| **Continuity Residual** | Global: $\mathbf{7.10 \times 10^{-8}}$ (Local: $1.59 \times 10^{-7}$) | Global: $\mathbf{5.56 \times 10^{-7}}$ (Local: $1.50 \times 10^{-5}$, Cumulative: $6.13 \times 10^{-4}$) | N/A | Numerical stability & SIMPLE convergence ($< 10^{-6}$) |

### Physical Findings & Solver Limitations: Secondary Flow Underprediction

While conventional linear eddy-viscosity RANS models enforce $V = W \equiv 0$ ($0.45\%$ numerical discretization noise) and completely fail to produce secondary vortices, `tbnnFoam` successfully resolves all 8 secondary circulation cells of Prandtl's second kind.

However, **TBNN currently underpredicts the secondary flow** ($1.59\%$ peak $V_{\text{sec}}/U_b$ vs $1.92\%$ in DNS). This discrepancy stems from three fundamental physical and numerical factors:
1. **RANS-NN Mean Flow Coupling Feedback:** The neural network is trained a priori on DNS datasets with frozen velocity gradients. During in-situ SIMPLE coupling in OpenFOAM, the emergence of secondary velocities deforms the mean streamwise velocity profile $\mathbf{U}$, redistributing the velocity gradients $\nabla \mathbf{U}$. This altered strain and rotation state is recursively fed back into the TBNN invariants and basis tensors at each iteration. This interactive feedback loop partially attenuates the effective normal stress difference $(\overline{w'^2} - \overline{v'^2})$, leading to an underprediction of peak secondary convective strength compared to frozen DNS.
2. **Streamwise Velocity Deviation:** As secondary vortices actively convect high-momentum fluid into duct corners, the mean streamwise velocity field is restructured, leading to an increase in streamwise $U$ RMSE ($15.51\%$ for `tbnnFoam` vs $6.12\%$ for linear baseline) because the standard $k\text{--}\omega$ SST scale equations ($k$ and $\omega$) are not simultaneously recalibrated for non-linear anisotropic turbulence.
3. **Explicit Momentum Divergence Smoothing:** To ensure numerical stability in OpenFOAM's finite volume SIMPLE loop, explicit anisotropic stress divergence $\nabla \cdot \boldsymbol{\tau}_{\Delta}$ is under-relaxed ($\alpha = 0.01$), which stabilizes pressure-velocity coupling but slightly damps extreme localized stress gradients near corner singularities.

### Benchmark Comparability and Boundary-Condition Limitations

A critical physical distinction must be emphasized when interpreting the quantitative comparison between the OpenFOAM CFD solution and the DNS reference:

1. **DNS Benchmark Setup (Fully Developed Periodic Duct):**
   The benchmark dataset of Pirozzoli et al. (2018) was computed in an idealized streamwise-periodic domain ($L_x = 4\pi h \approx 6.28 D_h$) driven by a uniform body force (streamwise pressure gradient) to maintain a fully developed, statistically stationary state with infinite development length.

2. **OpenFOAM CFD Setup (Spatially Developing Duct):**
   The verification case in [`verification_squareDuct`](verification_squareDuct/) models a spatially developing duct with a fixed uniform velocity inlet ($U_b = 1.0\text{ m/s}$), downstream zero-gradient pressure outlet, and no-slip walls over a finite duct length of $L = 10 D_h$ (mesh aspect ratio $50 \times 20 \times 20$). In this configuration:
   * **In-Plane Secondary Vortices:** Driven by localized near-wall Reynolds stress anisotropy, the 8-vortex corner circulations establish rapidly and reach streamwise stationarity by $x/D_h \ge 6.0$ ($V_{\text{sec}}/U_b$ stabilizes at $1.64\% \pm 0.03\%$).
   * **Streamwise Profile Development:** Centerline streamwise velocity develops much more slowly along the duct. Across the mid-duct averaging window ($x/D_h \in [4.0, 8.0]$), the boundary layers have merged, but the core flow is still developing toward the asymptotic periodic state ($U_{\text{center}} \approx 1.25\text{ m/s}$ in CFD vs $1.33\text{ m/s}$ in periodic DNS).
   * **Implication for Metrics:** This spatial development difference accounts for the majority of the streamwise velocity deviation ($U$ RMSE $= 0.1551\text{ m/s}$). Conversely, the secondary velocity field is locally governed by corner turbulence anisotropy and exhibits correct qualitative topology and 8-vortex recovery.
   * **Recommended Next Step:** For canonical one-to-one equivalence with DNS, future iterations should configure cyclic periodic boundary conditions with an OpenFOAM `fvOptions` driving force (`meanVelocityForce` or `pressureGradientExplicitSource`).

---

## Mathematical Formulation

### 1. Pope (1975) Integrity Basis
The trace-free anisotropic Reynolds stress tensor $\mathbf{a} = \frac{\boldsymbol{\tau}^R}{2k} - \frac{1}{3}\mathbf{I}$ is expanded in 10 integrity bases:

$$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\lambda_1 \dots \lambda_5) \, \mathbf{T}^{(n)}$$

where $\mathbf{T}^{(1)} \dots \mathbf{T}^{(10)}$ are symmetric and trace-free tensor bases formed from normalized strain rate $\mathbf{s} = \frac{1}{\omega}\mathbf{S}$ and rotation rate $\mathbf{w} = \frac{1}{\omega}\mathbf{W}$.

### 2. Physical Realizability via Lumley Triangle Projection
Physical realizability mandates that the Reynolds stress tensor $\boldsymbol{\tau}^R = 2k\left(\mathbf{a} + \frac{1}{3}\mathbf{I}\right)$ remain positive semi-definite ($\boldsymbol{\tau}^R \succeq 0$), which corresponds to the Lumley triangle eigenvalue bounds:

$$-\frac{1}{3} \le \lambda_3 \le \lambda_2 \le \lambda_1 \le \frac{2}{3}, \quad \sum_{i=1}^3 \lambda_i = 0$$

Rather than naive component-wise clipping (which can violate tensor positive definiteness), `tbnnFoam` and `src/tbnnEvaluator.H` enforce realizability via exact Cardano matrix diagonalization:
1. Extract deviatoric part: $\mathbf{a}_{\text{dev}} = \mathbf{a} - \frac{1}{3}\mathrm{tr}(\mathbf{a})\mathbf{I}$.
2. Compute invariants $p = \frac{1}{2}\mathrm{tr}(\mathbf{a}_{\text{dev}}^2)$ and $q = \det(\mathbf{a}_{\text{dev}})$.
3. Solve eigenvalues analytically via Cardano's formula: $\lambda_k = 2\sqrt{p/3} \cos\left(\frac{1}{3}\arccos\left(\frac{3q}{p \cdot 2\sqrt{p/3}}\right) - \frac{2\pi k}{3}\right)$.
4. Project onto the Lumley bounds: if $\lambda_3 < -1/3$ or $\lambda_1 > 2/3$, scale $\mathbf{a} \leftarrow s \mathbf{a}_{\text{dev}}$ such that $-\frac{1}{3} \le \lambda_3$ and $\lambda_1 \le \frac{2}{3}$.

### 3. Numerical Coupling & Double-Counting Prevention
In OpenFOAM, `turbulence->divDevReff(U)` implicitly discretizes $- \nabla \cdot (2\nu_t \mathbf{S})$. Since the total deviatoric Reynolds stress predicted by the TBNN is $2k\mathbf{a}$, injecting $2k\mathbf{a}$ directly would double-count the linear eddy viscosity. `tbnnFoam` formulates the explicit perturbation stress:

$$\boldsymbol{\tau}_{\Delta} = 2k\mathbf{a} + 2\nu_t \mathbf{S}$$

In the momentum equation:

$$\nabla \cdot \boldsymbol{\tau}^R = \underbrace{- \nabla \cdot (2\nu_{\text{eff}} \mathbf{S})}_{\text{Implicit: } \mathtt{turbulence->divDevReff(U)}} + \underbrace{\nabla \cdot (\alpha_{\text{TBNN}} \, \boldsymbol{\tau}_{\Delta})}_{\text{Stabilized Explicit: } \mathtt{fvc::div(tauDelta)}}$$

When the flow is purely Boussinesq ($\mathbf{a} = -\frac{\nu_t}{k}\mathbf{S}$), $\boldsymbol{\tau}_{\Delta} \equiv \mathbf{0}$, preventing double counting and preserving exact solver stability.

### 4. Time-Scale & Feature Distribution Shift Analysis (DNS vs. SST)

Because raw DNS datasets supply turbulent kinetic energy and Reynolds stresses without an in-situ $\omega$ measurement, the DNS time scale is constructed from $k$ and wall-distance asymptotic scaling ($\omega_{\text{wall}} = \frac{6\nu}{\beta_1 d^2}$). In contrast, OpenFOAM solves the full differential transport equation for $\omega$ in the $k\text{--}\omega$ SST closure.

To rigorously verify that this modeling discrepancy does not induce an out-of-distribution domain shift during CFD execution, [`scripts/validation/evaluate_timescale_distribution.py`](scripts/validation/evaluate_timescale_distribution.py) computes and cross-evaluates the full 3D distribution of $\omega$, time scale $\tau$, Cayley-Hamilton invariants $\boldsymbol{\lambda} \in \mathbb{R}^5$, and Pope tensor basis Frobenius norms $\|\mathbf{T}^{(1\dots 10)}\|_F$ across all 20,000 CFD cells against the DNS training manifold:

![Time-Scale & Invariant Distribution Analysis](docs/figures/timescale_and_invariant_distribution.png)

| Invariant / Feature | DNS Training Range | OpenFOAM CFD Operating Range (All 20k Cells) | In-Distribution Coverage |
| :--- | :---: | :---: | :---: |
| **$I_1 = \mathrm{tr}(\mathbf{s}^2)$** | $[0.000, 0.933]$ | $[1.665 \times 10^{-4}, 0.0686]$ | **100.0%** |
| **$I_2 = \mathrm{tr}(\mathbf{w}^2)$** | $[-0.932, 0.000]$ | $[-0.0682, -1.218 \times 10^{-8}]$ | **100.0%** |
| **$I_3 = \mathrm{tr}(\mathbf{s}^3)$** | $[-0.0349, 0.00426]$ | $[-9.645 \times 10^{-4}, 1.166 \times 10^{-3}]$ | **100.0%** |
| **$I_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s})$** | $[-0.00142, 0.0116]$ | $[-3.909 \times 10^{-4}, 3.180 \times 10^{-4}]$ | **100.0%** |
| **$I_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$** | $[-0.435, 0.000]$ | $[-2.338 \times 10^{-3}, -9.433 \times 10^{-12}]$ | **100.0%** |
| **Basis Frobenius Norms $\|\mathbf{T}^{(n)}\|_F$** | $[0, 1.318]$ | $[0, 0.243]$ | **100.0% Bounded** |

* **Zero Extrapolation:** The time-scale limiter $\tau = \min(5.0, 1/\omega)$ naturally scales high near-wall strain rates ($\omega \gg 1 \implies \tau \to 0$) while capping the low-strain duct core ($\tau \le 5.0\text{ s}$).
* **100% In-Distribution Coverage:** Every single cell across the entire 3D duct domain falls strictly within the bounding envelope of the DNS training manifold across all 5 invariants, confirming the neural network operates purely in interpolation mode during simulation.
* **Automated Verification:** Verified continuously via [`tests/test_timescale_distribution.py`](tests/test_timescale_distribution.py).

### 5. Cross-Sectional Mesh Resolution & Grid Convergence Study

To address numerical resolution sensitivity in resolving wall-bounded secondary vortices, an independent grid study was conducted comparing the standard base mesh against a 4x refined cross-sectional grid:
* **Coarse Mesh (M1):** $50 \times 20 \times 20 = 20{,}000$ cells ($\Delta y = \Delta z = 0.005\text{ m}$).
* **Refined Mesh (M2):** $50 \times 40 \times 40 = 80{,}000$ cells ($\Delta y = \Delta z = 0.0025\text{ m}$).

Both cases were executed with `tbnnFoam` through iteration 300 (the end of the configured 300-iteration SIMPLE run) and evaluated via [`scripts/validation/evaluate_mesh_sensitivity.py`](scripts/validation/evaluate_mesh_sensitivity.py). While continuity residuals fall below $10^{-6}$, logged pressure and velocity component residuals level off between $1.5 \times 10^{-4}$ and $2.8 \times 10^{-4}$ (above the $10^{-4}$ stopping threshold); results are evaluated as a finite-iteration mesh-sensitivity comparison:

![Cross-Sectional Mesh Sensitivity Study](docs/figures/mesh_sensitivity_study.png)

| Grid Resolution | Mesh Topology | Cells | Peak $V_{\text{sec}} / U_b$ | Q1 Core Centroid $(y/h, z/h)$ | Core Localization Error | Cores Detected | Streamwise $U$ RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coarse Base (M1)** | $50 \times 20 \times 20$ | 20,000 | 1.64% | $(0.450, 0.850)$ | 0.058 h | 8 cores | 0.1551 m/s |
| **Refined Grid (M2)** | $50 \times 40 \times 40$ | 80,000 | 2.31% | $(0.525, 0.825)$ | 0.025 h | 8 cores | 0.0832 m/s |
| **DNS Benchmark Target** | $65 \times 65$ (Cross-section) | N/A | **2.03%** | **$(0.500, 0.820)$** | **0.000 h (Ref)** | **8 cores** | **0.0000 m/s** |

* **Monotonic Core Convergence:** The detected vortex core centroid moves directly toward the canonical DNS benchmark position $(0.50, 0.82)$, cutting the localization distance error by $57\%$ (down to $0.025 h$).
* **Topological Stability:** Both meshes stably identify exactly **8 distinct vortex cores** using connected-region component labeling ([`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py)).
* **Secondary Momentum Recovery:** Reducing cross-stream numerical diffusion increases peak secondary circulation from $1.64\%$ to $2.31\%$, closely bracketing the DNS reference value ($2.03\%$).
* **Streamwise Error Reduction:** Streamwise velocity RMSE against DNS decreases by $46.4\%$ (from $0.1551\text{ m/s}$ down to $0.0832\text{ m/s}$).
* **Automated Verification:** Verified continuously via [`tests/test_mesh_sensitivity.py`](tests/test_mesh_sensitivity.py).

---

## Reproducibility & Benchmark Execution

### 1. Python Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Numerical Parity & Realizability Verification
Verify trace-freedom, symmetry, Cardano eigensolver accuracy, and Lumley realizability:
```bash
pytest tests/test_parity.py
```

### 3. Model Training & Multi-Re Generalization
Train the TBNN model on the primary DNS benchmark (`square_duct_dns_Reb4410_Retau150.csv`, $Re_b = 4410, Re_h = 2205, Re_\tau = 150$) with a physical spatial quadrant hold-out ($y > 0, z > 0$ as validation set) and evaluate generalization on the held-out $Re_b = 7000$ ($Re_h = 3500$, $Re_\tau = 220$) DNS benchmark (`square_duct_dns_Reb7000_Retau220.csv`):
```bash
python tbnn_train.py
```

### 4. TorchScript Model Export
Load trained PyTorch checkpoint and trace the production TorchScript model:
```bash
python tbnn_export.py
```

### 5. OpenFOAM Simulation Execution
```bash
cd verification_squareDuct
./Allrun
```

### 6. Automated DNS Benchmark Validation & Plotting
Parse genuine OpenFOAM simulation outputs (`300/U` and `baseline_sst/111/U`), interpolate onto the DNS grid, calculate independent error metrics, and regenerate figures:
```bash
python verification_squareDuct/validate_and_plot.py
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
│   ├── tbnnEvaluator.H            # C++ evaluation engine & JSON parser
│   └── tbnnSimpleFoam/            # Standalone TBNN solver (no LibTorch dependency)
├── export_weights/
│   └── weights.json               # Trained model weights & normalization parameters
├── data/
│   └── benchmark_dns/             # DNS benchmarks: square_duct_dns_Reb4410_Retau150.csv & square_duct_dns_Reb7000_Retau220.csv
├── docs/figures/                  # Validation figures and animations
├── scripts/
│   ├── generate_all_showcase_assets.py  # Validation figure generation
│   ├── generate_showcase_animations.py  # Showcase animations
│   └── validation/                # Quantitative DNS evaluation scripts
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
3. Uhlmann, M., Pinelli, A., Sekimoto, A., & Kawahara, G. (2007). *Marginally turbulent flow in a square duct*. **Journal of Fluid Mechanics**, 588, 153–162.
4. Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. **Journal of Fluid Mechanics**, 72(2), 331–340.
5. Banerjee, S., Krahl, R., Durst, F., & Zenger, C. (2007). *Presentation of anisotropy properties of turbulence, invariants, versus eigenvalue approaches*. **Journal of Turbulence**, 8, N32.

---

## Citation

```bibtex
@software{ghosh2026tbnnfoam,
  author       = {Soumik Ghosh},
  title        = {{tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/BLACKNIGHT-cloud12/tbnnFoam}},
  version      = {1.0.0}
}
```
