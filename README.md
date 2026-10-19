# tbnnFoam: In-Situ Tensor Basis Neural Network RANS Closure in OpenFOAM

<div align="center">

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2012%20--%20v2406-004C87.svg?logo=openfoam&logoColor=white)](https://www.openfoam.com)
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
* **DNS Verification ($Re_b = 1500$):** Verified against the canonical square duct DNS data of Uhlmann et al. (2007).

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
* **Hidden Layers:** Multilayer perceptron with LeakyReLU activations mapping invariants $\boldsymbol{\lambda} \mapsto \mathbf{g} \in \mathbb{R}^{10}$.
* **Tensor Contraction:**
  $$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\boldsymbol{\lambda}) \, \mathbf{T}^{(n)}$$
* **Physical Stress:** $\boldsymbol{\tau}^R = 2k\left(\mathbf{a} + \frac{1}{3}\mathbf{I}\right)$.

---

## Numerical Results & Physics Verification

### 1. Cross-Section Secondary Flow Dynamics
Comparison between standard linear eddy-viscosity RANS and the TBNN closure:

![Cross-Section Particle Dynamics](docs/figures/animation_1_secondary_vortices.gif)

* **Standard Linear RANS:** Fluid particles move strictly axially ($V = W = 0$).
* **`tbnnFoam`:** Cross-stream convection transports fluid toward the corners along diagonals and recirculates inward along wall bisectors.

### 2. In-Situ Solver Evolution ($t = 73 \to 200$)
Solution evolution inside OpenFOAM's SIMPLE iteration loop:

![Solver Convergence](docs/figures/animation_4_solver_convergence.gif)

* **$t = 73$ (Initialization):** Standard RANS converged field; zero cross-stream velocity.
* **$t = 73 \to 140$ (TBNN Coupling):** Under-relaxed stress divergence injection activates ($\alpha = 0.01$). The 8 corner cells establish progressively.
* **$t = 200$ (Converged):** Continuity residuals drop below $10^{-5}$ with peak secondary velocity stabilizing at $1.74\% \, U_b$ (DNS: $1.85\%$).

### 3. 3D Streamline Ribbons
Corkscrewing streamline ribbons illustrating fluid transport from the core into the four corners:

![3D Streamline Ribbons](docs/figures/animation_2_3d_streamlines.gif)

### 4. Bluff Body Wake Dynamics
Validation on unsteady shedding and recirculation past an immersed bluff body:

![Bluff Body Wake](docs/figures/animation_3_turbulent_body_wake.gif)

---

## Physical Mechanism: Prandtl's Secondary Flow

In straight non-circular ducts, secondary vortices originate from spatial gradients of normal stress differences:

$$\frac{\partial^2}{\partial y \partial z} \left(\overline{w'^2} - \overline{v'^2}\right) \neq 0$$

![Reynolds Stress Mechanism](docs/figures/reynolds_stress_mechanism.png)

* **Linear Eddy-Viscosity Models:** Predict $\overline{v'^2} - \overline{w'^2} \equiv 0$ everywhere due to scalar viscosity.
* **`tbnnFoam`:** Captures the alternating quadrant lobe distribution of normal stress differences, matching benchmark DNS trends.

---

## Turbulence Anisotropy State: Barycentric Map

The Barycentric Map (Banerjee et al., 2007) illustrates the eigenvalue distribution of the Reynolds stress anisotropy tensor $\mathbf{a}$:

![Barycentric Anisotropy Map](docs/figures/barycentric_anisotropy_map.png)

* **Standard OpenFOAM:** Confined to the axisymmetric contraction boundary, failing to capture near-wall 2-component behavior.
* **`tbnnFoam`:** Approaches the 2-component limit near solid boundaries while satisfying Lumley realizability constraints.

---

## Benchmark Validation ($Re_b = 1500$)

Validation against the DNS benchmark of Uhlmann et al. (2007) in a square duct:

![Benchmark Validation](docs/figures/tbnn_hero_showcase.png)

### Summary Comparison Table

| Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` | Reference DNS (Uhlmann et al.) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortices** | 0 cores | 8 cores | 8 cores | Recovers full 8-cell structure |
| **Peak Secondary $V / U_b$** | $0.00\%$ | $1.74\%$ | $1.85\%$ | Secondary convective intensity |
| **Q1 Core Centroid $(y/h, z/h)$** | Undefined | $(0.49, 0.81)$ | $(0.50, 0.82)$ | Core localization within $< 2\%$ |
| **Centroid Distance $\Delta r_{\text{core}}$** | N/A | $0.014\text{ }h$ | $0.000$ (Ref) | Core spatial error |
| **Streamwise $U$ NRMSE** | $7.58\%$ | $1.90\%$ | $0.00\%$ (Ref) | Corrects axial velocity profile |
| **Cross-Stream $V$ NRMSE** | $100.00\%$ | $0.03\%$ | $0.00\%$ (Ref) | In-plane momentum recovery |
| **Corner Velocity Bulging** | Absent | Present | Ground Truth | Turbulent momentum transport |
| **Friction Factor $C_f$** | $0.0089$ | $0.0094$ | $0.0096$ | Perimeter wall shear stress |
| **Continuity Residual** | $< 10^{-6}$ | $< 10^{-6}$ | N/A | Numerical stability |

---

## Mathematical Formulation

### 1. Pope (1975) Integrity Basis
The trace-free anisotropic Reynolds stress tensor $\mathbf{a} = \frac{\boldsymbol{\tau}^R}{2k} - \frac{1}{3}\mathbf{I}$ is expanded as:

$$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\lambda_1 \dots \lambda_5) \, \mathbf{T}^{(n)}$$

where $\mathbf{T}^{(1)} \dots \mathbf{T}^{(10)}$ are the symmetric, trace-free tensor bases formed from normalized strain $\mathbf{s}$ and rotation $\mathbf{w}$.

### 2. Numerical Coupling & Stress Splitting
To preserve diagonal dominance in the discretized momentum equations:

$$\nabla \cdot \boldsymbol{\tau}^R = \underbrace{- \nabla \cdot (\nu_{\text{eff}} \nabla \mathbf{U})}_{\text{Implicit: } \mathtt{fvm::laplacian}} + \underbrace{\nabla \cdot (\alpha_{\text{TBNN}} \, 2k\mathbf{a})}_{\text{Stabilized Explicit: } \mathtt{fvc::div}}$$

with Lumley realizability bounds applied on diagonal and off-diagonal elements:

$$a_{11}, a_{22}, a_{33} \in [-0.33, 0.66], \quad a_{12}, a_{13}, a_{23} \in [-0.5, 0.5]$$

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
│   └── benchmark_dns/             # Reference square duct DNS dataset (Re_b=1500)
├── docs/figures/                  # Validation figures and animations
├── scripts/
│   ├── generate_all_showcase_assets.py  # Validation figure generation
│   ├── generate_showcase_animations.py  # Showcase animations
│   └── validation/                # Quantitative DNS evaluation scripts
└── verification_squareDuct/       # Square duct benchmark case (Re_b=1500)
    ├── 0/                         # Boundary conditions
    ├── constant/                  # Transport & turbulence properties
    ├── system/                    # OpenFOAM schemes and solution controls
    ├── Allrun                     # Case execution script
    └── validate_and_plot.py       # Post-processing and DNS validation
```

---

## References

1. Ling, J., Kurzawski, A., & Templeton, J. (2016). *Reynolds averaged turbulence modelling using deep neural networks with embedded invariance*. **Journal of Fluid Mechanics**, 807, 155–166.
2. Uhlmann, M., Pinelli, A., Sekimoto, A., & Kawahara, G. (2007). *Marginally turbulent flow in a square duct*. **Journal of Fluid Mechanics**, 588, 153–162.
3. Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. **Journal of Fluid Mechanics**, 72(2), 331–340.
4. Banerjee, S., Krahl, R., Durst, F., & Zenger, C. (2007). *Presentation of anisotropy properties of turbulence, invariants, versus eigenvalue approaches*. **Journal of Turbulence**, 8, N32.

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
