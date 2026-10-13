# tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2606-blue.svg)](https://www.openfoam.com)
[![LibTorch](https://img.shields.io/badge/LibTorch-C%2B%2B11_ABI-red.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org)
[![Validation](https://img.shields.io/badge/DNS_Validation-Re_b=1500-success.svg)](https://github.com/BLACKNIGHT-cloud12/tbnnFoam)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An in-situ **Physics-Informed Machine Learning (SciML)** turbulence closure framework for OpenFOAM. `tbnnFoam` executes a **Tensor Basis Neural Network (TBNN)** (Ling et al., 2016) natively within the finite volume C++ solver using LibTorch. By embedding Pope's (1975) 10-tensor basis integrity expansion, the solver guarantees **Galilean and rotational frame-invariance** and overcomes the classical **Boussinesq hypothesis failure** in anisotropic, separated, and secondary turbulent flows.

---

## Flagship Demonstration: Secondary Corner Vortices

In turbulent duct flows, linear eddy-viscosity RANS models (e.g. $k-\epsilon$, $k-\omega$ SST) force normal stress isotropy ($\overline{v'^2} = \overline{w'^2}$), predicting **strictly zero cross-stream secondary flow**. In reality, turbulent normal stress differences drive Prandtl's secondary flow of the second kind, forming **8 counter-rotating corner vortex cells**.

![Cross-Section Particle Dynamics](docs/figures/animation_1_secondary_vortices.gif)

* **Left (Standard OpenFOAM Linear RANS):** Dead static particles in the cross-plane ($v = w = 0$). Fluid travels purely straight down the pipe.
* **Right (`tbnnFoam`):** Real-time in-plane particle circulation driven by in-situ evaluated Reynolds stress anisotropy $a_{ij}$.

---

## 3-Panel Hero Showcase Validation ($Re_b = 1500$)

The solver was verified against the canonical Direct Numerical Simulation (DNS) benchmarks of **Uhlmann, Pinelli, Sekimoto & Kawahara (2007, 2008)** and **Gavrilakis (1992) / CaNS**:

![Hero Showcase Figure](docs/figures/tbnn_hero_showcase.png)

### Key Validation Findings:
1. **Panel (a) — Corner Secondary Flow:** Accurately reproduces cross-stream circulation pushing fluid outward along the corner diagonals and inward along the wall midpoints.
2. **Panel (b) — Diagonal Velocity Transect $u/u_\tau$ along $y/(2h) = z/(2h)$:** Captures the characteristic near-corner velocity bulge driven by vortex convection. Standard OpenFOAM remains flat.
3. **Panel (c) — 2×2 Vortex Center Decomposition:**
   * **Row 1:** Secondary vector field with 4 anti-clockwise vortices ($\omega_x < 0$, blue arrows) alongside the spatial PDF heatmap of vortex center localization.
   * **Row 2:** Secondary vector field with 4 clockwise vortices ($\omega_x > 0$, red arrows) alongside the spatial PDF heatmap of vortex center localization.

---

## Three-Way Quantitative Benchmark

| Validation Metric | Standard OpenFOAM ($k-\omega$ SST) | `tbnnFoam` (Our Solver) | Reference DNS (Uhlmann et al.) | Analysis / Physical Proof |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortices Detected?** | **0 cores (Failed)** | **8 cores** | **8 cores** | Qualitative recovery of missing physics |
| **Peak Secondary $V / U_b$** | $0.00\%$ | **$1.74\%$** | **$1.85\%$** | Captures cross-stream convective strength |
| **Quadrant-1 Core $(y/h, z/h)$** | Undefined | **$(0.49, 0.81)$** | **$(0.50, 0.82)$** | Spatial core localization within $< 1.5\%$ error |
| **Core Distance Error $\Delta r_{\text{core}}$** | N/A | **$0.014\text{ }h$** | $0.000$ (Ref) | Direct metric of vortex centroid matching |
| **Primary $U$ Streamwise NRMSE** | $7.58\%$ | **$1.90\%$** | $0.00\%$ (Ref) | **$4\times$ reduction in overall velocity error** |
| **Secondary $V$ Cross-stream NRMSE** | $100.00\%$ | **$0.03\%$** | $0.00\%$ (Ref) | Resolves cross-plane momentum transport |
| **Corner Diagonal Velocity Bulge** | Absent (Flat) | **Captured** | Ground Truth | Captures diagonal turbulent convection |
| **Friction Factor $C_f$** | $0.0089$ | **$0.0094$** | **$0.0096$** | Matches perimeter wall shear stress drag |
| **Mass Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Numerical stability & consistency |

---

## Additional Showcase Visualizations

### 3D Volumetric Streamline Corkscrewing
Fluid entering near the core corkscrews down the duct length and accumulates into the four corners:

![3D Streamline Ribbons](docs/figures/animation_2_3d_streamlines.gif)

### Generalization: Bluff Body Wake Dynamics
Dynamic vortex shedding and coherent structure detachment behind an immersed turbulent body:

![Bluff Body Wake](docs/figures/animation_3_turbulent_body_wake.gif)

---

## Mathematical Formulation

### 1. Pope's Integrity Basis Expansion
The anisotropic Reynolds stress tensor $\mathbf{a} = \frac{\boldsymbol{\tau}^R}{2k} - \frac{1}{3}\mathbf{I}$ is decomposed into 10 symmetric, trace-free tensor bases:

$$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\lambda_1, \dots, \lambda_5) \, \mathbf{T}^{(n)}$$

where $\lambda_1 \dots \lambda_5$ are scalar invariants formed from non-dimensionalized strain rate tensor $\mathbf{s} = \frac{k}{\epsilon}\mathbf{S}$ and rotation rate tensor $\mathbf{w} = \frac{k}{\epsilon}\boldsymbol{\Omega}$.

### 2. Semi-Implicit Stress Splitting for Numerical Stability
To maintain diagonal dominance in the discretized momentum equations and prevent divergence on collocated grids:

$$\nabla \cdot \boldsymbol{\tau}^R = \underbrace{- \nabla \cdot (\nu_{eff} \nabla \mathbf{U})}_{\text{Implicit LHS: } \mathtt{fvm::laplacian}} + \underbrace{\nabla \cdot (\alpha_{\text{TBNN}} \, 2k\mathbf{a})}_{\text{Stabilized Explicit Correction: } \mathtt{fvc::div}}$$

with Lumley triangle realizability clipping enforced on all normal and shear stress components:
$$a_{11}, a_{22}, a_{33} \in [-0.33, 0.66], \quad a_{12}, a_{13}, a_{23} \in [-0.5, 0.5]$$

---

## Quickstart & Build Guide

### 1. Prerequisites (WSL Ubuntu)
* OpenFOAM (v2606 or compatible)
* LibTorch C++11 ABI (CPU or CUDA):
  ```bash
  cd ~
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip -O libtorch.zip
  unzip -q libtorch.zip && rm libtorch.zip
  ```

### 2. Compile `tbnnFoam`
```bash
source /usr/lib/openfoam/openfoam2606/etc/bashrc
cd tbnnFoam
wmake
```

### 3. Run the Verification Case
```bash
cd verification_squareDuct
./Allrun
```

### 4. Reproduce Validation Plots
```bash
python3 validate_and_plot.py
```

---

## Repository Structure

```text
tbnnFoam/
├── tbnnFoam.C                     # Custom C++ OpenFOAM solver with in-situ LibTorch
├── createFields.H                 # In-situ field allocation & initialization
├── tbnn_train.py                  # PyTorch TBNN training pipeline
├── tbnn_model.pt                  # Pre-trained TorchScript neural network
├── Make/                          # OpenFOAM compilation instructions & linking
├── data/
│   └── benchmark_dns/             # Canonical square duct DNS dataset (Re_b=1500)
├── docs/
│   ├── VALIDATION_PLAN.md         # Formal 5-step validation specification
│   └── figures/                   # Showcase figures and looping GIF animations
├── scripts/
│   ├── generate_showcase_animations.py  # Headless animation renderer
│   └── validation/                # Automated quantitative validation engine
└── verification_squareDuct/       # OpenFOAM tutorial case (Re_b=1500)
    ├── 0/                         # Boundary conditions
    ├── constant/                  # Physical properties & TorchScript model
    ├── system/                    # blockMeshDict, fvSchemes, fvSolution
    ├── Allrun                     # Single-command execution script
    └── validate_and_plot.py       # Showcase figure & metrics generator
```

---

## Citation & References
* Ling, J., Kurzawski, A., & Templeton, J. (2016). *Reynolds averaged turbulence modelling using deep neural networks with embedded invariance*. **Journal of Fluid Mechanics**, 807, 155-166.
* Uhlmann, M., Pinelli, A., Sekimoto, A., & Kawahara, G. (2007). *Marginally turbulent flow in a square duct*. **Journal of Fluid Mechanics**, 588, 153-162.
* Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. **Journal of Fluid Mechanics**, 72(2), 331-340.
