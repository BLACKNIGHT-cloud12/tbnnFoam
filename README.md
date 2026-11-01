# tbnnFoam

An in-situ Tensor Basis Neural Network (TBNN) turbulence closure for OpenFOAM, designed to capture turbulence-driven secondary flows of the second kind in non-circular ducts.

Developed by **Soumik Ghosh**

<div align="left">

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2012%20--%20v2606-004C87.svg?logo=openfoam&logoColor=white)](https://www.openfoam.com)
[![CI](https://github.com/BLACKNIGHT-cloud12/tbnnFoam/actions/workflows/ci.yml/badge.svg)](https://github.com/BLACKNIGHT-cloud12/tbnnFoam/actions/workflows/ci.yml)
[![CI Tests](https://img.shields.io/badge/CI%20Tests-18%20passed%2C%203%20skipped-2ea44f.svg)](tests/)
[![Local Tests](https://img.shields.io/badge/Local%2FWSL%20Tests-19%20passed%2C%202%20skipped-2ea44f.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12%20%28tested%29-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Built and tested on OpenFOAM v2606; the native-evaluator source is C++11-compatible for older OpenFOAM toolchains.*

</div>

---

## What is this project?

Standard linear eddy-viscosity RANS models (such as standard $k\text{--}\varepsilon$ and $k\text{--}\omega$ SST) rely on the Boussinesq hypothesis:

$$\boldsymbol{\tau}^R = 2\nu_t \mathbf{S} - \frac{2}{3}k\mathbf{I}$$

Because the turbulent viscosity $\nu_t$ is a scalar, linear models assume that turbulence is isotropic in the cross-stream plane ($\overline{v'^2} = \overline{w'^2}$). 

In non-circular ducts, however, turbulent secondary flows of the second kind (Prandtl, 1926) are driven directly by spatial gradients of turbulent normal stress differences:

$$\mathbf{U} \cdot \nabla \omega_x \approx \frac{\partial^2}{\partial y \partial z}\left(\overline{w'^2} - \overline{v'^2}\right) + \left(\frac{\partial^2}{\partial z^2} - \frac{\partial^2}{\partial y^2}\right)\overline{v'w'}$$

Because linear models force $\overline{w'^2} - \overline{v'^2} = 0$, the production of axial vorticity from normal-stress anisotropy is identically zero. As a result, the baseline linear $k\text{--}\omega$ SST closure in standard OpenFOAM tested in this study predicts an essentially unidirectional axial flow ($V \approx 0, W \approx 0$), failing to reproduce the eight corner counter-rotating secondary recirculation cells that transport momentum and heat into duct corners.

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

Parity between both engines is continuously tested and verified in [`tests/test_parity.py`](tests/test_parity.py), including numerical parity of the momentum-equation coupling itself: the `test_momentum_coupling_parity` test assembles the fixed solver's momentum equation on a mesh with the real `kOmegaSST` model and verifies that (i) when the network outputs the Boussinesq-consistent anisotropy $\mathbf{a} = -(\nu_t/k)\mathbf{S}$, the equation reduces exactly to the baseline linear model (to $\sim 10^{-17}$ relative), and (ii) the pre-fix assembly using a bare $-\nabla\cdot(\nu_{\mathrm{eff}}\nabla\mathbf{U})$ without the transpose divergence does not.

---

## In-Situ Momentum Coupling Pipeline

At each iteration of the SIMPLE loop, the solver couples the neural network to the flow field:

![Architecture Pipeline](docs/figures/architecture_pipeline.png)

1. **Gradient Extraction:** Reads cell-centered velocity gradients $\nabla \mathbf{U}$, turbulent kinetic energy $k$, and specific dissipation $\omega$.
2. **Dimensionless Invariants & Bases:** Computes the normalized strain rate $\mathbf{s} = \tau \mathbf{S}$ and rotation rate $\mathbf{w} = \tau \mathbf{W}$ using the timescale limiter $\tau = \min(1/\omega, 5.0\text{ s})$. From these, it evaluates the 5 Cayley-Hamilton scalar invariants ($\lambda_1 \dots \lambda_5$) and Pope's (1975) 10 symmetric, trace-free tensor bases ($\mathbf{T}^{(1)} \dots \mathbf{T}^{(10)}$).
3. **Neural Network Evaluation:** Feeds the 5 scalar invariants into the network to predict 10 scalar expansion coefficients $g^{(1)} \dots g^{(10)}$.
4. **Anisotropy Assembly:** Computes the deviatoric anisotropy tensor:
   $$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\boldsymbol{\lambda}) \, \mathbf{T}^{(n)}$$
5. **Realizability Enforcement:** Uses a closed-form Cardano cubic eigensolver to clamp the eigenvalues of $\mathbf{a}$ to the Lumley triangle, ensuring Reynolds-stress realizability without iterative overhead.
6. **Momentum Injection:** Formulates the perturbation stress $\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$ and injects its divergence $\nabla \cdot \boldsymbol{\tau}_\Delta$ into OpenFOAM's momentum predictor (`UEqn.H`).

### Key Implementation Safeguards

* **Preventing Double-Counting:** OpenFOAM's `turbulence->divDevReff(U)` implicitly discretizes $-\nabla \cdot (2\nu_t \mathbf{S})$ for numerical stability. Injecting the full Reynolds stress $2k\mathbf{a}$ directly would double-count the linear viscosity. By defining $\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$, the explicit source term represents only the non-linear deviation. When the flow is purely Boussinesq ($\mathbf{a} = -\frac{\nu_t}{k}\mathbf{S}$), $\boldsymbol{\tau}_\Delta \equiv 0$, preserving diagonal dominance.
* **Analytical Cardano Realizability:** If a neural network predicts unphysical stress states, CFD solvers diverge rapidly. Rather than using slow iterative projections, [`src/tbnnEvaluator.H`](src/tbnnEvaluator.H) solves the cubic characteristic equation analytically:
  $$p = \frac{1}{2}\mathrm{tr}(\mathbf{a}^2), \quad q = \det(\mathbf{a}), \quad \lambda_k = 2\sqrt{p/3} \cos\left(\frac{1}{3}\arccos\left(\frac{3q}{2p\sqrt{p/3}}\right) - \frac{2\pi k}{3}\right)$$
  If the smallest eigenvalue violates $\lambda_3 \ge -1/3$, the tensor is scaled radially in $\mathcal{O}(1)$ operations, guaranteeing frame-invariant stability.
* **Turbulent Timescale Limiter:** Capping $\tau = \min(1/\omega, 5.0\text{ s})$ prevents high near-wall strain rates from causing extreme invariant values, bounding the invariant inputs and limiting invariant excursions (joint 5D manifold containment is assessed separately by the distribution-shift diagnostics below).

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
  - Both map onto the same dimensionless equations at approximately matched bulk Reynolds number (a difference of approximately 0.78%; $Re_\tau = 150$ is the DNS reference value) and are formally documented in [`data/benchmark_dns/benchmark_metadata.json`](data/benchmark_dns/benchmark_metadata.json).
* **Canonical DNS Reference (raw-field verified):** The raw Pirozzoli et al. (2018) export carries the DNS's own vorticity field (column 13, `omx/(ub/h)`). [`scripts/validation/verify_dns_reference.py`](scripts/validation/verify_dns_reference.py) recomputes $\omega_x^*$ from the raw velocity columns (interior agreement with the DNS-provided field to **0.43%** of field max) and localizes the primary Q1 anti-clockwise (ACW) core at **$(y/h, z/h) = (0.577, 0.847)$**; the repository detector on the mirrored $65 \times 65$ grid finds $(0.594, 0.844)$ — within $0.017\,h$ of the raw field. An earlier reference value of $(0.50, 0.82)$ (retired) sat $0.081\,h$ from the raw-field peak; all core-distance metrics are now measured against the raw-verified coordinates.

![Benchmark Validation](docs/figures/tbnn_hero_showcase.png)

### Independent Vortex Core Detection Methodology

To ensure that vortex core comparisons are objective and reproducible rather than relying on visual inspection or static annotations, the repository implements an independent algorithmic vortex detector in [`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py):

1. **Discretization & Vorticity Sign Convention:** The non-dimensional axial vorticity field $\omega_x^*$ is computed via second-order central differences on normalized coordinates:
   $$\omega_x^*(y/h, z/h) = \frac{\omega_x}{U_b / h} = \frac{\partial (W / U_b)}{\partial (y / h)} - \frac{\partial (V / U_b)}{\partial (z / h)}$$
   Here, $\omega_x^* > 0$ denotes positive circulation (**anti-clockwise / ACW**), and $\omega_x^* < 0$ denotes negative circulation (**clockwise / CW**).
2. **Boundary Layer Wall-Margin Mask:** An interior mask $|y/h| \le 0.92, |z/h| \le 0.92$ is applied to prevent attached wall shear layers ($\partial U/\partial n$) from registering as secondary flow vortices.
3. **Filtering Fragmented Vorticity from Physical Cores:** To distinguish the 8 primary physical secondary cells from weak numerical artifacts or fragmented sub-structures, the detector enforces three filtering criteria:
   * **Strength Thresholding:** Cells must satisfy $|\omega_x^*| = |\omega_x| / (U_b / h) \ge 0.04$, discarding low-level numerical background noise.
   * **Minimum Cluster Area:** Connected components (identified via 4-connectivity labeling with `scipy.ndimage.label`, explicit cross-shaped structure) must contain at least 4 contiguous cells (`min_area = 4`), eliminating single-cell numerical spikes.
   * **Non-Maximum Suppression (Distance Separation):** A separation radius `min_separation = 0.20 h` is enforced between candidate cores of the same sign, merging fragmented sub-peaks into the dominant local vortex core.
   * **High-Reynolds-Number Case Study Thresholds:** the $Re_b = 17{,}800$ case study applies stricter detection thresholds (strength $\ge 0.15$, separation $\ge 0.25\,h$; area and margin unchanged) to suppress weak roll-up structures during streamwise development.
4. **Vortex Center Extraction & Matching Rule:** The detected core coordinate $(y_{\text{core}}, z_{\text{core}})$ is the location of the maximum $|\omega_x|$ within each retained component. In Quadrant 1 ($y > 0, z > 0$), which contains one counter-rotating vortex pair, the detected ACW core closest to the raw-field-verified DNS reference $(0.577, 0.847)$ in Euclidean distance is matched for quantitative tracking:
   $$\Delta r = \sqrt{(y_{\text{CFD}} - 0.577)^2 + (z_{\text{CFD}} - 0.847)^2} \quad [h]$$

Reproduce the detection and metrics directly on the raw CFD outputs:
```bash
python scripts/validation/evaluate_mesh_sensitivity.py
python scripts/validation/evaluate_validation_plan.py
```

### Quantitative Comparison & Error Definitions

Metrics are evaluated on both the native CFD mesh cross-section and the full $65 \times 65 = 4{,}225$ uniform benchmark grid $(y/h, z/h) \in [-1, 1] \times [-1, 1]$ (sampled over the downstream averaging window $x/D_h \in [4.1,\, 7.9]$, i.e. $x/h \in [8.2,\, 15.8]$, cell-averaged over streamwise cells 20–40 of 50), comparing `tbnnFoam` (`verification_squareDuct/300/U`) against converged baseline $k\text{--}\omega$ SST (`verification_squareDuct/baseline_sst/111/U`) and reference DNS:

* **Streamwise Velocity Normalization:** Normalized by the bulk mean velocity $U_{\text{ref}} = U_b = 1.00\text{ m/s}$:
  $$\mathrm{RMSE}_U = \sqrt{\frac{1}{N}\sum_{i=1}^N (U_{\text{CFD}, i} - U_{\text{DNS}, i})^2} = 0.0934\text{ m/s}, \quad \mathrm{NRMSE}_U = \frac{\mathrm{RMSE}_U}{U_b} \times 100\% = 9.34\%$$
* **Secondary Velocity RMSE:** Evaluated over all cross-stream velocity components:
  $$\mathrm{RMSE}_{\text{sec}} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left[(V_{\text{CFD}, i} - V_{\text{DNS}, i})^2 + (W_{\text{CFD}, i} - W_{\text{DNS}, i})^2\right]} = 0.0261\text{ m/s}$$

| Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` (Native Mesh / DNS Grid) | Reference DNS (Pirozzoli et al. 2018) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortex Cores** | 0 cores (Failed) | **8 cores (Detected)** | 8 cores (Canonical) | Recovers Prandtl's 8 corner vortex pairs |
| **Peak Secondary Speed ($V_{\text{sec}} / U_b$)** | 0.45% (small residual) | **7.99% (native) / 6.38% (DNS grid)** | 1.93% | Non-zero secondary flow generation |
| **Q1 Core Location $(y/h, z/h)$** | Undefined (No vortices) | **(0.850, 0.350) / (0.91, 0.78)** | (0.577, 0.847) | Native discrete cell center vs. $65\times65$ interpolated peak |
| **Core Distance Error** | N/A | **0.567 h (native) / 0.336 h (DNS grid)** | 0.000 h (Ref) | Distance to raw-field-verified core location |
| **Streamwise $U$ RMSE** | 0.0616 m/s (6.16%) | **0.0934 m/s (9.34%)** | 0.0000 m/s (Ref) | Full cross-section NRMSE ($U_{\text{ref}} = 1.00\text{ m/s}$) |
| **Secondary $(V, W)$ RMSE** | 0.0087 m/s | **0.0261 m/s** | 0.0000 m/s (Ref) | In-plane cross-stream velocity recovery |
| **Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Both meet numerical steady-state convergence |

> [!NOTE]
> **Reconciliation of Native Mesh vs. Interpolated Grid Metrics:**
> On the native coarse CFD mesh ($20 \times 20$ cross-section), the discrete cell centers have spacing $\Delta y/h = \Delta z/h = 0.10$. Connected-component labeling identifies the discrete cell center $(0.850, 0.350)$ as the cell-centered peak, giving a core localization error of $0.567\,h$ and peak secondary velocity $7.99\%$. When the velocity field is bilinearly interpolated onto the $65 \times 65$ DNS grid ($\Delta y/h = \Delta z/h = 0.03125$), the resolved peak shifts to $(0.91, 0.78)$, yielding a distance error of $0.336\,h$ and peak secondary velocity $6.38\%$. Both evaluations are generated directly and reproducibly by `scripts/validation/evaluate_mesh_sensitivity.py` (native mesh) and `scripts/validation/evaluate_validation_plan.py` (interpolated DNS grid).

> [!NOTE]
> **Committed Snapshot Provenance:**
> The committed field set `verification_squareDuct/300/` is the canonical solution snapshot from which every metric and figure in this README is reproducibly generated by the validation scripts. The tracked `log.tbnnFoam` in the same case directory records a separate solver execution that reported convergence in 287 iterations, whereas the snapshot corresponds to the run carried to `endTime = 300`; the two artifacts should therefore not be assumed to originate from the same execution. (The $Re_b = 17{,}800$ case is internally consistent: its log and committed snapshot both correspond to the 294-iteration converged run.)

> **Reference-Value Provenance & Convergence Status:**
> **(a) Peak secondary velocity.** Re-deriving the DNS target from the raw exports shows the shipped $Re_b = 4410$ cross-section peaks at **1.93%** of $U_b$ (1.92% after bilinear resampling to $65 \times 65$), while the 2.03% figure previously quoted as the $Re_b = 4410$ reference matches the raw $Re_b = 7000$ ($Re_\tau = 220$) export exactly. Tables now quote the raw-verified **1.93%** for the $Re_b = 4410$ benchmark; the $Re_b = 17{,}800$ comparison range (2.03%–2.30%) is unchanged, being anchored to the higher-Reynolds-number asymptotic data.
> **(b) Vortex coordinates.** The raw export carries the DNS's own vorticity column; recomputing $\omega_x^*$ from the raw velocities agrees with it to 0.43% (interior), and the raw-field Q1 ACW core sits at $(0.577, 0.847)$ — the repository detector on the $65 \times 65$ grid finds $(0.594, 0.844)$, within $0.017\,h$. The retired $(0.50, 0.82)$ reference was $0.081\,h$ from the raw field; all core-distance metrics in this README are now measured against the raw-verified coordinates (which *reduces* the reported CFD localization errors).
> **(c) Solver convergence.** The tracked `log.tbnnFoam` run satisfied `residualControl` at iteration 287 with initial residuals of $3\times10^{-5}$ (U), $6\times10^{-6}$ (k), $1\times10^{-6}$ (ω) and $p$ marginally at the $10^{-4}$ threshold. A two-iteration continuation probe of the committed `300/` snapshot — restart-capable since the relaxed TBNN stress field is now read back (`READ_IF_PRESENT`) — yields initial residuals of $U \approx 1\times10^{-5}$ and $p \approx 1.7$–$2.4\times10^{-4}$. The snapshot is therefore documented as a **finite-iteration stationary state sitting at the pressure-residual stopping threshold**, not an asymptotically iterated solution; the stability of the reported 8-vortex topology and metrics under this residual level is separately evidenced by the mesh-refinement study.

---

## Honest Technical Discussion: Physical Validation & Model Limitations

### Representational Capability vs. Physical Realization
Generating non-zero normal stress differences $(\overline{w'^2} - \overline{v'^2} \ne 0)$ is a necessary *mathematical representational capability* of non-linear tensor basis closures beyond the linear Boussinesq hypothesis. However, non-zero normal stresses alone do not prove that the model has successfully overcome the Boussinesq limitation in the physical solution. 

The physical validation rests on coupling the anisotropic stress divergence $\nabla \cdot \boldsymbol{\tau}_\Delta$ into OpenFOAM's momentum predictor and demonstrating that:
1. The coupled pressure-velocity loop spontaneously generates, sustains, and organizes into the canonical **8-vortex secondary flow pattern** in the correct quadrant orientations without numerical divergence.
2. Under cross-stream grid refinement, peak secondary circulation trends from $7.99\%$ toward the raw-field DNS benchmark of $1.93\%$, and core localization error drops from $0.567\,h$ down to $0.447\,h$.
3. Standard linear $k\text{--}\omega$ SST with Boussinesq closure completely fails this test, yielding identically zero vortex generation ($0$ cores detected).

### Understanding Model Discrepancies and Limitations:
1. **Developing Spatial Duct vs. Streamwise-Periodic Box:**
   The reference DNS benchmark of Pirozzoli et al. (2018) was simulated in an idealized streamwise-periodic box driven by a body force (representing an asymptotically fully developed duct of infinite length). In contrast, our OpenFOAM simulation is a spatially developing duct ($L = 20h = 1.0\text{ m}$) with a uniform velocity inlet ($U_b = 1.0\text{ m/s}$) and physical boundary layers growing along the walls. 
   Because secondary flow is triggered locally by near-wall corner normal stress differences, the 8 secondary vortex cells establish quickly ($x/h \ge 6$). However, the centerline streamwise core requires a longer development length ($U_{\text{center}} \approx 1.25\text{ m/s}$ in our CFD case vs $1.33\text{ m/s}$ in the periodic DNS). Standard linear $k\text{--}\omega$ SST exhibits slightly lower streamwise error ($6.16\%$) simply because it lacks cross-stream convective momentum transport to the corners, yet it fails to reproduce the secondary-flow topology ($0$ vortex cores).
2. **Neural Network Model-Form Uncertainty:**
   The $9.3\%$ streamwise RMSE is not attributable solely to boundary conditions. The neural network closure is an empirical approximation trained on frozen DNS gradients. In live CFD coupling, the emerging secondary flow deforms the mean velocity field $\mathbf{U}$, modifying the velocity gradients fed back into the network at each SIMPLE iteration. This interactive feedback, combined with the finite expressivity of the MLP, introduces genuine model-form discrepancy.
3. **Modelled $\omega$ Surrogate & Turbulent Timescale Sensitivity:**
   Direct numerical simulation yields exact mean velocity gradients $\nabla \mathbf{U}$ and true Reynolds stress tensors $\boldsymbol{\tau}^R$, but DNS inherently lacks the specific dissipation rate $\omega$ because $\omega$ is a semi-empirical RANS modeling construct rather than a physical Navier-Stokes field.
   In the training dataset, $\omega$ was estimated using an empirical SST surrogate relation ($\omega \approx k^{1/2} / (C_\mu^{1/4} \ell)$ with $C_\mu = 0.09, \ell = \min(\kappa d_w, 0.07 D_h)$). In live OpenFOAM CFD execution, $\omega$ is governed dynamically by its transport equation.
   
   **Influence on Timescale $\tau$:** The turbulent timescale $\tau = \min(1/\omega, 5.0\text{ s})$ directly scales the dimensionless strain $\mathbf{s} = \tau \mathbf{S}$ and rotation $\mathbf{w} = \tau \mathbf{W}$. Consequently:
   * The scalar invariants scale non-linearly with $\tau$: $I_1, I_2 \propto \tau^2$, $I_3 \propto \tau^3$, $I_4 \propto \tau^4$, and $I_5 \propto \tau^5$.
   * Pope's basis tensors scale as $\mathbf{T}^{(1)} \propto \tau$, $\mathbf{T}^{(2..4)} \propto \tau^2$, $\mathbf{T}^{(5..8)} \propto \tau^3$, and $\mathbf{T}^{(9..10)} \propto \tau^4$.
   * As a result, an underestimation of $\omega$ (overestimating $\tau$) strongly amplifies the higher-order tensor bases and accelerates secondary flow intensity, whereas overestimating $\omega$ damps the perturbation stress $\boldsymbol{\tau}_\Delta$ back toward linear RANS. Capping $\tau \le 5.0\text{ s}$ serves as an essential regularizer preventing singular invariant growth near the duct center where $\omega \to 0$.
4. **First-Order Convective Schemes & Absence of Near-Wall Grading:**
   All convective terms in both verification cases use bounded first-order Gauss upwind (`system/fvSchemes`), and both cross-sectional meshes are uniform with no near-wall grading (first-cell $y^+ \approx 7.5$ at $Re_b \approx 4410$ and $\approx 13$ at $Re_b = 17{,}800$, both under wall functions). First-order upwinding introduces numerical diffusion precisely on the $\sim 2\%$-of-$U_b$ secondary-flow signal being validated, and the unclustered near-wall resolution is thin for wall-function RANS at these Reynolds numbers. Neither discretization error source is decomposed out of the reported RMSE figures, so the tabulated errors should be read as combined physical-model-plus-discretization discrepancies rather than closure error alone.
5. **Finite-Difference Training Gradients, Wall-Ring Targets & Reflection-Structured Split:**
   The training dataset (`square_duct_dns_Reb4410_Retau150.csv`) is generated by [`scripts/convert_real_dns.py`](scripts/convert_real_dns.py) from the 64×64 quarter-duct DNS cross-section: the field is symmetry-mirrored to the full duct, bilinearly resampled to 65×65, and the velocity gradients are then recomputed by second-order finite differences on that grid. The network is therefore trained on numerical derivatives of mirrored, resampled data — with the associated near-wall gradient noise — rather than on raw DNS gradients, and $\partial/\partial x \equiv 0$ follows from the streamwise-periodic reference. Two further subtleties: (i) the 256 wall-boundary points have $k$ clipped to $10^{-7}$, producing degenerate targets $\mathbf{a} = -\frac{1}{3}\mathbf{I}$ (trace $-1$) that are not valid anisotropy states for the trace-free tensor basis. These points are retained in the training loss, whose Frobenius form makes the incompatible trace component a constant, zero-gradient offset (since $\mathrm{dev}(-\frac{1}{3}\mathbf{I}) = \mathbf{0}$), so the network is trained only toward the zero trace-free projection $\mathbf{a} \to \mathbf{0}$ there — a marginal bias on the $g$-map near the origin with negligible bulk effect; (ii) the quadrant-1 spatial validation split is weaker than it appears because all five invariants are even in the rotation tensor $\mathbf{w}$, so a reflected quadrant produces identical inputs. The split remains a legitimate test of the basis/sign structure — the odd-in-$\mathbf{w}$ bases $\mathbf{T}^{(2)}, \mathbf{T}^{(5)}, \mathbf{T}^{(7)}, \mathbf{T}^{(8)}, \mathbf{T}^{(10)}$ must reconstruct the reflected stresses from identical coefficients — but it should not be read as an independent-flow generalization test.

---

## Cross-Sectional Mesh Sensitivity Study

A formal Grid Convergence Index (GCI) according to Roache and ASME V&V 20 standards requires at least three geometrically scaled grid levels with constant refinement ratio ($r \ge 1.3$) to evaluate the asymptotic observed order of convergence $p$. 

Here, we conduct an independent **2-level cross-sectional grid resolution sensitivity check** comparing the standard base mesh ($20\text{k}$ cells) and a 4× cross-stream refined mesh ($80\text{k}$ cells) to examine discretization sensitivity and verify that the detected secondary vortices are genuine physical solutions rather than numerical grid artifacts:

* **Coarse Mesh (M1):** $50 \times 20 \times 20 = 20{,}000$ cells ($\Delta y = \Delta z = 0.005\text{ m}$).
* **Refined Mesh (M2):** $50 \times 40 \times 40 = 80{,}000$ cells ($\Delta y = \Delta z = 0.0025\text{ m}$).

![Mesh Sensitivity Study](docs/figures/mesh_sensitivity_study.png)

| Grid Resolution | Cells | Peak $V_{\text{sec}} / U_b$ | Q1 Core Location $(y/h, z/h)$ | Core Localization Error | Vortices Detected | Streamwise $U$ RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coarse Base (M1)** | 20,000 | 7.99% | (0.850, 0.350) | 0.567 h | 8 cores | 0.0934 m/s |
| **Refined Grid (M2)** | 80,000 | 2.62% | (0.825, 0.475) | 0.447 h | 8 cores | 0.0857 m/s |
| **Reference DNS Target** | N/A | **1.93%** | **(0.577, 0.847)** | **0.000 h (Ref)** | **8 cores** | **0.0000 m/s** |

* **Secondary Circulation Discretization Sensitivity:** Increasing cross-stream resolution reduces peak secondary speed from $7.99\%$ to $2.62\%$, demonstrating a clear trend toward the DNS benchmark reference value ($2.03\%$).
* **Core Localization Improvement:** The detected Q1 anti-clockwise core location error decreases from $0.567\,h$ down to $0.447\,h$.
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

| Station | $x/D_h$ | $U_c / U_b$ | Raw Max $V_{\mathrm{sec}}/U_b$ | Core Max ($|y/h|,|z/h| \le 0.85$) | Cores Detected | Q1 Peak Location $(y/h, z/h)$ | Distance to DNS $(0.577, 0.847)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | $1.1$ | $1.019$ | $9.63\%$ | $1.91\%$ | 0 (Rollup) | N/A | N/A |
| **#2** | $2.1$ | $1.037$ | $10.43\%$ | $2.80\%$ | 0 (Rollup) | N/A | N/A |
| **#3** | $3.1$ | $1.054$ | $10.43\%$ | $2.39\%$ | 0 (Rollup) | N/A | N/A |
| **#4** | $4.1$ | $1.069$ | $10.20\%$ | $1.95\%$ | 0 (Rollup) | N/A | N/A |
| **#5** | $5.1$ | $1.083$ | $9.92\%$ | $1.87\%$ | 0 (Rollup) | N/A | N/A |
| **#6** | $6.1$ | $1.096$ | $9.69\%$ | $1.84\%$ | **8 cores** | $(0.625, 0.875)$ | $0.056\,h$ |
| **#7** | $7.1$ | $1.108$ | $9.55\%$ | $1.78\%$ | **8 cores** | $(0.575, 0.875)$ | $0.028\,h$ |
| **#8** | $8.1$ | $1.121$ | $9.34\%$ | $1.70\%$ | **8 cores** | $(0.575, 0.875)$ | $0.028\,h$ |
| **#9** | $9.1$ | $1.132$ | $9.39\%$ | $1.67\%$ | **8 cores** | $(0.525, 0.875)$ | $0.059\,h$ (1.2 cells) |
| **DNS Ref** | $\infty$ | $1.33$ | $2.03\% - 2.30\%$ | **$2.03\% - 2.30\%$** | **8 cores** | **$(0.577, 0.847)$** | $\mathbf{0.000\,h}$ (Reference) |

* **Key Physical & Numerical Insights:**
  1. **Downstream Core Speed Underprediction:** Across the downstream developing-flow region with detected secondary-flow vortices ($x/D_h \ge 6.1$), the secondary speed evaluated at the 8 detected core locations settles at **$1.71\%$** (with interior-masked station maxima ranging between **$1.67\%$ and $1.84\%$**). Relative to the lower bound of the cited asymptotic DNS range ($2.03\%$), the station interior maxima are **$9.4\% - 17.7\%$ below** ($15.8\%$ below for the $1.71\%$ core-location mean). This underprediction is consistent with an incompletely developed spatial duct ($U_c/U_b = 1.132$ at $x/D_h = 9.1$ vs. the asymptotic fully developed value $U_c/U_b \approx 1.33$ in periodic DNS). The transient values during early rollup ($2.39\% - 2.80\%$ at $x/D_h = 2 - 3$) reflect developing flow overshoot and should not be confused with fully developed benchmark agreement.
  2. **Core Location Convergence & Resolution Limits:** As the boundary layer develops downstream from $x/D_h = 6.1$ to $9.1$, the detected Quadrant 1 core peak location migrates from $(0.625, 0.875)$ to $(0.525, 0.875)$, with core distance errors of $0.028$–$0.059\,h$ against the raw-field-verified DNS reference $(0.577, 0.847)$. The downstream-averaged section ($x/D_h \in [6.1, 9.1]$) yields a station-averaged core location of $(0.575, 0.875)$ ($\Delta r = 0.028\,h$, $\approx 0.6$ grid cells). While demonstrating recovery of the 8-vortex topology, this spatial localization ($\approx 1.2$ discrete grid cells at station $9.1$ on the $40 \times 40$ mesh with $\Delta y/h = 0.05$) is resolution-limited by cell discretization.
  3. **Proposed Near-Wall Discretization Mechanism:** The raw unmasked grid maximum ($9.3\% - 10.4\%$) occurs strictly in the outermost boundary-layer cell center ($|z/h| = 0.975$). This is framed as a proposed numerical/wall-treatment mechanism where steep wall-normal velocity gradients ($\partial U_x / \partial z \sim 400\text{ s}^{-1}$) interact with explicit anisotropic stress divergence under high-shear wall functions, rather than a confirmed cause, pending full budget decomposition of $\nabla \cdot \boldsymbol{\tau}_\Delta$. Inside the physical vortex core region, secondary circulation remains smooth at $\sim 1.71\%$.

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

This establishes that live CFD operating points remain closely clustered around observed training states (median gap $< 0.25$ standard deviations), indicating limited distributional extrapolation and reducing concern about severe out-of-distribution tensor predictions, while transparently characterizing the joint distribution shift inherent in live Navier-Stokes coupling.

---

## Automated Verification Suite

The repository includes a comprehensive 21-test `pytest` verification suite covering tensor mathematical foundations, coordinate conventions, mesh geometry reconstruction, cross-engine numerical parity, and independent vortex detection algorithms.

The test results reflect the execution environment and available optional dependencies:

1. **GitHub-Hosted CI Runner (`ubuntu-latest`): 18 passed, 3 skipped (0 failures)**
   * **1 test skipped (OpenFOAM C++ toolchain absent):** the compiled-binary test `test_standalone_cpp_numerical_parity` self-skips on hosted runners where OpenFOAM is not installed. Since the v1.0.1 verification-script correction, `test_momentum_coupling_parity` runs in pure Python and no longer requires the toolchain.
   * **2 tests skipped (External DNS data absent):** Invariant bounding tests (`test_cfd_invariants_within_training_envelope` and `test_basis_norms_bounded_by_training_distribution`) self-skip cleanly when upstream benchmark DNS data is not locally supplied.
2. **Local Development Environment (Windows + OpenFOAM-enabled WSL): 19 passed, 2 skipped (0 failures)**
   * The OpenFOAM C++ parity test executes and passes via the local WSL toolchain.
   * The 2 DNS invariant tests skip cleanly unless external DNS data is supplied via `--dns-dir` or `TBNN_DNS_DIR`.
3. **Full Integration Environment (OpenFOAM + External DNS data supplied): 21 passed, 0 skipped**

```bash
pytest -v
```

```text
tests/test_coordinate_convention.py::test_non_square_gradient_ordering PASSED            [  4%]
tests/test_coordinate_convention.py::test_non_square_interpolation_ordering PASSED        [  9%]
tests/test_coordinate_convention.py::test_benchmark_dns_grid_convention PASSED            [ 14%]
tests/test_mesh_reconstruction_verification.py::test_coarse_mesh_geometry_and_ordering PASSED [ 19%]
tests/test_mesh_reconstruction_verification.py::test_fine_mesh_geometry_and_ordering PASSED [ 23%]
tests/test_mesh_sensitivity.py::test_mesh_sensitivity_core_and_velocity PASSED            [ 28%]
tests/test_parity.py::test_basis_trace_freedom PASSED                                      [ 33%]
tests/test_parity.py::test_basis_symmetry PASSED                                           [ 38%]
tests/test_parity.py::test_cardano_eigenvalue_accuracy PASSED                              [ 42%]
tests/test_parity.py::test_realizability_bounds PASSED                                     [ 47%]
tests/test_parity.py::test_evaluator_parity PASSED                                         [ 52%]
tests/test_parity.py::test_torchscript_parity PASSED                                       [ 57%]
tests/test_parity.py::test_json_weights_parity PASSED                                      [ 61%]
tests/test_parity.py::test_standalone_cpp_numerical_parity PASSED (or SKIPPED on CI)       [ 66%]
tests/test_parity.py::test_solver_momentum_assembly_consistency PASSED                     [ 71%]
tests/test_parity.py::test_momentum_coupling_parity PASSED                                 [ 76%]
tests/test_timescale_distribution.py::test_cfd_invariants_within_training_envelope SKIPPED [ 80%]
tests/test_timescale_distribution.py::test_basis_norms_bounded_by_training_distribution SKIPPED [ 85%]
tests/test_vortex_detection.py::test_zero_vorticity_field PASSED                           [ 90%]
tests/test_vortex_detection.py::test_known_four_vortex_field PASSED                        [ 95%]
tests/test_vortex_detection.py::test_known_eight_vortex_field PASSED                       [100%]

================== 18 passed, 3 skipped in GitHub Actions CI ==================
================ 19 passed, 2 skipped in Local WSL/OpenFOAM ===================
```

### Mesh Geometry Reconstruction & Cell Ordering Verification

To eliminate verification limitations associated with hard-coded mesh dimensions and assumed cell ordering:
1. **Direct Geometry Parsing:** [`scripts/validation/evaluate_mesh_sensitivity.py`](scripts/validation/evaluate_mesh_sensitivity.py) and [`scripts/validation/evaluate_validation_plan.py`](scripts/validation/evaluate_validation_plan.py) directly parse OpenFOAM's cell center vector field (`0/C`) generated by `postProcess -func writeCellCentres`.
2. **Explicit Block Ordering Verification:** In [`tests/test_mesh_reconstruction_verification.py`](tests/test_mesh_reconstruction_verification.py), we verify against OpenFOAM's mesh geometry that:
   - Streamwise cells $X$ strictly increment along the fastest axis (axis 2, $N_x = 50$).
   - Horizontal cross-stream cells $Y$ strictly increment along axis 1 ($N_y = 20$ coarse, $40$ fine).
   - Vertical cross-stream cells $Z$ strictly increment along axis 0 ($N_z = 20$ coarse, $40$ fine).
   - True cell center coordinates $y/h$ and $z/h$ derived from the mesh geometry agree with uniform cell-center definitions to within floating-point roundoff ($\Delta_{\max} < 2 \times 10^{-16}$).
This ensures that field reshaping introduces no detectable distortion at floating-point precision into the computed vortex centers and velocity RMSE.

---

## Quick Start: How to Build and Run

> A detailed step-by-step installation and usage guide — prerequisites, both solver build paths, the validation workflow, retraining, and troubleshooting — is in [INSTALL.md](INSTALL.md).

### 1. Python Environment
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Tests
```bash
pytest
```

### 3. Model Training & Export
Pre-trained model weights are bundled (`tbnn_model.pt`, `export_weights/weights.json`). To re-train from scratch using external DNS data:
```bash
# Pass external DNS directory or file:
python tbnn_train.py --dns-dir /path/to/dns_data
# Or set TBNN_DNS_DIR environment variable:
export TBNN_DNS_DIR="/path/to/dns_data"
python tbnn_train.py

# Export standalone weights:
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
├── tbnn_train.py                  # PyTorch model training script (shipped model)
├── tbnn_export.py                 # TorchScript export script
├── tbnn_tensor_math.py            # Shared numpy tensor algebra (invariants, bases, realizability)
├── tbnn_model.pt                  # Serialized TorchScript model
├── Make/                          # wmake configuration for tbnnFoam
├── src/
│   ├── tbnnEvaluator.H            # Standalone C++ evaluation engine & JSON parser
│   └── tbnnSimpleFoam/            # Standalone TBNN solver (no LibTorch dependency)
├── export_weights/
│   └── weights.json               # Exported model weights & normalization parameters
├── data/
│   └── benchmark_dns/             # Canonical benchmark metadata & download/preprocessing instructions
├── docs/
│   ├── figures/                   # Publication-grade validation figures and animations
│   └── report/                    # Compiled project summary PDF reports
├── scripts/
│   ├── generate_all_showcase_assets.py  # Figure generation script
│   ├── generate_concise_pdf.py          # PDF project summary compiler
│   └── validation/                      # Quantitative DNS evaluation & vortex detection
├── tests/                         # 21-test automated verification suite
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
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23799523},
  howpublished = {\url{https://github.com/BLACKNIGHT-cloud12/tbnnFoam}},
  version      = {1.0.1}
}
```


---

## License & Attribution

* **Software:** The `tbnnFoam` codebase, C++ solver implementations, Python training and export scripts, verification test suite, and case setups are released under the [MIT License](LICENSE).
* **Third-Party DNS Benchmark Data:** The Direct Numerical Simulation datasets for square duct flow originate from the Sapienza University of Rome CFD Group database (Pirozzoli et al., *J. Fluid Mech.*, 2018; Vinuesa et al., 2014). To respect upstream intellectual property and redistribution terms, raw DNS data files are not redistributed within this software deposit. Instead, complete instructions and automated ingestion utilities to download and convert the official datasets are provided in [`data/benchmark_dns/README.md`](data/benchmark_dns/README.md). All validation and training scripts accept an external DNS location via `--dns-dir` or the `TBNN_DNS_DIR` environment variable. Any academic publication, report, or comparative study utilizing these datasets must cite Pirozzoli et al. (2018).
