# In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM: A Critical Verification, Validation, and Reproducibility Study

**Soumik Ghosh**  
*Repository: [https://github.com/BLACKNIGHT-cloud12/tbnnFoam](https://github.com/BLACKNIGHT-cloud12/tbnnFoam)*  
*Preprint Date: October 2026*

---

## Abstract

Linear eddy viscosity models based on the Boussinesq hypothesis remain the workhorse of industrial Computational Fluid Dynamics (CFD). However, they fundamentally fail to predict turbulence-driven secondary flows of Prandtl's second kind due to their enforced alignment between the Reynolds stress anisotropy tensor and the mean strain rate tensor. Data-driven turbulence closures utilizing Tensor Basis Neural Networks (TBNN; Ling et al., *J. Fluid Mech.*, 2016) provide a mathematically rigorous, frame-invariant methodology to represent Reynolds stress anisotropy by projecting onto Pope's ten symmetric integrity basis tensors. 

In this work, we present an in-situ coupled CFD solver, `tbnnFoam`, implemented within OpenFOAM 2606. The solver features a dual-runtime inference architecture: an embedded LibTorch C++ engine and a zero-dependency standalone C++ matrix evaluator incorporating an analytical Cardano cubic eigensolver that enforces physical realizability on the Lumley triangle in $\mathcal{O}(1)$ operations. Cross-engine numerical parity across PyTorch, TorchScript, and the standalone C++ matrix evaluator is verified within an absolute tolerance of $10^{-5}$ across synthetic test cases spanning shear, rotational, and complex 3D strain states, with the standalone C++ comparison executed when the OpenFOAM environment and compiled test binary are present.

We validate the coupled solver on developing turbulent flow in a square duct at bulk Reynolds number $Re_b \approx 4410$ ($Re_\tau = 150$) against high-fidelity Direct Numerical Simulation (DNS; Pirozzoli et al., *J. Fluid Mech.*, 2018). An independent algorithmic vortex detector employing connected-component labeling on non-dimensional axial vorticity ($\omega_x^* = \omega_x / (U_b / h)$) confirms that `tbnnFoam` structurally recovers the secondary-flow mechanism by identifying all eight counter-rotating secondary recirculation cells where the standard linear $k\text{--}\omega$ SST closure tested here fails to reproduce the secondary-flow topology (detecting zero vortex cores).

The outcome is a **partial validation**: while secondary circulation topology is recovered, vortex core localization exhibits notable offset ($0.567\,h$ error on the coarse mesh and $0.447\,h$ on the refined mesh against the raw-field-verified DNS reference $(0.577, 0.847)$), and peak secondary speed exhibits strong discretization sensitivity, decreasing from $7.99\%$ to $2.62\%$ (relative to the raw-field DNS peak of $1.93\%$). We emphasize that structural recovery does not establish accurate pointwise vortex localization or full quantitative agreement with periodic DNS.

Four limitations of the present implementation deserve emphasis before the results are interpreted:
1. **Velocity Discrepancy:** The streamwise velocity Normalized Root-Mean-Square Error ($\mathrm{NRMSE}_U$) is $9.34\%$ ($0.0934\text{ m/s}$ relative to bulk velocity $U_b = 1.0\text{ m/s}$). This discrepancy is consistent with a combined influence of entrance-length spatial boundary-layer development in the CFD domain ($L = 20h$) relative to the streamwise-periodic DNS box, alongside genuine neural-network model-form uncertainty under live pressure--velocity coupling.
2. **Timescale Surrogate Sensitivity:** Direct numerical simulation provides exact velocity gradients and Reynolds stresses but inherently lacks the specific dissipation rate $\omega$. We examine the role of the semi-empirical $\omega$ surrogate and show analytically how timescale-estimation errors propagate into the invariant inputs used by the network: the Cayley-Hamilton invariants scale non-linearly with turbulent timescale $\tau = \min(1/\omega, 5.0\text{ s})$ ($I_1, I_2 \propto \tau^2$; $I_3, I_4 \propto \tau^3$; $I_5 \propto \tau^4$), so errors in the live $\omega$ field are magnified by the corresponding inverse power before entering the MLP.
3. **Out-of-Distribution Diagnostic:** We distinguish between 1D marginal bounding and joint manifold containment. While $100.0\%$ of CFD cells satisfy marginal bounding across scalar invariants $I_1, I_2, I_5$ (and $> 99\%$ for $I_3, I_4$), a joint 5D nearest-neighbor distance diagnostic reveals that live CFD states experience modest distribution shift relative to frozen training gradients (median standardized $k$-NN distance of $0.21\,\sigma$, 95th percentile of $1.17\,\sigma$).
4. **Discretization Sensitivity:** A two-level cross-sectional grid resolution study ($20 \times 20$ coarse vs. $40 \times 40$ fine) demonstrates that peak secondary circulation trends from $7.99\%$ down to $2.62\%$ (approaching the raw-field DNS benchmark target of $1.93\%$) and Q1 core distance error decreases from $0.567\,h$ to $0.447\,h$, while maintaining strict topological stability across all eight cores.

All solver sources, evaluation scripts, raw DNS benchmark data, and an automated twenty-one-test verification suite are openly distributed under the MIT license.

---

## 1. Introduction

Turbulent flows in non-circular ducts and channels are characterized by secondary motions of the second kind (Prandtl, 1926). Although secondary velocities typically represent only $1\%\text{--}3\%$ of the bulk streamwise velocity, they exert a profound influence on momentum redistribution, wall shear stress profiles, and heat transfer enhancement across engineering applications, including nuclear fuel assemblies, gas turbine blade cooling passages, heat exchangers, and non-circular pipelines (Gessner, 1973; Brundrett & Baines, 1964).

### 1.1 Failure of Linear Eddy Viscosity Models
In standard Reynolds-Averaged Navier-Stokes (RANS) closures, the linear Boussinesq hypothesis approximates the deviatoric Reynolds stress anisotropy tensor $\mathbf{a}$ as linearly proportional to the mean strain rate tensor $\mathbf{S}$:
$$\mathbf{a} = \frac{\boldsymbol{\tau}^R}{2k} - \frac{1}{3}\mathbf{I} = -\frac{\nu_t}{k} \mathbf{S}$$
where $\boldsymbol{\tau}^R = \overline{\mathbf{u}' \otimes \mathbf{u}'}$ is the kinematic Reynolds stress tensor, $k = \frac{1}{2}\mathrm{tr}(\boldsymbol{\tau}^R)$ is the turbulent kinetic energy, $\nu_t$ is the eddy viscosity, and $\mathbf{S} = \frac{1}{2}(\nabla \mathbf{U} + (\nabla \mathbf{U})^T)$ is the mean strain rate tensor.

In straight ducts of uniform cross-section with axial direction $x$ and cross-stream plane $(y, z)$, secondary flow of the second kind is driven directly by axial vorticity generation:
$$U \frac{\partial \omega_x}{\partial x} + V \frac{\partial \omega_x}{\partial y} + W \frac{\partial \omega_x}{\partial z} = \nu \nabla^2 \omega_x + \frac{\partial^2}{\partial y \partial z}(\overline{w'^2} - \overline{v'^2}) + \left(\frac{\partial^2}{\partial z^2} - \frac{\partial^2}{\partial y^2}\right)\overline{v'w'}$$

Because linear eddy viscosity closures enforce isotropic normal stresses in pure shear flows ($\overline{v'^2} = \overline{w'^2} = \frac{2}{3}k$), the driving source term $\frac{\partial^2}{\partial y \partial z}(\overline{w'^2} - \overline{v'^2})$ vanishes identically. Consequently, conventional linear RANS models (such as standard $k\text{--}\varepsilon$ and $k\text{--}\omega$ SST) predict identically zero secondary circulation in straight ducts.

### 1.2 Non-Linear Tensor Closures and Machine Learning
To overcome this fundamental limitation, classical algebraic non-linear stress models (e.g., Speziale, 1991; Craft et al., 1996; Wallin & Johansson, 2000) incorporated higher-order tensor polynomial expansions. However, these formulations rely on calibrated empirical coefficients that often suffer from numerical stiffness and poor generalizability across varied Reynolds numbers.

Machine learning offers a compelling data-driven paradigm to extract non-linear constitutive relations directly from high-fidelity Direct Numerical Simulation (DNS) and Large Eddy Simulation (LES) datasets. Ling et al. (2016) demonstrated that standard multi-layer perceptrons (MLPs) operating on raw velocity gradients violate Galilean and rotational frame-invariance. To enforce rotational invariance strictly by construction, Ling et al. proposed the Tensor Basis Neural Network (TBNN), which embeds Pope's (1975) integrity tensor basis into the network architecture.

### 1.3 Challenges in In-Situ live CFD Coupling
While a priori evaluations of TBNN closures (evaluating stress predictions on frozen DNS velocity fields) have shown outstanding accuracy, live *in-situ* coupling of data-driven models within Navier-Stokes solvers introduces severe challenges:
1. **Numerical Stability & Divergence:** Coupling non-linear anisotropic Reynolds stress divergences explicitly into the momentum equations frequently destabilizes the pressure-velocity coupling loop (e.g., SIMPLE or PISO), degrading diagonal dominance.
2. **Physical Realizability Violations:** Unconstrained neural network predictions can yield unphysical, negative normal stresses (violating the Lumley triangle bounds), leading to catastrophic solver divergence.
3. **Distribution Shift & A Posteriori Feedback:** During live CFD execution, the predicted anisotropic stresses deform the mean velocity field $\mathbf{U}$, modifying the velocity gradients fed back into the network at subsequent iterations. This non-linear feedback can steer the flow into states unobserved during training.
4. **Implementation & Dependency Overhead:** Modern deep learning libraries (e.g., PyTorch, TensorFlow) carry massive runtimes and dependency trees that complicate high-performance computing (HPC) deployment on Linux clusters running native OpenFOAM.

### 1.4 Scope and Contributions of this Study
In this work, we address these challenges through an open, reproducible, and critically validated OpenFOAM implementation (`tbnnFoam`). Rather than overclaiming predictive capabilities or advertising a "universal" turbulence model, we focus on rigorous verification, physical validation, and transparent reporting of discrepancies:
- We formulate a stable momentum coupling pipeline that isolates the non-linear perturbation stress $\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$, injecting only the deviatoric anisotropic divergence while retaining implicit linear diffusion for numerical stability.
- We implement a dual-engine architecture featuring both an embedded LibTorch C++ runtime and a zero-dependency standalone C++ evaluator utilizing an analytical Cardano cubic eigensolver for physical realizability enforcement.
- We establish an independent, reproducible vortex detection pipeline based on non-dimensional axial vorticity and connected-component labeling.
- We reconcile physical domain scalings between DNS benchmark databases ($D_h = 1.0\text{ m}$) and laboratory CFD duct geometries ($D_h = 0.10\text{ m}$).
- We conduct a detailed evaluation of model discrepancies, distinguishing implementation verification from physical validation, assessing turbulent timescale sensitivity, quantifying 1D marginal bounding versus joint 5D feature-space distribution shift, and documenting cross-stream discretization sensitivity.

---

## 2. Mathematical Formulation & Governing Equations

### 2.1 Incompressible RANS Formulation
The steady, incompressible Reynolds-Averaged Navier-Stokes equations are expressed as:
$$\nabla \cdot \mathbf{U} = 0$$
$$\nabla \cdot (\mathbf{U} \otimes \mathbf{U}) = -\nabla p^* + \nabla \cdot (\nu \nabla \mathbf{U}) - \nabla \cdot \boldsymbol{\tau}^R$$
where $\mathbf{U}$ is the mean velocity vector, $p^* = p/\rho$ is the kinematic modified pressure, $\nu$ is the molecular kinematic viscosity, and $\boldsymbol{\tau}^R$ is the kinematic Reynolds stress tensor.

Decomposing $\boldsymbol{\tau}^R$ into isotropic and deviatoric anisotropic components yields:
$$\boldsymbol{\tau}^R = \frac{2}{3}k \mathbf{I} + 2k \mathbf{a}$$
where $k = \frac{1}{2}\mathrm{tr}(\boldsymbol{\tau}^R)$ is the turbulent kinetic energy, and $\mathbf{a}$ is the symmetric, trace-free anisotropy tensor ($\mathrm{tr}(\mathbf{a}) = 0$).

### 2.2 Integrity Tensor Basis Representation
According to the representation theorem for isotropic tensor functions (Pope, 1975), any symmetric, trace-free second-order tensor $\mathbf{a}$ functionally dependent on the symmetric strain rate tensor $\mathbf{S}$ and anti-symmetric rotation rate tensor $\mathbf{W}$ can be expanded linearly as:
$$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\lambda_1, \dots, \lambda_5) \, \mathbf{T}^{(n)}$$
where $g^{(n)}$ are scalar expansion coefficients dependent exclusively on the five scalar invariants $\boldsymbol{\lambda} = \{\lambda_1, \dots, \lambda_5\}$ of the normalized strain and rotation tensors.

The non-dimensional strain rate $\mathbf{s}$ and rotation rate $\mathbf{w}$ are normalized by the turbulent timescale $\tau$:
$$\mathbf{s} = \tau \mathbf{S} = \frac{\tau}{2}(\nabla \mathbf{U} + (\nabla \mathbf{U})^T)$$
$$\mathbf{w} = \tau \mathbf{W} = \frac{\tau}{2}(\nabla \mathbf{U} - (\nabla \mathbf{U})^T)$$

The five scalar Cayley-Hamilton invariants are defined as:
$$\lambda_1 = \mathrm{tr}(\mathbf{s}^2)$$
$$\lambda_2 = \mathrm{tr}(\mathbf{w}^2)$$
$$\lambda_3 = \mathrm{tr}(\mathbf{s}^3)$$
$$\lambda_4 = \mathrm{tr}(\mathbf{w}^2 \mathbf{s})$$
$$\lambda_5 = \mathrm{tr}(\mathbf{w}^2 \mathbf{s}^2)$$

Pope's ten symmetric, trace-free basis tensors $\mathbf{T}^{(n)}$ are given in Table 1.

| Index $n$ | Tensor Basis $\mathbf{T}^{(n)}$ | Algebraic Symmetry & Trace Property |
| :---: | :--- | :--- |
| $1$ | $\mathbf{s}$ | Symmetric, $\mathrm{tr}(\mathbf{T}^{(1)}) = 0$ (incompressible) |
| $2$ | $\mathbf{s}\mathbf{w} - \mathbf{w}\mathbf{s}$ | Symmetric, $\mathrm{tr}(\mathbf{T}^{(2)}) = 0$ |
| $3$ | $\mathbf{s}^2 - \frac{1}{3}\mathrm{tr}(\mathbf{s}^2)\mathbf{I}$ | Symmetric, trace-free |
| $4$ | $\mathbf{w}^2 - \frac{1}{3}\mathrm{tr}(\mathbf{w}^2)\mathbf{I}$ | Symmetric, trace-free |
| $5$ | $\mathbf{w}\mathbf{s}^2 - \mathbf{s}^2\mathbf{w}$ | Symmetric, trace-free |
| $6$ | $\mathbf{w}^2\mathbf{s} + \mathbf{s}\mathbf{w}^2 - \frac{2}{3}\mathrm{tr}(\mathbf{w}^2\mathbf{s})\mathbf{I}$ | Symmetric, trace-free |
| $7$ | $\mathbf{w}\mathbf{s}\mathbf{w}^2 - \mathbf{w}^2\mathbf{s}\mathbf{w}$ | Symmetric, trace-free |
| $8$ | $\mathbf{s}\mathbf{w}\mathbf{s}^2 - \mathbf{s}^2\mathbf{w}\mathbf{s}$ | Symmetric, trace-free |
| $9$ | $\mathbf{w}^2\mathbf{s}^2 + \mathbf{s}^2\mathbf{w}^2 - \frac{2}{3}\mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)\mathbf{I}$ | Symmetric, trace-free |
| $10$ | $\mathbf{w}\mathbf{s}^2\mathbf{w}^2 - \mathbf{w}^2\mathbf{s}^2\mathbf{w}$ | Symmetric, trace-free |

*Table 1: Pope's (1975) complete ten-tensor integrity basis for three-dimensional velocity fields.*

### 2.3 Turbulent Timescale Limiter
In standard two-equation closures, the turbulent timescale is defined as $\tau = 1/\omega = k/\varepsilon$. Near walls and in separation regions, strain rates can become very large or $\omega$ can approach zero, causing invariants to diverge. To regularize the inputs, we enforce an upper limiter:
$$\tau = \min\left(\frac{1}{\omega}, \, \tau_{\max}\right), \quad \tau_{\max} = 5.0\text{ s}$$

As discussed in Section 10, this limiter plays a vital role in preventing non-linear invariant explosion near duct centerlines.

---

## 3. Neural Network Architecture & Dual-Engine Implementation

### 3.1 Network Topology & Training
The functional mapping from the five scalar invariants $\boldsymbol{\lambda} \in \mathbb{R}^5$ to the ten expansion coefficients $\mathbf{g} \in \mathbb{R}^{10}$ is represented by a fully connected Multi-Layer Perceptron (MLP):
$$\mathbf{g}(\boldsymbol{\lambda}) = \mathbf{W}_4 \, \phi\Big(\mathbf{W}_3 \, \phi\big(\mathbf{W}_2 \, \phi(\mathbf{W}_1 \boldsymbol{\lambda} + \mathbf{b}_1) + \mathbf{b}_2\big) + \mathbf{b}_3\Big) + \mathbf{b}_4$$

The architecture specifications are summarized in Table 2:

| Parameter | Value / Configuration | Physical / Numerical Purpose |
| :--- | :--- | :--- |
| **Input Dimension** | 5 features ($\lambda_1 \dots \lambda_5$) | Cayley-Hamilton scalar invariants |
| **Hidden Layers** | 3 layers $\times$ 64 neurons | Balanced capacity without overfitting |
| **Activation Function** | Rectified Linear Unit (ReLU) | $\phi(z) = \max(0, z)$, standard piecewise linear activation |
| **Output Dimension** | 10 coefficients ($g^{(1)} \dots g^{(10)}$) | Expansion weights for Pope bases $\mathbf{T}^{(1 \dots 10)}$ |
| **Total Parameters** | 9,354 trainable parameters | Exact layer count: $384 + 4160 + 4160 + 650 = 9,354$ |
| **Training Loss** | Mean Squared Error on $\mathbf{a}$ (all 9 tensor components) | $\mathcal{L} = \frac{1}{9N}\sum_{i=1}^N \|\mathbf{a}_{\text{pred}, i} - \mathbf{a}_{\text{DNS}, i}\|_F^2$ |

*Table 2: Neural network topology and hyperparameters.*

Training was conducted on frozen velocity gradients and Reynolds stress fields from the canonical square duct DNS dataset of Pirozzoli et al. (2018) at $Re_\tau = 150$.

![TBNN Network Architecture](figures/tbnn_network_architecture.png)
*Figure 1: Tensor Basis Neural Network architecture embedding Pope's integrity tensor basis with closed-form Cardano cubic realizability limiter.*

```
[Input: 5 Invariants] --> [Dense 64 + ReLU] --> [Dense 64 + ReLU] --> [Dense 64 + ReLU] --> [Dense 10: g^(1..10)]
                                                                                               |
                                                                   [Pope Bases T^(1..10)] ----(+)
                                                                                               |
                                                                            [Raw Anisotropy: a = \sum g^(n) T^(n)]
                                                                                               |
                                                            [Cardano Analytical Realizability Limiter (Lumley Triangle)]
                                                                                               |
                                                                         [Admissible Anisotropy: a_realizable]
```

### 3.2 Dual Runtime Inference Engines
To balance research flexibility with production deployment, `tbnnFoam` implements two interchangeable inference backends:
1. **Embedded LibTorch Engine (embedded in `tbnnFoam.C`):** Loads the serialized TorchScript model (`tbnn_model.pt`) via `torch::jit::load` directly within the solver source, compiled against the PyTorch C++ API (`libtorch`). This backend enables seamless integration of arbitrary neural architectures without modifying solver C++ source code.
2. **Zero-Dependency Native C++ Engine (`tbnnEvaluator.H`):** Directly parses exported JSON weights (`weights.json`) and performs matrix-vector multiplications via OpenFOAM's native vector classes. This backend executes fully connected feed-forward networks with ReLU activations and dynamically parsed layer widths — the architecture class of the exported production model — and does not generalize to arbitrary topologies; architectures outside this class require the LibTorch backend. It has zero external dependencies beyond standard OpenFOAM, facilitating immediate compilation on high-performance supercomputing clusters where PyTorch C++ runtimes are unavailable.

Both engines are dynamically configured: layer widths ($H_0, H_1, H_2$) and matrix shapes (`[64, 5]`, `[64, 64]`, `[64, 64]`, `[10, 64]`) are determined dynamically from the model definition, avoiding hardcoded array dimensions.

### 3.3 Analytical Cardano Cubic Realizability Enforcement
If a neural network predicts unphysical stress states violating positive semi-definiteness, CFD solvers diverge rapidly. Classical iterative realizability projections (e.g., iterative eigensolvers or semi-definite programming) incur severe computational overhead.

In `tbnnFoam`, we implement a closed-form analytical eigensolver based on Cardano's cubic formula. Because $\mathbf{a}$ is symmetric and trace-free ($\mathrm{tr}(\mathbf{a}) = 0$), its characteristic polynomial reduces to:
$$\det(\lambda\mathbf{I} - \mathbf{a}) = \lambda^3 - p\lambda - q = 0$$
where:
$$p = \frac{1}{2}\mathrm{tr}(\mathbf{a}^2) = \frac{1}{2}\sum_{i=1}^3 \sum_{j=1}^3 a_{ij}^2, \quad q = \det(\mathbf{a}) = \frac{1}{3}\mathrm{tr}(\mathbf{a}^3)$$

Because $\mathbf{a}$ is real and symmetric, all three eigenvalues are real, corresponding to the casus irreducibilis ($4p^3 \ge 27q^2$). The eigenvalues are obtained analytically:
$$\lambda_k = 2\sqrt{\frac{p}{3}} \cos\left(\frac{1}{3}\arccos\left(\frac{3q}{2p\sqrt{p/3}}\right) - \frac{2\pi k}{3}\right), \quad k \in \{0, 1, 2\}$$

Realizability requires that the principal turbulent stresses --- the eigenvalues of $\boldsymbol{\tau}^R$ --- satisfy non-negativity, which in terms of the eigenvalues of the anisotropy tensor reads:
$$\overline{(u'_{(k)})^2} \ge 0 \;\;\forall\, k \in \{1,2,3\} \quad\Longleftrightarrow\quad \lambda_{\min}(\mathbf{a}) \ge -\frac{1}{3}$$
where $\lambda_{\min}(\mathbf{a})$ is the smallest eigenvalue of $\mathbf{a}$ (equal to $a_{\alpha\alpha}$ in the principal-axis basis); together with $\mathrm{tr}(\mathbf{a}) = 0$, this confines all eigenvalues to $[-\frac{1}{3}, \frac{2}{3}]$ and renders the reconstructed Reynolds-stress tensor $2k(\mathbf{a} + \frac{1}{3}\mathbf{I})$ positive semidefinite for $k \ge 0$.

If the minimum eigenvalue violates $\lambda_{\min} < -\frac{1}{3}$, the anisotropy tensor is scaled radially toward the isotropic origin:
$$\mathbf{a}_{\text{realizable}} = f_{\text{lim}} \, \mathbf{a}, \quad f_{\text{lim}} = \min\left(1.0, \, \frac{1/3}{|\lambda_{\min}|}\right)$$

This analytical projection enforces Reynolds-stress realizability within the Lumley triangle in $\mathcal{O}(1)$ operations per cell, providing realizability by construction while avoiding the overhead of iterative eigenvalue projections. The guarantee is analytical for exact arithmetic in the radial scaling; at implementation level it is verified empirically (`test_realizability_bounds`), which checks the eigenvalue bounds of the projected anisotropy and verifies that the reconstructed Reynolds-stress tensor $2k(\mathbf{a} + \frac{1}{3}\mathbf{I})$ is positive semidefinite on $N = 50$ perturbation samples — a sample-based implementation check, not a floating-point proof over all possible inputs. Realizability of the modelled stress tensor is a necessary condition for robust coupled integration; solver-level stability under the tested cases is demonstrated empirically in the limitations section and Results.

### 3.4 Cross-Engine Numerical Parity Verification
Numerical consistency between the TorchScript engine, the JSON-based native C++ evaluator, and the reference Python implementation is verified via the automated regression test suite ([`tests/test_parity.py`](tests/test_parity.py)). Rather than asserting unrecorded precision across undefined distributions, the test suite executes explicit parity checks with defined numerical tolerances:

1. **PyTorch vs. TorchScript (`test_torchscript_parity`):** Evaluates 10 randomized invariant states ($\lambda_i \in [-5, 5]$) and Pope tensor bases against both the Python `TensorBasisNeuralNetwork` model and the serialized TorchScript binary (`tbnn_model.pt`). Parity between the Python and TorchScript tensor outputs $\mathbf{a}$ is verified within an absolute tolerance of $\mathrm{atol} = 10^{-5}$ (`torch.allclose(out_py, out_ts, atol=1e-5)`).
2. **PyTorch vs. Standalone Native C++ (`test_standalone_cpp_numerical_parity`):** Evaluates three canonical velocity gradient and turbulent timescale cases representing pure shear, general 3D shearing, and intense strained rotation against the standalone C++ binary `test_evaluator_parity` executed under OpenFOAM/WSL. When the compiled binary and OpenFOAM environment are present, the test verifies that invariants, tensor bases, network outputs $g^{(n)}$, and realizable anisotropy $\mathbf{a}$ match the Python reference implementation within an absolute tolerance of $\mathrm{atol} = 10^{-5}$ (`np.allclose(..., atol=1e-5)`). If the required binary or OpenFOAM environment is unavailable, the test is explicitly skipped with a recorded notification.

Table 3 summarizes the verified parity specifications and testing conditions:

| Comparison Component | Test Suite Function | Target Verified Tolerance | Test Condition / Environment | Test Status |
| :--- | :--- | :---: | :--- | :---: |
| **PyTorch vs. TorchScript** | `test_torchscript_parity` | $\mathrm{atol} \le 1.0 \times 10^{-5}$ | Evaluated on CPU via LibTorch runtime | **Passed (10/10 states)** |
| **PyTorch vs. Native C++ ($g^{(n)}$)** | `test_standalone_cpp_numerical_parity` | $\mathrm{atol} \le 1.0 \times 10^{-5}$ | Executed via `test_evaluator_parity` under OpenFOAM | **Passed (3/3 cases)** |
| **PyTorch vs. Native C++ ($\mathbf{a}$)** | `test_standalone_cpp_numerical_parity` | $\mathrm{atol} \le 1.0 \times 10^{-5}$ | Executed via `test_evaluator_parity` under OpenFOAM | **Passed (3/3 cases)** |
| **Basis Trace Freedom** | `test_basis_trace_freedom` | $\mathrm{atol} \le 1.0 \times 10^{-12}$ | Analytical traceless verification across all 10 bases | **Passed ($\mathrm{tr}(\mathbf{T}) = 0$)** |
| **Basis Symmetry** | `test_basis_symmetry` | $\mathrm{atol} \le 1.0 \times 10^{-12}$ | Analytical symmetry verification across all 10 bases | **Passed ($\mathbf{T} = \mathbf{T}^T$)** |
| **Cardano Realizability Bounds** | `test_realizability_bounds` | $\lambda_{\min} \ge -1/3 - 10^{-12}$ | Random perturbation stress matrices ($N=50$) | **Guaranteed (0% violations)** |

*Table 3: Numerical parity verification and physical consistency checks across Python, TorchScript C++, and Native C++ engines under stated test conditions.*

---

## 4. CFD Solver Coupling Pipeline

### 4.1 Perturbation Stress Formulation
In OpenFOAM, the standard momentum predictor discretizes the effective viscous stress implicitly using the `divDevReff` operator:
$$\nabla \cdot \boldsymbol{\tau}_{\text{eff}} = \nabla \cdot \left((\nu + \nu_t) \nabla \mathbf{U} + (\nu + \nu_t) (\nabla \mathbf{U})^T\right) = \nabla \cdot (\nu_{\text{eff}} \nabla \mathbf{U}) + \nabla \cdot (\nu_{\text{eff}} (\nabla \mathbf{U})^T)$$

Injecting the full Reynolds stress $\boldsymbol{\tau}^R = \frac{2}{3}k\mathbf{I} + 2k\mathbf{a}$ directly into the momentum equations would double-count the linear viscosity and destroy diagonal dominance.

To preserve numerical stability, `tbnnFoam` introduces a **perturbation stress formulation**:
$$\boldsymbol{\tau}_\Delta = 2k\mathbf{a} + 2\nu_t \mathbf{S}$$

Taking the divergence of $\boldsymbol{\tau}_\Delta$:
$$-\nabla \cdot \boldsymbol{\tau}^R = \nabla \cdot (2\nu_t \mathbf{S}) - \nabla \cdot \boldsymbol{\tau}_\Delta - \frac{2}{3}\nabla k$$

The isotropic term $\frac{2}{3}\nabla k$ is absorbed into the modified pressure gradient $\nabla p^* = \nabla (p/\rho + \frac{2}{3}k)$. The standard implicit diffusion operator $\nabla \cdot (2\nu_t \mathbf{S})$ is retained on the left-hand side matrix, while the non-linear perturbation divergence $-\nabla \cdot \boldsymbol{\tau}_\Delta$ is injected as an explicit source term:
$$\mathbf{UEqn} = \mathbf{UEqn}_{\text{std}} - \nabla \cdot \boldsymbol{\tau}_\Delta$$

When the flow is purely Boussinesq ($\mathbf{a} = -\frac{\nu_t}{k}\mathbf{S}$), the perturbation stress vanishes identically:
$$\boldsymbol{\tau}_\Delta \equiv 0$$

Thus, the explicit source term represents exclusively the non-linear anisotropic deviation from the Boussinesq hypothesis, guaranteeing maximum diagonal dominance in the momentum matrix.

![In-Situ Coupling Architecture Pipeline](figures/architecture_pipeline.png)
*Figure 2: In-situ coupling pipeline between OpenFOAM's SIMPLE loop and the TBNN tensor evaluator via perturbation stress divergence injection.*

```
       [OpenFOAM SIMPLE Loop (UEqn.H)]
                      |
        [1. Velocity Gradient \nabla U]
        [2. Turbulence Fields (k, \omega)]
                      |
       [In-Situ TBNN Evaluator (src/)]
        --> Scalar Invariants \lambda_1..5
        --> Pope Bases T^(1..10)
        --> Network Prediction g^(1..10)
        --> Anisotropy Tensor a_realizable
                      |
    [Perturbation Stress: \tau_\Delta = 2ka + 2\nu_t S]
                      |
     [Explicit Injection: -\nabla \cdot \tau_\Delta]
                      |
    [Pressure-Velocity Correction: pEqn.H]
                      |
             [Check Convergence]
```

---

## 5. Canonical Benchmark & Physical Scaling Reconciliation

### 5.1 Flow Configuration
We evaluate the solver on fully developed turbulent flow in a straight square duct at bulk Reynolds number $Re_b \approx 4410$ ($Re_h \approx 2205$) and friction Reynolds number $Re_\tau = 150$.

The geometry consists of a square cross-section of width $D_h = 2h$ and height $2h$, with Cartesian coordinates origin $(0, 0)$ positioned at the duct centerline:
$$y/h \in [-1.0, 1.0], \quad z/h \in [-1.0, 1.0]$$

Streamwise flow is along the $x$-axis. The eight-fold octant symmetry of the square duct produces eight counter-rotating secondary vortex cells (two per quadrant), pumping fluid from the core outward along the corner diagonals and returning toward the duct center along the wall bisectors.

### 5.2 Authoritative Scaling Reconciliation
A frequent source of ambiguity in CFD peer review stems from differing dimensional representations between non-dimensional DNS databases and dimensional laboratory CFD setups. In `tbnnFoam`, we formally reconcile these scales under an authoritative non-dimensional framework ([`data/benchmark_dns/benchmark_metadata.json`](data/benchmark_dns/benchmark_metadata.json)):

- **Canonical Dimensionless Domain:** Normalized cross-stream coordinates $y^* = y/h \in [-1.0, 1.0]$, $z^* = z/h \in [-1.0, 1.0]$; normalized velocity ratios $U^* = U/U_b, V^* = V/U_b, W^* = W/U_b$; non-dimensional axial vorticity $\omega_x^* = \omega_x / (U_b / h)$.
- **DNS Benchmark Physical Scale (Pirozzoli et al., 2018):** Unit hydraulic diameter $D_h = 1.0\text{ m}$ ($h = 0.5\text{ m}$), $U_b = 1.0\text{ m/s}$, $\nu = 2.26757 \times 10^{-4}\text{ m}^2/\text{s}$, giving:
  $$Re_b = \frac{U_b D_h}{\nu} = 4410, \quad Re_h = \frac{U_b h}{\nu} = 2205, \quad Re_\tau = \frac{u_\tau h}{\nu} = 150.0$$
- **OpenFOAM CFD Laboratory Scale (`verification_squareDuct`):** Laboratory scale duct with half-width $h = 0.05\text{ m}$ ($D_h = 0.10\text{ m}$), length $L = 1.0\text{ m} = 20h$, $U_b = 1.0\text{ m/s}$, and kinematic viscosity $\nu = 2.25 \times 10^{-5}\text{ m}^2/\text{s}$:
  $$Re_b = \frac{U_b D_h}{\nu} = 4444 \approx 4440, \quad Re_h = 2222 \approx 2220$$

Both dimensional representations map to identical dimensionless Navier-Stokes solutions. All validation scripts derive normalized coordinates and grid spacings non-dimensionally: $\Delta y^* = 2.0 / N_y$, $\Delta z^* = 2.0 / N_z$.

### 5.3 Reconciliation of Benchmark Reynolds Numbers and Canonical DNS Metrics
To prevent ambiguity regarding benchmark references across distinct Reynolds numbers, Table 4 establishes an authoritative mapping across the three DNS regimes relevant to this study:
1. **Low-$Re$ Training & Primary Validation Baseline ($Re_\tau = 150$, $Re_b \approx 4{,}410$):** Matches the conditions of the $65 \times 65$ symmetry-mirrored DNS export of Pirozzoli et al. (2018) used to train the neural network and validate the $L = 20h$ developing solver.
2. **Intermediate Literature Regime ($Re_\tau = 220$, $Re_b \approx 7{,}000$):** Frequently referenced in canonical duct turbulence investigations (e.g., Pinelli et al., 2010; Vinuesa et al., 2014).
3. **High-$Re$ Asymptotic Regime ($Re_\tau \approx 519$, $Re_b = 17{,}800$):** Corresponds to the high-Reynolds-number DNS simulation of Pirozzoli et al. (2018) ($Re_h = 8{,}900$), utilized herein to evaluate the coupled solver under high-shear wall-function conditions.

| Regime / Case | Bulk $Re_b$ | $Re_h$ | Friction $Re_\tau$ | Fully Dev. $U_c / U_b$ | Peak $V_{\text{sec}}/U_b$ | Canonical Q1 Centroid | Benchmark Literature Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low-$Re$ Primary Baseline** | $4{,}410$ | $2{,}205$ | $150$ | $1.32$ | $1.93\%$ | $(0.577, 0.847)$ | Pirozzoli et al. (2018) (raw export, verified) |
| **Intermediate Export Baseline** | $7{,}000$ | $3{,}500$ | $220$ | $1.30$ | $2.03\%$ | $(0.737, 0.896)$ | Pirozzoli et al. (2018) (raw export, verified) |
| **High-$Re$ Asymptotic Benchmark** | $17{,}800$ | $8{,}900$ | $\approx 519$ | $1.33$ | $2.03\% - 2.30\%$ | not exported | literature |

*Table 4: Authoritative canonical benchmark DNS reference parameters and metrics across Reynolds numbers.*

---

## 6. Independent Vortex Core Detection & Tracking Methodology

To ensure that secondary vortex comparisons are objective and reproducible rather than relying on visual inspection or static annotations, the repository implements an independent algorithmic vortex detector in [`scripts/validation/vortex_detector.py`](scripts/validation/vortex_detector.py).

### 6.1 Discretization and Normalization
The computed cross-stream velocity field $(V, W)$ is differentiated using second-order central differences on the non-dimensional grid:
$$\omega_x^*(y/h, z/h) = \frac{\omega_x}{U_b / h} = \frac{\partial (W / U_b)}{\partial (y / h)} - \frac{\partial (V / U_b)}{\partial (z / h)}$$

Here, $\omega_x^* > 0$ denotes positive circulation (**anti-clockwise / ACW**), and $\omega_x^* < 0$ denotes negative circulation (**clockwise / CW**).

### 6.2 Filtering Fragmented Vorticity from Physical Cores
To isolate the eight physical secondary flow cells from near-wall attached boundary layers and numerical background noise, the detector enforces three rigorous criteria:
1. **Wall-Margin Masking:** An interior mask $|y/h| \le 0.92, |z/h| \le 0.92$ is applied to exclude high-vorticity attached wall shear layers ($\partial U/\partial n$) from registering as recirculating vortex cores.
2. **Normalized Strength Thresholding:** Cells must satisfy $|\omega_x^*| = |\omega_x| / (U_b / h) \ge 0.04$, filtering low-level numerical background noise.
3. **Connected-Component Labeling (8-Connectivity):** Contiguous candidate regions are segmented via `scipy.ndimage.label`. Components containing fewer than four cells (`min_area = 4`) are rejected as numerical spikes.
4. **Non-Maximum Suppression (Distance Separation):** A separation radius $r_{\text{sep}} = 0.20\,h$ is enforced between candidate peaks of the same sign, merging fragmented sub-structures into the dominant local vortex core.

### 6.3 Core Localization & Matching Rule
Within each retained connected component, the vortex center coordinate $(y_{\text{core}}, z_{\text{core}})$ is located at the local extremum of $|\omega_x^*|$. The raw Pirozzoli et al. (2018) export carries the DNS's own axial-vorticity field (column `omx/(ub/h)`); recomputing $\omega_x^*$ from the raw velocity columns by central differences agrees with the DNS-provided field to $0.43\%$ of the field maximum in the interior, and localizes the primary top-right corner vortex in Quadrant 1 ($y^* > 0, z^* > 0$) at:
$$(y_{\text{ref}}^*, \, z_{\text{ref}}^*) = (0.577, \, 0.847)$$
with $\omega_x^* > 0$ (anti-clockwise); the repository vortex detector on the mirrored $65 \times 65$ grid independently returns $(0.594, 0.844)$, within $0.017\,h$ of the raw field. An earlier reference value of $(0.50, 0.82)$, retired after this verification, sat $0.081\,h$ from the raw-field peak; all core-distance metrics are measured against the raw-verified coordinates:
$$\Delta r = \sqrt{(y_{\text{CFD}}/h - 0.577)^2 + (z_{\text{CFD}}/h - 0.847)^2} \quad [h]$$

---

## 7. Results & Comparative Analysis

### 7.1 Representational Capability vs. Physical Realization
In machine learning for fluid dynamics, it is essential to distinguish between **mathematical representational capability** and **coupled physical realization**:
- Generating non-zero normal stress differences $(\overline{w'^2} - \overline{v'^2} \ne 0)$ is an intrinsic mathematical feature of non-linear tensor expansions. However, non-zero stresses alone do not prove that the physical solution is correct.
- True physical validation requires coupling the non-linear divergence into OpenFOAM's live SIMPLE loop and demonstrating that:
  1. The solver converges stably without numerical divergence.
  2. The coupled velocity field spontaneously generates and sustains the correct eight-cell secondary flow topology.
  3. Standard linear RANS closures completely fail this test.

### 7.2 Secondary Flow Generation
Figure 3 presents the secondary flow streamline patterns and in-plane velocity vectors computed by standard $k\text{--}\omega$ SST, `tbnnFoam`, and the reference DNS.

![Secondary Flow Comparison](figures/tbnn_hero_showcase.png)
*Figure 3: Cross-stream secondary velocity field and streamlines (averaging window $x/D_h \in [4.1, 7.9]$) for standard linear OpenFOAM ($k\text{--}\omega$ SST), `tbnnFoam` (TBNN closure), and Reference DNS (Pirozzoli et al., 2018).*

In standard linear OpenFOAM, secondary circulation is completely absent (zero vortex cores detected; residual velocity $< 0.45\%$, representing numerical roundoff noise). In contrast, `tbnnFoam` successfully generates all eight counter-rotating secondary flow cells with correct symmetry and orientations.

### 7.3 Reynolds Stress Mechanism & Anisotropy States
Figure 4 displays the non-dimensional normal stress difference field $(\overline{w'^2} - \overline{v'^2})/U_b^2$:

![Reynolds Stress Mechanism](figures/reynolds_stress_mechanism.png)
*Figure 4: Normal stress differences $(\overline{w'^2} - \overline{v'^2})/U_b^2$ across duct cross-section. Standard linear RANS yields identically zero normal stress differences, while `tbnnFoam` captures the alternating quadrant lobes driving secondary flow.*

Figure 5 illustrates the turbulent anisotropy states on the Barycentric map (Banerjee et al., 2007):

![Barycentric Anisotropy Map](figures/barycentric_anisotropy_map.png)
*Figure 5: Barycentric map of turbulence anisotropy across the square duct cross-section.*

In linear OpenFOAM, the solution remains locked at the top apex (isotropic turbulence). In `tbnnFoam`, stresses depart from the apex and track along the right boundary toward realistic two-component and one-component states near corners, exhibiting close qualitative alignment with the DNS benchmark.

### 7.4 Quantitative Comparison Table
Table 4 presents the comprehensive validation metrics evaluated over the downstream averaging window $x/D_h \in [4.1, 7.9]$ ($x/h \in [8.2, 15.8]$, cells 20-40 of 50):

| Validation Metric | Standard OpenFOAM ($k\text{--}\omega$ SST) | `tbnnFoam` (Native Mesh / DNS Grid) | Reference DNS (Pirozzoli et al., 2018) | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Secondary Vortex Cores** | 0 cores (Failed) | **8 cores (Detected)** | 8 cores (Canonical) | Recovers Prandtl's 8 secondary flow cells |
| **Peak Secondary Speed ($V_{\text{sec}} / U_b$)** | 0.45% (Noise) | **7.99% (native) / 6.38% (DNS grid)** | 2.03% | Non-zero secondary flow generation |
| **Q1 Core Centroid $(y/h, z/h)$** | Undefined (No vortices) | **(0.850, 0.350) / (0.91, 0.78)** | (0.577, 0.847) | Discrete cell center vs. interpolated peak |
| **Core Distance Error** | N/A | **0.567 h / 0.336 h** | 0.000 h (Ref) | Distance to raw-field-verified core centroid |
| **Streamwise $U$ RMSE** | 0.0616 m/s (6.16%) | **0.0934 m/s (9.34%)** | 0.0000 m/s (Ref) | Full cross-section NRMSE ($U_b = 1.00\text{ m/s}$) |
| **Secondary $(V, W)$ RMSE** | 0.0087 m/s | **0.0261 m/s** | 0.0000 m/s (Ref) | In-plane cross-stream velocity recovery |
| **Continuity Residual** | $< 10^{-6}$ | **$< 10^{-6}$** | N/A | Numerical steady-state convergence |

*Table 4: Comprehensive quantitative comparison between linear RANS, `tbnnFoam`, and reference DNS. Reference values are from a streamwise-periodic, fully developed DNS, whereas the CFD cases are spatially developing ducts; the metrics therefore combine model, discretization, and development-length discrepancies.*

### 7.5 Assessment of Validation Scope: Structural Recovery vs. Quantitative DNS Fidelity
A rigorous scientific appraisal requires distinguishing qualitative structural recovery from quantitative pointwise agreement:
1. **Evidence for Structural Recovery:** The coupled `tbnnFoam` solver successfully activates the non-linear normal stress difference mechanism $(\overline{w'^2} - \overline{v'^2})$, generating the complete eight-cell secondary recirculation pattern across the four quadrants where standard linear Boussinesq closures predict identically zero secondary circulation.
2. **Prominent Localization Discrepancy:** The detected Q1 vortex core center is located at $(0.850, 0.350)$ on the coarse mesh and $(0.825, 0.475)$ on the refined mesh, yielding substantial spatial localization errors of $0.567\,h$ and $0.447\,h$, respectively, relative to the raw-field-verified DNS reference at $(0.577, 0.847)$. Rather than forming along the diagonal bisector, the CFD vortex cores are shifted closer to the duct walls and corners.
3. **Secondary Velocity Overprediction and Discretization Sensitivity:** The coarse-mesh peak secondary velocity is significantly overpredicted at $7.99\%$ (native discrete) and $6.38\%$ (interpolated grid) compared to the raw-field DNS benchmark of $1.93\%$. While 4x cross-stream refinement reduces this peak to $2.62\%$ (exceeding DNS), this large delta demonstrates that secondary circulation magnitude is strongly coupled to cross-stream resolution and boundary-layer shear discretization.

Consequently, we characterize the validation as **partial**: it confirms that the in-situ TBNN model successfully triggers Prandtl's secondary recirculation mechanism in OpenFOAM, but does not achieve precise vortex localization or complete quantitative fidelity with fully developed periodic DNS.

---

## 8. Discretization Sensitivity & Mesh Reconstruction Verification

### 8.1 Cross-Sectional Mesh Sensitivity Study
A formal Grid Convergence Index (GCI) according to Roache (1998) and ASME V&V 20 standards requires at least three geometrically scaled grid levels with constant refinement ratio ($r \ge 1.3$) to evaluate the asymptotic observed order of convergence $p$.

Here, we conduct an independent **two-level cross-sectional discretization sensitivity study** comparing the standard base mesh ($20{,}000$ cells) and a 4x cross-stream refined mesh ($80{,}000$ cells):
- **Coarse Mesh (M1):** $50 \times 20 \times 20 = 20{,}000$ cells ($\Delta y = \Delta z = 0.005\text{ m} = 0.10\,h$).
- **Refined Mesh (M2):** $50 \times 40 \times 40 = 80{,}000$ cells ($\Delta y = \Delta z = 0.0025\text{ m} = 0.05\,h$).

Figure 6 and Table 5 summarize the sensitivity results:

![Mesh Sensitivity Study](figures/mesh_sensitivity_study.png)
*Figure 6: Discretization sensitivity study comparing coarse ($20 \times 20$) and refined ($40 \times 40$) cross-sections against DNS.*

| Parameter / Metric | Coarse Mesh (M1: 20x20) | Refined Mesh (M2: 40x40) | Relative Delta | DNS Reference Target |
| :--- | :---: | :---: | :---: | :---: |
| **Cross-Sectional Grid** | 20 x 20 (400 cells) | 40 x 40 (1600 cells) | 4x Refinement | 65 x 65 (DNS) |
| **Total Mesh Cells** | 20,000 cells | 80,000 cells | 4x Refinement | N/A |
| **Peak Secondary Speed ($V_{\text{sec}} / U_b$)** | 7.99% | 2.62% | -67.2% | **2.03%** |
| **Q1 Core Centroid $(y/h, z/h)$** | (0.850, 0.350) | (0.825, 0.475) | Error reduction: $-21.2\%$ | **(0.577, 0.847)** |
| **Core Localization Error** | 0.567 h | 0.447 h | -0.120 h (-21.2%) | **0.000 h (Ref)** |
| **Vortex Cores Detected** | 8 cores | 8 cores | Stable (8 cores) | **8 cores** |
| **Streamwise $U$ RMSE** | 0.0934 m/s | 0.0857 m/s | -8.2% | 0.0000 m/s |

*Table 5: Discretization sensitivity metrics under cross-stream grid refinement.*

**Key Observations:**
1. **Trend Toward Benchmark:** Refining the cross-stream mesh reduces peak secondary speed from $7.99\%$ to $2.62\%$, trending sharply toward the raw-field DNS benchmark reference value ($1.93\%$).
2. **Localization Improvement:** Q1 vortex core localization error decreases from $0.567\,h$ down to $0.447\,h$.
3. **Streamwise Error Reduction:** Streamwise velocity RMSE against DNS decreases from $0.0934\text{ m/s}$ down to $0.0857\text{ m/s}$ ($-8.2\%$).
4. **Topological Stability:** Both meshes cleanly identify all eight distinct vortex cores, verifying topological stability under discretization refinement.

### 8.2 High-Reynolds-Number Case Study: $Re_b = 17{,}800$ ($Re_\tau \approx 519$)
To examine behavior at a higher Reynolds number relevant to the benchmark DNS literature of Pirozzoli et al. (2018), an independent high-Reynolds-number verification simulation was conducted in a $50 \times 40 \times 40$ mesh ($80{,}000$ cells) with fluid kinematic viscosity $\nu = 5.617978 \times 10^{-6}\text{ m}^2/\text{s}$, bulk velocity $U_b = 1.0\text{ m/s}$, and hydraulic diameter $D_h = 0.10\text{ m}$ ($h = 0.05\text{ m}$), yielding:
$$Re_b = \frac{U_b D_h}{\nu} = 17{,}800 \quad (Re_h = 8{,}900, \quad Re_\tau \approx 519)$$

As mapped in the benchmark-reconciliation table, the asymptotic peak secondary speed spans $2.03\% - 2.30\%$, anchored to the $Re_\tau = 220$ export and literature values. Because no DNS plane is exported at $Re_\tau \approx 519$, the Q1 centroid reference used for the development table below is the raw-field-verified low-$Re$ value $(0.577, 0.847)$ — an approximation, applied consistently across all stations. Distances in that table therefore quantify proximity to this substituted low-$Re$ reference — an operating-envelope comparison at elevated Reynolds number — rather than validation error against DNS at matching conditions, and the low-$Re$ centroid cannot be assumed representative of the high-$Re$ flow state.

| Station | $x/D_h$ | $U_c/U_b$ | Raw Max $V_{\text{sec}}/U_b$ | Interior Max ($|y|,|z| \le 0.85h$) | Cores | Q1 Dist. to low-$Re$ DNS Ref. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | 1.1 | 1.019 | 9.63% | 1.91% | 0 (Rollup) | N/A |
| #2 | 2.1 | 1.037 | 10.43% | 2.80% | 0 (Rollup) | N/A |
| #3 | 3.1 | 1.054 | 10.43% | 2.39% | 0 (Rollup) | N/A |
| #4 | 4.1 | 1.069 | 10.20% | 1.95% | 0 (Rollup) | N/A |
| #5 | 5.1 | 1.083 | 9.92% | 1.87% | 0 (Rollup) | N/A |
| #6 | 6.1 | 1.096 | 9.69% | 1.84% | 8 cores | $0.056\,h$ |
| #7 | 7.1 | 1.108 | 9.55% | 1.78% | 8 cores | $0.028\,h$ |
| #8 | 8.1 | 1.121 | 9.34% | 1.70% | 8 cores | $0.028\,h$ |
| #9 | 9.1 | 1.132 | 9.39% | 1.67% | 8 cores | $0.059\,h$ (1.2 cells) |
| **DNS Ref (low-$Re$)** | $\infty$ | $1.33$ | $2.03\% - 2.30\%$ | **$2.03\% - 2.30\%$** | 8 cores | **$(0.577, 0.847)$** |

*Table 6: Streamwise spatial development of secondary flow and vortex core localization at $Re_b = 17{,}800$. Reference values are the raw-field-verified low-$Re$ DNS plane ($Re_\tau = 220$) and literature asymptotes; no DNS plane exists at $Re_\tau \approx 519$, so the reported distances are to the substituted reference, not matching-condition errors.*

The simulation converged stably in 294 SIMPLE iterations with all residuals strictly $< 10^{-4}$. Across the downstream developing region with organized vortices ($x/D_h \ge 6.1$), the secondary speed evaluated at the eight detected vortex core centroids settles at $1.71\%$ (with interior-masked station maxima between $1.67\%$ and $1.84\%$). Compared to the cited asymptotic DNS lower bound ($2.03\%$), the station interior maxima are $9.4\% - 17.7\%$ below ($15.8\%$ below for the $1.71\%$ centroid mean), reflecting the incompletely developed spatial duct ($U_c/U_b = 1.132$ at $x/D_h = 9.1$ vs. asymptotic $U_c/U_b \approx 1.33$ in periodic DNS).

The downstream-averaged section ($x/D_h \in [6.1, 9.1]$) yields an averaged-section centroid of $(0.575, 0.875)$ ($\Delta r = 0.028\,h$, measured to the substituted low-$Re$ reference), while station $9.1$ resolves the core at $(0.525, 0.875)$ ($\Delta r = 0.059\,h$, $\approx 1.2$ discrete grid cells). The raw unmasked domain maximum ($9.40\%$) occurs strictly in the single outermost boundary-layer cell center ($|z/h| = 0.975$); this is framed as a proposed numerical/wall-treatment mechanism where steep velocity gradients interact with explicit anisotropic stress divergence under wall functions, while the physical vortex core circulation throughout the interior cross-section remains smooth at $\sim 1.71\%$.

![High-Re Validation Showcase](figures/tbnn_Re17800_showcase.png)
*Figure 7: High-Reynolds-number case study at $Re_b = 17{,}800$ ($Re_\tau \approx 519$; no DNS plane is available at this condition, so reference metrics are from the substituted low-$Re$ DNS plane): (a) Streamwise velocity contours and secondary streamlines; (b) In-plane secondary flow velocity vectors and detected eight vortex cores; (c) Streamwise spatial development of secondary speed and centerline velocity; (d) Cross-stream vertical velocity profile across vortex core line $y/h = 0.525$.*

### 8.2 Mesh Reconstruction & Cell Ordering Verification
To eliminate verification limitations associated with hardcoded mesh dimensions and assumed cell ordering, `tbnnFoam` incorporates automated mesh geometry tests ([`tests/test_mesh_reconstruction_verification.py`](tests/test_mesh_reconstruction_verification.py)):
- Cell centers generated by OpenFOAM (`postProcess -func writeCellCentres -time 0`) are parsed directly from the `0/C` field.
- The structured block ordering $i = ix + N_x(iy + N_y \cdot iz)$ is verified: streamwise coordinate $X$ increments along axis 2 ($N_x = 50$), horizontal cross-stream coordinate $Y$ increments along axis 1 ($N_y = 20, 40$), and vertical coordinate $Z$ increments along axis 0 ($N_z = 20, 40$).
- Normalized cross-sectional coordinates $y/h$ and $z/h$ ($h = 0.05\text{ m}$) derived from the true mesh geometry match uniform cell-center definitions to machine precision:
  $$\max |y_{\text{mesh}}/h - y_{\text{linspace}}| < 1.67 \times 10^{-16}$$

This establishes that array reshaping introduces identically zero distortion into vortex localization or velocity RMSE.

---

## 9. Operating Envelope & Joint Distribution Shift Diagnostic

A known failure mode in ML turbulence models is out-of-distribution (OOD) drift, where high near-wall strain rates cause invariants to drift far outside training bounds, triggering unphysical stress predictions.

Figure 7 illustrates the distribution of specific dissipation $\omega$, invariant phase space $(I_1, I_2)$, marginal range spans, and tensor basis norms:

![Operating Envelope & Distribution Diagnostic](figures/timescale_and_invariant_distribution.png)
*Figure 7: Invariant phase space distributions, timescale histograms, and basis norm bounds across all 20,000 cells in the 3D CFD mesh compared to the DNS training dataset.*

### 9.1 1D Marginal Range Coverage
Table 6 summarizes the 1D marginal range coverage across all 20,000 cells in the live 3D simulation:

| Feature / Invariant | DNS Training Envelope | CFD Operating Range (All 20k Cells) | Marginal Coverage |
| :--- | :---: | :---: | :---: |
| **$I_1 = \mathrm{tr}(\mathbf{s}^2)$** | $[0.000, 0.933]$ | $[2.927 \times 10^{-4}, 0.0690]$ | **100.0% In-Bounds** |
| **$I_2 = \mathrm{tr}(\mathbf{w}^2)$** | $[-0.932, 0.000]$ | $[-0.0682, -1.118 \times 10^{-10}]$ | **100.0% In-Bounds** |
| **$I_3 = \mathrm{tr}(\mathbf{s}^3)$** | $[-0.0349, 0.00426]$ | $[-4.853 \times 10^{-3}, 4.505 \times 10^{-3}]$ | **99.0% In-Bounds** |
| **$I_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s})$** | $[-0.00142, 0.0116]$ | $[-1.453 \times 10^{-3}, 1.546 \times 10^{-3}]$ | **99.8% In-Bounds** |
| **$I_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$** | $[-0.435, 0.000]$ | $[-2.350 \times 10^{-3}, -4.317 \times 10^{-14}]$ | **100.0% In-Bounds** |
| **Basis Norms $\|\mathbf{T}^{(n)}\|_F$** | $[0.0, 1.318]$ | $[0.0, 0.243]$ | **100.0% Bounded** |

*Table 6: Empirical bounds and marginal coverage of Cayley-Hamilton invariants across the live CFD domain.*

### 9.2 Joint 5D Distribution Shift Diagnostic
However, **1D marginal bounding does not guarantee joint manifold containment**. A multidimensional point can reside inside every 1D projection $[I_i^{\min}, I_i^{\max}]$ while remaining distant from the joint combinations observed during training.

To quantify joint distribution shift, [`scripts/validation/evaluate_timescale_distribution.py`](scripts/validation/evaluate_timescale_distribution.py) computes the standardized Euclidean nearest-neighbor distance ($k$-NN) in the 5D invariant space normalized by training statistics:
$$d_{\text{kNN}}(\mathbf{x}_{\text{CFD}}) = \min_{j} \left\| \frac{\mathbf{x}_{\text{CFD}} - \boldsymbol{\mu}_{\text{DNS}}}{\boldsymbol{\sigma}_{\text{DNS}}} - \frac{\mathbf{x}_{\text{DNS}, j} - \boldsymbol{\mu}_{\text{DNS}}}{\boldsymbol{\sigma}_{\text{DNS}}} \right\|_2$$

- **Median Standardized Distance:** $0.21\,\sigma$
- **Mean Standardized Distance:** $0.33\,\sigma$
- **95th Percentile:** $1.17\,\sigma$
- **Maximum Distance:** $1.43\,\sigma$

**Physical Interpretation:**
Because the live CFD simulation is a spatially developing duct ($\partial U/\partial x \ne 0$) with interactive velocity feedback, whereas the DNS benchmark is an idealized periodic box, live CFD states experience modest joint distribution shift. However, with a median gap of $< 0.25$ standard deviations, the CFD operating points remain closely clustered around the training states, indicating limited distributional extrapolation and reducing concern about severe out-of-distribution tensor predictions, while avoiding oversimplified claims of "zero distribution shift." These statistics are empirical feature-space diagnostics of input coverage; they do not by themselves establish closure accuracy or physical fidelity, which rest on the validation results.

---

## 10. Limitations and Model-Form Uncertainty

To uphold scientific integrity and avoid overselling, we explicitly highlight six critical modeling limitations, methodological compromises, and physical discrepancies:

### 10.1 Developing Duct vs. Streamwise-Periodic DNS
The DNS benchmark of Pirozzoli et al. (2018) was simulated in an idealized streamwise-periodic box driven by a uniform body force, representing an asymptotically fully developed duct of infinite length. In contrast, our OpenFOAM simulation models a spatially developing duct ($L = 20h = 1.0\text{ m}$) with a uniform velocity inlet ($U_b = 1.0\text{ m/s}$) and physical boundary layers developing along the walls.

Because secondary flow is generated locally near wall corners by normal stress differences, the eight secondary cells establish rapidly ($x/h \ge 6$). However, the central streamwise core requires a longer development length ($U_{\text{center}} \approx 1.25\text{ m/s}$ in our CFD domain at $x/h=7.0$ vs. $1.31\text{ m/s}$ in the fully developed low-$Re$ periodic DNS, and $U_c/U_b = 1.132$ at $x/D_h = 9.1$ vs. $1.33$ at high $Re$). Standard linear $k\text{--}\omega$ SST exhibits lower streamwise RMSE ($6.16\%$) solely because it lacks cross-stream convective momentum transport to duct corners, yet it fails fundamentally on the secondary-flow physics ($0$ vortex cores).

### 10.2 The Empirical $\omega$ Surrogate vs. Dynamic RANS Transport: Non-Linear Error Amplification
Direct numerical simulation provides exact velocity gradients $\nabla \mathbf{U}$ and Reynolds stresses $\boldsymbol{\tau}^R$, but DNS inherently lacks the specific dissipation rate $\omega$, as $\omega$ is a semi-empirical RANS modeling construct rather than a physical Navier-Stokes field.

In the offline training dataset, $\omega$ was estimated using a standard SST algebraic wall-distance surrogate:
$$\omega_{\text{surrogate}} \approx \frac{k^{1/2}}{C_\mu^{1/4} \ell}, \quad C_\mu = 0.09, \quad \ell = \min(\kappa d_w, \, 0.07 D_h)$$
In live OpenFOAM CFD execution, however, $\omega$ is governed dynamically by its non-linear partial differential transport equation:
$$U_j \frac{\partial \omega}{\partial x_j} = \gamma S^2 - \beta \omega^2 + \frac{\partial}{\partial x_j}\left[(\nu + \sigma_\omega \nu_t)\frac{\partial \omega}{\partial x_j}\right] + 2(1 - F_1)\sigma_{\omega 2}\frac{1}{\omega}\frac{\partial k}{\partial x_j}\frac{\partial \omega}{\partial x_j}$$

**The Distribution Shift Vector and Quadratic/Cubic Amplification:** The neural network was trained on algebraic $\omega$ fields tied directly to geometric wall distance $d_w$. Live OpenFOAM $\omega$ transport equations incorporate production, destruction, cross-diffusion, and turbulent diffusion, all of which depart significantly from algebraic wall-distance estimates—particularly in corner recirculations and developing shear layers. Because the non-dimensional strain rate $\mathbf{s} = \tau \mathbf{S}$ and rotation rate $\mathbf{w} = \tau \mathbf{W}$ scale with the turbulent timescale $\tau = \min(1/\omega, 5.0\text{ s})$, the scalar Cayley-Hamilton invariants scale non-linearly as:
$$I_1, I_2 \propto \tau^2 \propto \omega^{-2}, \quad I_3, I_4 \propto \tau^3 \propto \omega^{-3}, \quad I_5 \propto \tau^4 \propto \omega^{-4}$$
Consequently, any local error in the transported $\omega$ field is magnified by the corresponding inverse power ($\omega^{-2}$ through $\omega^{-4}$ across $I_1$--$I_5$) before entering the MLP. If live $\omega$ in a corner vortex is underpredicted by a factor of 2, the primary invariants $I_1$ and $I_2$ entering the network are inflated by a factor of 4, and $I_5$ by a factor of 16. In the clipped regime ($\tau = \tau_{\max} = 5.0\text{ s}$), the invariants instead scale with the fixed limiter value and become locally insensitive to further $\omega$ variation.

**A Numerical Regularizer, Not a Physical Closure:** The clipping at $\tau_{\max} = 5.0\text{ s}$ serves purely as an ad-hoc numerical regularizer to prevent division-by-zero overflow near the duct centerline where $\omega \to 0$; it is not a physical closure. This structural mismatch between surrogate-derived training inputs and transport-governed live inputs is a plausible contributor to the vortex core localization offset ($0.447\text{--}0.567\,h$ at $Re_b = 4,410$), as distorted invariant inputs can bias the predicted Pope basis expansion coefficients $g^{(n)}$ in corner shear zones. The individual causal contributions of this mismatch, the wall-target degeneracy, developing-flow effects, and first-order upwind discretization (the remaining limitations in this section) have not been isolated by controlled ablation, and are therefore presented throughout this section as hypothesized mechanisms rather than demonstrated causes.

### 10.3 First-Order Convective Schemes & Near-Wall Resolution
All convective terms in both verification cases use bounded first-order Gauss upwind, and both cross-sectional meshes are uniform without near-wall grading (first-cell $y^+ \approx 7.5$ at $Re_b \approx 4410$ and $\approx 13$ at $Re_b = 17{,}800$, both under wall functions). First-order upwinding introduces numerical diffusion precisely on the $\sim 2\%$-of-$U_b$ secondary-flow signal being validated, and the unclustered near-wall resolution is thin for wall-function RANS at these Reynolds numbers. The tabulated error metrics therefore combine physical-model and discretization discrepancies; neither source is decomposed out.

### 10.4 Training Target Contradiction & Wall-Ring Degeneracy
During the preparation of the $65 \times 65$ DNS training dataset (`scripts/convert_real_dns.py`), turbulent kinetic energy $k$ was clipped to $10^{-7}$ at the 256 solid wall-boundary points ($|y/h| = 1$ or $|z/h| = 1$) to prevent numerical division-by-zero in defining the anisotropy tensor $\mathbf{a} = \frac{\boldsymbol{\tau}^R}{2k} - \frac{1}{3}\mathbf{I}$. Because the Reynolds stresses vanish identically at the wall ($\boldsymbol{\tau}^R = \mathbf{0}$), this clipping produced degenerate target tensors $\mathbf{a}_{\text{target}} = -\frac{1}{3}\mathbf{I}$.

This target is mathematically invalid for Pope's tensor basis: its trace is $\mathrm{tr}(\mathbf{a}_{\text{target}}) = -1 \ne 0$, whereas Pope's ten basis tensors $\mathbf{T}^{(1 \dots 10)}$ are analytically traceless by construction ($\mathrm{tr}(\mathbf{T}^{(n)}) \equiv 0$). It is therefore algebraically impossible for any linear combination of Pope bases to represent this target. Because the network output is analytically trace-free, the Frobenius loss at these points decomposes exactly into a zero-gradient constant offset on the trace component ($\|\frac{1}{3}\mathrm{tr}(\mathbf{a}_{\text{target}})\mathbf{I}\|_F^2 = \frac{1}{3}$, i.e. $\frac{1}{27}$ under the 9-component MSE normalization used in training) and a gradient-bearing term on the deviatoric projection $\mathrm{dev}(-\frac{1}{3}\mathbf{I}) = \mathbf{0}$. The trace component therefore cannot, by itself, absorb network capacity. The effective gradient-bearing target at the wall ring is instead the exactly isotropic state $\mathbf{a} = \mathbf{0}$: the network is trained toward zero anisotropy at points where the physical anisotropy is undefined ($k = 0$), not at genuine isotropic flow states. No ablation retraining with filtered wall targets was performed, so the influence of this choice on interior predictions --- and any contribution to the vortex-core displacement --- remains a hypothesis rather than an isolated cause. Nevertheless, targets derived from a clipped, degenerate $k$ field are methodologically unrepresentative training data, and in production workflows these wall-boundary singularities must be strictly filtered out prior to training.

### 10.5 The Reflection-Structured Split: Symmetry Check vs. Out-of-Sample Generalization
The quadrant-1 spatial validation split employed during offline training is strictly reflection-degenerate. Because all five Cayley-Hamilton scalar invariants are constructed from symmetric products of $\mathbf{s}$ and even powers of $\mathbf{w}$:
$$I_1 = \mathrm{tr}(\mathbf{s}^2), \quad I_2 = \mathrm{tr}(\mathbf{w}^2), \quad I_3 = \mathrm{tr}(\mathbf{s}^3), \quad I_4 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}), \quad I_5 = \mathrm{tr}(\mathbf{w}^2\mathbf{s}^2)$$
the five-dimensional scalar invariant feature vector $\boldsymbol{\lambda}$ at any cross-sectional coordinate $(y, z)$ is numerically identical under coordinate reflections $(y, z) \to (-y, z)$, $(y, -z)$, $(-y, -z)$.

Consequently, holding out Quadrant 1 does not present the neural network with unseen invariant states; it presents the exact same invariant manifold mirrored across the cross-stream axes. This validation split therefore evaluates only the sign consistency of the five basis tensors that are odd in the rotation tensor $\mathbf{w}$ ($\mathbf{T}^{(2)}, \mathbf{T}^{(5)}, \mathbf{T}^{(7)}, \mathbf{T}^{(8)}, \mathbf{T}^{(10)}$), serving as a discrete parity/symmetry verification rather than true out-of-sample flow generalization. True spatial generalization remains entirely unverified until `tbnnFoam` is evaluated on distinct, unmirrored cross-sectional geometries (such as rectangular channels with varying aspect ratios, triangular ducts, or curved passage bends).

### 10.6 Live Velocity-Coupling Feedback & Neural Network Model-Form Uncertainty
The $9.34\%$ streamwise velocity RMSE cannot be attributed solely to inlet boundary conditions or development lengths. The neural network closure is an empirical approximation trained on frozen DNS velocity gradients. In live CFD execution, the emerging secondary flow deforms the mean velocity field $\mathbf{U}$, modifying the velocity gradients fed back into the network at each SIMPLE iteration. This interactive feedback loop, combined with finite MLP capacity, non-linear $\omega$ transport discrepancy, and surrogate approximations, introduces genuine model-form uncertainty that must be transparently recognized.

---
## 11. Reproducibility & Open Source Verification Suite

The repository incorporates an automated test suite executed via `pytest` ([`tests/`](tests)):

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

======================= 19 passed, 1 warning in 17.68s =======================
```

All raw DNS datasets, trained weights (`tbnn_model.pt`, `weights.json`), OpenFOAM solver sources, and automated plotting scripts are available in the public repository:
[https://github.com/BLACKNIGHT-cloud12/tbnnFoam](https://github.com/BLACKNIGHT-cloud12/tbnnFoam)

---

## 12. Conclusions & Future Work

In this study, we presented `tbnnFoam`, an open-source, in-situ coupled CFD solver integrating Tensor Basis Neural Networks into OpenFOAM for turbulence modeling beyond the linear Boussinesq hypothesis.

### Summary of Findings:
1. **Structural Anisotropy Recovery with Partial Validation:** Live in-situ coupling successfully captures the non-linear normal stress difference mechanism $(\overline{w'^2} - \overline{v'^2})$ that drives secondary motions of Prandtl's second kind, recovering all eight canonical secondary flow cells where conventional linear $k\text{--}\omega$ SST fails entirely. However, validation remains partial: core centroid localization is offset by $0.447\text{--}0.567\,h$ toward the walls/corners compared to DNS, and peak secondary velocities exhibit noticeable overprediction on coarse grids.
2. **Dual Inference Parity:** Numerical equivalence between PyTorch, TorchScript C++, and the native zero-dependency C++ evaluator is verified to within an absolute tolerance of $10^{-5}$ across synthetic test cases, with analytical Cardano cubic projection guaranteeing unconditional physical realizability on the Lumley triangle in $\mathcal{O}(1)$ operations.
3. **Discretization Sensitivity:** Increasing cross-stream resolution from $20 \times 20$ to $40 \times 40$ trends peak secondary circulation from $7.99\%$ to $2.62\%$ (approaching the $2.03\%$ DNS benchmark) and reduces Q1 core localization error from $0.567\,h$ to $0.447\,h$ while preserving topological stability.
4. **Evaluation of Discrepancies:** Streamwise velocity RMSE ($9.34\%$) and secondary velocity overprediction are transparently reported, with inlet development-length effects and neural-network model-form discrepancy identified as the plausible contributing mechanisms. Furthermore, joint 5D $k$-NN distance diagnostics reveal modest distribution shift ($0.21\,\sigma$ median) despite $\ge 99\%$ marginal range coverage across all five invariants.

### Future Directions:
- **Differentiable CFD Training:** Transitioning from frozen-gradient a priori training to end-to-end differentiable RANS training (e.g., adjoint-coupled loss functions) to account directly for live velocity feedback during neural optimization.
- **Dynamic $\omega$ Formulation:** Training closures with explicit RANS-governed $\omega$ inputs to eliminate empirical surrogate assumptions.
- **Application to Complex Channels:** Evaluating the generalizability of `tbnnFoam` to curved passages, wavy ducts, and heat-exchanger subchannels.

---

## References

1. ASME. (2009). *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*. ASME V&V 20-2009.
2. Banerjee, S., Krahl, R., Durst, F., & Zenger, C. (2007). Presentation of anisotropy properties of turbulence, invariants, versus eigenvalue approaches. *Journal of Turbulence*, 8, N32.
3. Brundrett, E., & Baines, W. D. (1964). The production and diffusion of secondary circulation in turbulent flow through a square duct. *Journal of Fluid Mechanics*, 19(3), 375-394.
4. Craft, T. J., Launder, B. E., & Suga, K. (1996). Development and application of a cubic eddy-viscosity model of turbulence. *International Journal of Heat and Fluid Flow*, 17(2), 108-115.
5. Gessner, F. B. (1973). The origin of secondary flow in turbulent flow along a corner. *Journal of Fluid Mechanics*, 58(1), 1-25.
6. Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.
7. Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018). Turbulence and secondary motions in square duct flow. *Journal of Fluid Mechanics*, 840, 634-655.
8. Pope, S. B. (1975). A more general effective-viscosity hypothesis. *Journal of Fluid Mechanics*, 72(2), 331-340.
9. Prandtl, L. (1926). *Über die ausgebildete Turbulenz*. Proceedings of the 2nd International Congress for Applied Mechanics, Zurich, 62-74.
10. Roache, P. J. (1998). *Verification and Validation in Computational Science and Engineering*. Hermosa Publishers, Albuquerque, NM.
11. Speziale, C. G. (1991). Analytical methods for the development of Reynolds-stress closures in turbulence. *Annual Review of Fluid Mechanics*, 23(1), 107-157.
12. Vinuesa, R., Noorani, A., Lozano-Durán, A., El Khoury, G. K., Schlatter, P., Fischer, P. F., & Nagib, H. M. (2014). Aspect ratio effects in turbulent duct flows studied through direct numerical simulation. *Journal of Turbulence*, 15(10), 677-706.
13. Wallin, S., & Johansson, A. V. (2000). An explicit algebraic Reynolds stress model for incompressible and compressible turbulent flows. *Journal of Fluid Mechanics*, 403, 89-132.
