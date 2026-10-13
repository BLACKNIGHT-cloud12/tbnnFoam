# tbnnFoam: OpenFOAM C++ In-Solver Tensor Basis Neural Network RANS Closure

`tbnnFoam` is a custom OpenFOAM steady-state RANS solver with in-situ **LibTorch** C++ integration. It evaluates Pope's (1975) 10-tensor basis integrity expansion to predict the non-linear Reynolds stress anisotropy tensor $\mathbf{a}$ at every cell:

$$\mathbf{a} = \sum_{n=1}^{10} g^{(n)}(\lambda_1, \dots, \lambda_5) \, \mathbf{T}^{(n)}$$

$$\boldsymbol{\tau}^R = 2 k \left(\mathbf{a} + \frac{1}{3}\mathbf{I}\right)$$

Numerical stability is preserved via **semi-implicit eddy-viscosity splitting**:

$$\nabla \cdot \boldsymbol{\tau}^R = \underbrace{- \nabla \cdot (\nu_{t,\text{base}} \nabla \mathbf{U})}_{\text{Implicit LHS: } \mathtt{fvm::laplacian}} + \underbrace{\nabla \cdot (\boldsymbol{\tau}_{\text{TBNN}}^R - \boldsymbol{\tau}_{\text{Boussinesq}}^R)}_{\text{Explicit Correction: } \mathtt{fvc::div}}$$

---

## Step 1: Download LibTorch C++ (in WSL Ubuntu)

Open your WSL terminal and run:

```bash
# 1. Navigate to your home directory in WSL
cd ~

# 2. Download the C++11 ABI LibTorch (CPU version, compatible with Ubuntu GCC)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip -O libtorch.zip

# 3. Unzip into ~/libtorch
unzip -q libtorch.zip
rm libtorch.zip

# 4. Verify ~/libtorch/include and ~/libtorch/lib exist
ls -d ~/libtorch/include ~/libtorch/lib
```

---

## Step 2: Export the Pre-Trained TorchScript Model

In your Windows or WSL terminal where PyTorch is installed, generate the TorchScript binary `tbnn_model.pt`:

```bash
cd "C:/Users/soumi/OneDrive/Documents/Zero data solver/tbnnFoam"
# (or in WSL: cd "/mnt/c/Users/soumi/OneDrive/Documents/Zero data solver/tbnnFoam")

python3 tbnn_export.py
```

This creates `tbnn_model.pt` containing the neural network computational graph.

---

## Step 3: Compile `tbnnFoam` with OpenFOAM `wmake`

In your WSL terminal:

```bash
# 1. Source OpenFOAM environment
source /usr/lib/openfoam/openfoam2606/etc/bashrc

# 2. Navigate to the tbnnFoam solver source directory
cd "/mnt/c/Users/soumi/OneDrive/Documents/Zero data solver/tbnnFoam"

# 3. Compile the solver
wmake
```

When compilation finishes, the binary will be located at:
`$FOAM_USER_APPBIN/tbnnFoam` (e.g., `~/OpenFOAM/soumi-v2606/platforms/linux64GccDPInt32Opt/bin/tbnnFoam`).

To verify it is in your path:
```bash
which tbnnFoam
```

---

## Step 4: Run a Case with `tbnnFoam`

To run any OpenFOAM RANS case (e.g., square duct or backward-facing step):

1. Copy `tbnn_model.pt` into the case's `constant/` directory:
   ```bash
   cp tbnn_model.pt <case_dir>/constant/
   ```

2. Run `tbnnFoam`:
   ```bash
   tbnnFoam
   ```

3. The solver will output the standard velocity $\mathbf{U}$ and pressure $p$, along with:
   * `aTBNN` — The predicted Reynolds stress anisotropy tensor $\mathbf{a}$.
   * `tauR` — The full Reynolds stress tensor $\boldsymbol{\tau}^R$.
   * `tauDelta` — The non-linear anisotropic stress correction.
