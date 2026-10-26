# Contributing to tbnnFoam

Thank you for your interest in contributing to **`tbnnFoam`**! We welcome contributions from the computational fluid dynamics (CFD), physics-informed machine learning (SciML), and scientific computing communities.

---

## Areas Where You Can Contribute

1. **New DNS/LES Benchmarks:** Additional canonical verification test cases (e.g., periodic hills, curved channels, backward-facing steps, asymmetric diffuser).
2. **Turbulence Architectures:** Enhancing invariant representations (e.g., additional invariants, Wall-distance embeddings, pressure-strain tensor models).
3. **OpenFOAM Compatibility:** Testing and maintaining compatibility across newer OpenFOAM versions (v2406, v2606, OpenFOAM-9/10/11).
4. **Performance & GPU Offloading:** CUDA/ROCm LibTorch acceleration for large-scale production meshes (> 5M cells).
5. **Documentation & Tutorials:** Improving build instructions, post-processing scripts, and case setups.

---

## Development Workflow

### 1. Fork & Clone
```bash
git clone https://github.com/BLACKNIGHT-cloud12/tbnnFoam.git
cd tbnnFoam
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Coding Standards & Conventions
* **C++ Code:** Follow OpenFOAM coding style conventions (indentation with 4 spaces, camelCase variable naming, descriptive type aliases).
* **Python Code:** Follow PEP 8 guidelines. Format code using `black` or `flake8`.
* **Preserve Invariance:** Any new machine learning formulation must mathematically respect Galilean and rotational frame-invariance.

### 4. Running Validation Tests
Before submitting a pull request, run the canonical verification case to ensure no numerical divergence or metric regression:
```bash
cd verification_squareDuct
./Allrun
python3 validate_and_plot.py
```

### 5. Submit a Pull Request
* Provide a clear description of the problem solved or feature added.
* Include simulation metrics or validation figures when modifying solver physics.
* Tag related issues.

---

## Code of Conduct
Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.
