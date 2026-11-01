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

## Repository Data Policy

Some large-looking artifacts are committed deliberately, and knowing which ones keeps PRs reviewable:

* **Committed solver result fields** (`verification_squareDuct/300/`, `verification_squareDuct/fine_mesh/300/`, `verification_squareDuct/baseline_sst/`, `verification_squareDuct_Re17800/294/`) are the canonical evaluation inputs: the pytest suite, the CI validation step, and every README metric are generated from them. Do not delete or replace them without regenerating and updating the corresponding metrics.
* **Solver logs** (`log.tbnnFoam`, `log.blockMesh`) are kept as run provenance for the convergence claims in the README. Note that the `verification_squareDuct` log documents a verification re-run that is not the source of the committed snapshot (see the README "Committed Snapshot Provenance" note).
* **`tbnn_model.pt` copies** inside each case's `constant/` are intentional: cases must run standalone from a fresh clone.
* **Documents** exist once each; `docs/tbnnFoam_preprint.pdf` is the canonical compiled manuscript (the former duplicate `docs/latex/main.pdf` was removed).
* **Legacy experiments** under `python/` write only to `python/legacy_exports/` — never to `export_weights/` or the repository-root model artifacts, which must only ever reflect the shipped model trained by `tbnn_train.py`.

Moving the result fields to Git LFS or release assets is possible but would break the pure-Python CI (LFS artifacts are not fetched by default), which is why they remain regular tracked files.

---

## Code of Conduct
Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.
