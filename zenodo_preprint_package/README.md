# In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM: A Critical Verification, Validation, and Reproducibility Study

**Author:** Soumik Ghosh (Department of Civil Engineering, Jadavpur University, India)

**Version:** 1.0 · **Deposit date:** 2026-11-01 · **License:** CC BY 4.0 (paper) — the companion software is MIT-licensed

## Abstract

Linear eddy-viscosity closures cannot reproduce the normal-stress anisotropy that drives Prandtl's second-kind secondary flows in square ducts. Tensor Basis Neural Networks (TBNN; Ling et al., J. Fluid Mech., 2016) represent Reynolds-stress anisotropy by projection onto Pope's ten-term integrity basis, but their in-situ coupling into live CFD solvers remains challenging. We present tbnnFoam, an in-situ coupled solver within OpenFOAM 2606 featuring a dual-runtime inference architecture: an embedded LibTorch engine and a zero-dependency standalone C++ evaluator with an analytical Cardano cubic eigensolver that enforces realizability on the Lumley triangle in O(1) operations per cell. Cross-engine parity across PyTorch, TorchScript, and the standalone evaluator is verified within an absolute tolerance of 10^-5, and in the Boussinesq limit the coupled momentum equation reduces exactly to the baseline linear model (maximum relative deviation 3.6 × 10^-17). Validation on developing square-duct flow at Re_b = 4410 against the DNS of Pirozzoli et al. (J. Fluid Mech., 2018) yields a partial validation: the coupled solver recovers all eight secondary-flow cells that the tested linear k–ω SST closure fails to reproduce, but detected core locations are offset by 0.447–0.567 h from the raw-field-verified DNS reference (0.577, 0.847), the streamwise-velocity NRMSE is 9.34%, and peak secondary speed decreases from 7.99% to 2.62% of U_b under 4× cross-stream refinement against the 1.93% DNS benchmark. We analyze the sources of these discrepancies—the empirical ω surrogate, first-order upwinding, wall-target degeneracy, and the developing-flow versus periodic-DNS mismatch—and quantify distribution shift between training and live states (median k-NN distance 0.21 σ). All solver sources, trained weights, and an automated 21-test verification suite are openly available under the MIT license, with a documented workflow for obtaining the third-party DNS benchmark data.

## Related software

This preprint is the research paper accompanying **tbnnFoam v1.0.0**, archived separately on Zenodo:
DOI [10.5281/zenodo.23799524](https://doi.org/10.5281/zenodo.23799524) (MIT license). Source code: https://github.com/BLACKNIGHT-cloud12/tbnnFoam

The third-party benchmark DNS data is not redistributed with this deposit; the acquisition and
conversion workflow is documented in the software repository (`data/benchmark_dns/README.md`).

## How to cite

> Ghosh, S. (2026). *In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM: A Critical Verification, Validation, and Reproducibility Study*. Zenodo. Preprint, version 1.0.
> The DOI for this record is assigned by Zenodo when the record is published and is displayed on the record page.
> Please also cite the companion software record (10.5281/zenodo.23799524) when the solver itself is used.

## Files in this deposit

All six files in this folder are intended for upload to the Zenodo record:

- `final_preprint.pdf` — the manuscript (28 pages).
- `README.md` — this file.
- `final_preprint_latex_source.zip` — LaTeX source (main.tex, references.bib, main.bbl, figures) for reproducible compilation.
- `zenodo_preprint.json` — prepared Zenodo metadata (machine-readable reference for the record).
- `final_preprint.pdf.sha256`, `final_preprint_latex_source.zip.sha256` — integrity checksums.
