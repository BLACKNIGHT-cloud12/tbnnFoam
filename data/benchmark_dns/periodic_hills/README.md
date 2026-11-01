# Periodic Hills Benchmark Data (tbnnFoam v2) — Provenance & Acquisition

Same posture as v1: **raw third-party DNS data is NOT redistributed** and stays out of the
git tree. This file documents provenance, licenses, download commands, and the exact role of
each dataset. Both datasets are open — no upstream permission is required.

---

## A. VALIDATION dataset (THE figure) — TUM DNS reference data

| | |
| :--- | :--- |
| Title | DNS reference data of flow over periodic hills at Re = 5600 and 10595 |
| Authors | Krank, Benjamin; Kronbichler, Martin; Wall, Wolfgang A. (TUM Institute for Computational Mechanics) |
| DOI | `10.14459/2018mp1415670` |
| License | **CC BY 4.0** (attribution only) |
| Contents | Mean-velocity, Reynolds-shear-stress, and TKE **profiles at 10 streamwise stations**; **skin-friction (Cf) and pressure (Cp) coefficients at upper and lower walls**. Re_H = 5600 and 10,595. |
| Size | 2 files, ~382 KB |

**Role:** a posteriori validation — Cf/Cp distributions, station profiles, reattachment
length. This is the reference side of THE figure. **Not usable for training** (no full-plane
fields).

**Download** (either):
- Browser: <https://mediatum.ub.tum.de/1415670>
- `rsync rsync://m1370728@dataserv.ub.tum.de/m1415670/` (rsync password: `m1415670`)

Place files in `data/benchmark_dns/periodic_hills/tum_raw/` (git-ignored).

## B. TRAINING dataset (a priori) — full-field DNS (Voet & Laizet)

| | |
| :--- | :--- |
| Title | RANS and DNS simulations of flow over 2D periodically arranged hills |
| Authors | Voet, Laurens; Laizet, Sylvain (Imperial College; Incompact3d) |
| DOI | `10.17632/by72n3s9bn.2` (Mendeley Data, version 2) |
| Contents | **7 DNS + 30 RANS** of flow over periodically arranged hills; geometry varied by two parameters (hill steepness angle, streamwise separation) |
| License | Confirm exact terms on the dataset page at download time and record them here (Mendeley datasets are typically CC BY 4.0 / CC0) |

**Role:** full-plane training fields (velocity + Reynolds stresses on the x–y grid) for the
frozen training CSV (see `scripts/convert_hills_dns.py`). The **canonical-geometry DNS** is
the v2 training case. The geometry-parameter variation is a future generalization study —
explicitly out of v2 scope.

**Download:** <https://data.mendeley.com/datasets/by72n3s9bn> → place in
`data/benchmark_dns/periodic_hills/voet_raw/` (git-ignored).

The dataset's own geometry files also provide the **hill contour** used for (a) the
converter's wall-distance computation and (b) the TASK-03 OpenFOAM case mesh.

---

## Conversion workflow

1. `python scripts/convert_hills_dns.py --raw <raw_path> --out <csv_path> --contour <contour_file>`
2. Output schema is frozen (TASK-02): `x_over_h, y_over_h, U_over_Ub, V_over_Ub, W_over_Ub,
   uu, vv, ww, uv, uw, vw, k, omega`
3. The parser (`default_parser`) is adapted to the actual raw format once the datasets are
   downloaded — the schema does not change.
4. Converted CSVs stay local (git-ignored), exactly like the v1 duct CSVs.

## Citations (mandatory in any publication using these data)

1. Krank, B., Kronbichler, M., & Wall, W. A. (2018). *Direct numerical simulation of flow
   over periodic hills up to Re_H = 10,595.* Computers & Fluids. Data: DOI
   `10.14459/2018mp1415670` (CC BY 4.0).
2. Voet, L. J. A., & Laizet, S. (2020). *A hybrid approach combining DNS and RANS
   simulations to quantify uncertainties in turbulence modelling.* Applied Mathematical
   Modelling. Data: DOI `10.17632/by72n3s9bn.2`. (Free PDF via the author's site,
   turbulencesimulation.com.)

## Directory layout (post-download)

```
data/benchmark_dns/periodic_hills/
├── README.md          # this file (committed)
├── tum_raw/           # git-ignored
├── voet_raw/          # git-ignored
└── *.csv              # converted training CSVs (git-ignored, stays local)
```
