# Benchmark DNS Datasets

Authoritative Direct Numerical Simulation (DNS) benchmark datasets for turbulent square duct flow from the Sapienza University of Rome CFD Group.

## Primary Citations & Provenance
* **Pirozzoli, S., Modesti, D., Orlandi, P., & Grasso, F. (2018).** *Turbulence and secondary motions in square duct flow.* Journal of Fluid Mechanics, 840, 634-655.
* **Vinuesa, R., Noorani, A., Lozano-Durán, A., El Khoury, G. K., Schlatter, P., Fischer, P. F., & Nagib, H. M. (2014).** *Aspect ratio effects in turbulent duct flows studied through direct numerical simulation.* Journal of Turbulence, 15(10), 677-706.
* **Database Repository:** Sapienza University of Rome CFD Group DNS Database (http://newton.dma.uniroma1.it/square_duct/).

## Dataset Files
1. **`square_duct_dns_Reb4410_Retau150.csv`**:
   - Reb = 4,410 (Reh = 2,205, Retau = 150).
   - 4,225 grid cells (65 x 65 cross-section).
   - Contains: (y, z) coordinates, mean velocity gradients dUi/dxj, Reynolds stress anisotropy tensor components a_ij, and specific dissipation surrogate omega.
2. **`square_duct_dns_Reb7000_Retau220.csv`**:
   - Reb = 7,000 (Reh = 3,500, Retau = 220).
   - 4,225 grid cells (65 x 65 cross-section).
3. **`raw_dns/plotyz_Retau150.dat`** & **`raw_dns/plotyz_Retau220.dat`**:
   - Raw exported data files from the Pirozzoli et al. database.
4. **`benchmark_metadata.json`**:
   - Comprehensive machine-readable specification of physical scaling, Reynolds number definitions, coordinate systems, and baseline flow parameters.

## License and Terms of Use
* **Software License Independence:** The project's MIT software license applies exclusively to the `tbnnFoam` source code, solvers, conversion routines, and verification case configurations authored by the project.
* **Third-Party Data Ownership:** All intellectual property and copyrights for the raw and processed DNS simulation datasets belong to the original authors: Sergio Pirozzoli, Daniele Modesti, Paolo Orlandi, and Francesco Grasso (Sapienza University of Rome CFD Group).
* **Research Use & Attribution:** These datasets are provided solely as reference inputs for benchmark verification and computational reproducibility. The inclusion of these datasets does not imply sublicensing, ownership transfer, or commercial redistribution rights. Any academic publication, report, or comparative study utilizing these datasets must cite Pirozzoli et al. (2018).
