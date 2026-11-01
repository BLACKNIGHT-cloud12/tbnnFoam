# TASK-03 — `verification_periodicHills/` OpenFOAM case (SST baseline infrastructure)

You are implementing a fully-specified OpenFOAM case package. Do not redesign. The physical
RUN happens later in WSL by the supervisor — your acceptance is file-level correctness.

## Context

tbnnFoam v2 targets separated flow over the canonical periodic hills (Breuer geometry,
Re = 5600). Reference results to validate against (Voet & Laizet 2020, Table 2):
RANS k-ω SST reattachment x/h = 7.684 (DNS: 4.148); SST bubble height at x/h = 2 is 0.62
(DNS 0.53); at x/h = 4: 0.47 (DNS 0.14). The TASK-01 module
(`scripts/validation/separated_flow_metrics.py`) will extract reattachment from τ_w.

## Physical parameters (fixed — do not alter)

- Hill height h = 0.028 m; domain Lx = 9h = 0.252 m, Ly = 3.036h = 0.085008 m (2D, 1-cell z).
- ν = 1.4857e-05 m²/s (air: ρ = 1.225, μ = 1.82e-5). U_b = 2.9714 m/s → Re = 5600.
- Flow driven by constant mass flux: `meanVelocityForce` fvOption with Ubar = (2.9714 0 0).
- Streamwise PERIODIC (cyclic patch pair at x = 0 and x = Lx); no-slip lower (hill) + top walls.
- Geometry: lower wall = the canonical hill contour, provided as a ready blockMesh polyLine
  subsample at `verification_periodicHills/hill_contour_blockmesh.dat`
  (58 points, columns "x/h y/h", x from 0 to 8.9824, y from 0.9951 down to 0 and back).
  At x=0 and x=Lx the contour height is y/h = 0.9951 — the cyclic end faces span
  y/h ∈ [0.9951, 3.036] on both sides (matching faces → valid cyclic pair).

## Files to create (ONLY these)

```
verification_periodicHills/
├── Allrun
├── Allclean
├── 0/U  0/p  0/k  0/omega  0/nut
├── constant/transportProperties
├── constant/turbulenceProperties
├── constant/polyMesh/.gitkeep        (placeholder; blockMesh generates the rest)
└── system/blockMeshDict  controlDict  fvSchemes  fvSolution  fvOptions
```

## Requirements

1. **blockMeshDict**: `convertToMeters 0.028`; vertices
   v0=(0, 0.9951h), v1=(0, 3.036h), v2=(9h, 3.036h), v3=(9h, 0.9951h) [in meters:
   y values × 0.028 — write exact numbers, not expressions]; bottom edge v0→v3 is a
   `polyLine` through ALL 58 contour points converted to meters (inline the literal point
   list — the file must be self-contained, no runtime reads); top edge v1→v2 straight.
   Single hex block: 300 cells x, 120 cells y, 1 cell z; y-grading simpleGrading
   (1 1 1) is NOT acceptable — use two-directional wall refinement via `edgeGrading` or
   multi-grading so first-cell height ≈ 2e-5 m at BOTH walls (target y+ ≈ 1–2 at U_b).
   Patches: `cyclicLeft` (x=0 face), `cyclicRight` (x=Lx face) — type cyclic, matched as a
   cyclic pair via `neighbourPatch`; `hillLower` (bottom polyLine), `topWall` (top),
   `frontBack` (z faces, type empty).
2. **0/ fields**: U initial (2.9714 0 0) uniform; cyclic pair BCs `cyclic`; walls `noSlip`;
   frontBack `empty`. p: zeroGradient walls, cyclic, empty. k/omega/nut: `kqRWallFunction`,
   `omegaWallFunction`, `nutkWallFunction` on hillLower+topWall; cyclic + empty elsewhere.
   Initial k = 1.5·(0.05·2.9714)² ≈ 0.03313; initial omega = sqrt(k)/(0.09^0.25·0.07·0.085008)
   ≈ 40.0 (compute the exact value and write it); nut = nu.
3. **system/fvOptions**: meanVelocityForce as specified above.
4. **constant/transportProperties**: `nu [0 2 -1 0 0 0 0] 1.4857e-05;`
   **constant/turbulenceProperties**: RAS model `kOmegaSST`.
5. **system/controlDict**: application `simpleFoam` (SST baseline; tbnnSimpleFoam run comes
   later); endTime 3000; writeInterval 500; adjustTimeStep no.
6. **fvSchemes/fvSolution**: copy the structure and settings from
   `verification_squareDuct/system/` (READ those files; keep identical scheme choices —
   bounded Gauss upwind etc. — so the v1→v2 comparison stays clean), adapting only patch
   names and relaxation factors if needed.
7. **Allrun** (sh, LF endings): cd to script dir, source RunFunctions if available,
   `rm -rf [1-9]* log.* constant/polyMesh` (keep .gitkeep), `blockMesh > log.blockMesh 2>&1`,
   `simpleFoam > log.simpleFoam 2>&1`. **Allclean**: remove generated fields/logs/mesh,
   keep sources.

## Constraints

- LF line endings everywhere, no emoji. OpenFOAM v2606 (OpenFOAM.org) syntax.
- Do NOT modify any file outside `verification_periodicHills/`.
- numpy allowed only if you add a helper script (prefer none — inline literals).

## Acceptance criteria (file-level — the supervisor runs OpenFOAM afterwards)

1. blockMeshDict parses visually: all 58 polyLine points present with strictly increasing x,
   y converted to meters (× 0.028) exactly.
2. Every patch named in blockMeshDict has corresponding entries in ALL 0/ files.
3. Cyclic pair declarations correct (`cyclicLeft` ↔ `cyclicRight`, `neighbourPatch` set).
4. Allrun/Allclean are LF sh scripts; no CRLF.
5. Numbers cross-checked: Re = U_b·h/ν = 2.9714·0.028/1.4857e-05 = 5600 (±0.1%).

## When done

Write `parallel-tasks/TASK-03-RESULT.md`: files created, the self-checks you ran
(patch-name cross-check, polyLine monotonicity, unit conversion spot-checks), deviations (none expected).
