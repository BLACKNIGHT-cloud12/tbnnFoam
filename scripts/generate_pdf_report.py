import base64
import os
import subprocess
from pathlib import Path

def img_to_b64(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')

repo_root = Path(__file__).resolve().parent.parent
fig_dir = repo_root / 'docs' / 'figures'

b64_arch = img_to_b64(fig_dir / 'architecture_pipeline.png')
b64_reynolds = img_to_b64(fig_dir / 'reynolds_stress_mechanism.png')
b64_network = img_to_b64(fig_dir / 'tbnn_network_architecture.png')
b64_bary = img_to_b64(fig_dir / 'barycentric_anisotropy_map.png')
b64_hero = img_to_b64(fig_dir / 'tbnn_hero_showcase.png')
b64_envelope = img_to_b64(fig_dir / 'timescale_and_invariant_distribution.png')
b64_mesh = img_to_b64(fig_dir / 'mesh_sensitivity_study.png')
b64_re17800 = img_to_b64(fig_dir / 'tbnn_Re17800_showcase.png')

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>tbnnFoam Technical Report - Soumik Ghosh</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 14mm 16mm 14mm;
    @bottom-right {{
      content: "Page " counter(page) " of " counter(pages);
      font-family: 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: #666;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1a202c;
    background-color: #ffffff;
    line-height: 1.45;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
  }}
  .page {{
    page-break-after: always;
    position: relative;
    padding-bottom: 5mm;
  }}
  .page:last-child {{
    page-break-after: avoid;
  }}
  .header-banner {{
    border-bottom: 2.5px solid #004c87;
    padding-bottom: 8px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .header-banner h1 {{
    margin: 0;
    color: #004c87;
    font-size: 19pt;
    font-weight: 800;
    letter-spacing: -0.5px;
  }}
  .header-banner .subtitle {{
    margin: 3px 0 0 0;
    font-size: 10.5pt;
    color: #4a5568;
    font-weight: 500;
  }}
  .meta-box {{
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 8.5pt;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 16px;
    margin-bottom: 14px;
  }}
  .meta-item strong {{
    color: #2d3748;
  }}
  h2 {{
    color: #004c87;
    font-size: 12.5pt;
    font-weight: 700;
    border-left: 3.5px solid #004c87;
    padding-left: 8px;
    margin-top: 14px;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
  }}
  h3 {{
    color: #2b6cb0;
    font-size: 10.5pt;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 5px;
  }}
  p {{
    margin: 0 0 8px 0;
    text-align: justify;
  }}
  .highlight-card {{
    background: #ebf8ff;
    border-left: 3.5px solid #3182ce;
    padding: 8px 12px;
    margin-bottom: 10px;
    font-size: 9pt;
    border-radius: 0 4px 4px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 8px 0 12px 0;
  }}
  th, td {{
    padding: 5px 8px;
    border: 1px solid #cbd5e0;
    text-align: left;
  }}
  th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 700;
  }}
  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}
  .table-center th, .table-center td {{
    text-align: center;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 7.5pt;
    font-weight: 700;
  }}
  .badge-pass {{ background: #c6f6d5; color: #22543d; }}
  .badge-primary {{ background: #bee3f8; color: #2a4365; }}
  .badge-warning {{ background: #feebc8; color: #744210; }}
  .fig-container {{
    text-align: center;
    margin: 10px 0;
  }}
  .fig-container img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .fig-caption {{
    font-size: 8pt;
    color: #4a5568;
    margin-top: 4px;
    font-style: italic;
  }}
  .formula-box {{
    background: #f7fafc;
    border: 1px solid #edf2f7;
    border-radius: 4px;
    padding: 6px 12px;
    margin: 6px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.5pt;
    text-align: center;
  }}
  .code-inline {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.5pt;
    background: #edf2f7;
    padding: 1px 4px;
    border-radius: 3px;
    color: #805ad5;
  }}
  ul, ol {{
    margin: 0 0 8px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3px;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 8px 0;
  }}
</style>
</head>
<body>

<!-- PAGE 1: TITLE & EXECUTIVE SUMMARY -->
<div class="page">
  <div class="header-banner">
    <div>
      <h1>tbnnFoam: Technical Project Report</h1>
      <div class="subtitle">In-Situ Tensor Basis Neural Network Turbulence Closure in OpenFOAM</div>
    </div>
    <div style="text-align:right;">
      <span class="badge badge-primary">CFD & SciML Portfolio</span>
    </div>
  </div>

  <div class="meta-box">
    <div class="meta-item"><strong>Author:</strong> Soumik Ghosh (Student Project)</div>
    <div class="meta-item"><strong>Date:</strong> October 2026</div>
    <div class="meta-item"><strong>OpenFOAM Compatibility:</strong> v2012 — v2606 (C++)</div>
    <div class="meta-item"><strong>Public Repository:</strong> github.com/BLACKNIGHT-cloud12/tbnnFoam</div>
    <div class="meta-item"><strong>Verification Suite:</strong> 17/17 Pytest Tests Passing (100%)</div>
    <div class="meta-item"><strong>Benchmark Dataset:</strong> Pirozzoli et al. (2018) DNS (Re_tau = 150 & 220)</div>
  </div>

  <h2>1. Executive Summary & Problem Motivation</h2>
  <p>
    Standard linear eddy-viscosity RANS turbulence models (e.g., standard k-&epsilon; and k-&omega; SST) rely on the Boussinesq approximation, which assumes an isotropic turbulent viscosity &nu;<sub>t</sub>. Because normal stresses are constrained to be isotropic (&macr;v'<sup>2</sup> = &macr;w'<sup>2</sup>), linear RANS models are mathematically incapable of predicting turbulence-driven secondary flows of the second kind (Prandtl, 1926). In non-circular ducts, this failure causes linear solvers to predict strictly axial flow (V = W = 0), completely missing the 8 characteristic corner vortex pairs and corner velocity bulging observed in experiments and Direct Numerical Simulations (DNS).
  </p>
  <p>
    This project introduces <strong><code>tbnnFoam</code></strong>, a complete, production-grade, in-situ implementation of the Tensor Basis Neural Network (TBNN) of Ling et al. (2016) directly integrated inside OpenFOAM's finite volume discretization loop.
  </p>

  <div class="highlight-card">
    <strong>Key Technical Achievements:</strong>
    <ul style="margin-top:4px; margin-bottom:0;">
      <li><strong>Dual C++ Inference Engines:</strong> An embedded LibTorch solver (<code>tbnnFoam</code>) evaluating TorchScript models alongside a standalone, zero-dependency C++ header-only evaluator (<code>tbnnSimpleFoam</code>) loading JSON weights via native <code>wmake</code>.</li>
      <li><strong>Analytical Realizability Limiter:</strong> Exact closed-form Cardano cubic eigensolver enforcing Lumley triangle bounds in &Omicron;(1) operations per cell, preventing solver blowup.</li>
      <li><strong>Double-Counting Prevention:</strong> Formulated an explicit perturbation stress &tau;<sub>&Delta;</sub> = 2ka + 2&nu;<sub>t</sub>S preserving the diagonal dominance of OpenFOAM's implicit momentum predictor.</li>
      <li><strong>DNS Benchmark Verification:</strong> Successfully recovered all 8 secondary corner circulation cells on a canonical square duct mesh, verified with an automated 17-test suite.</li>
    </ul>
  </div>

  <h2>2. Solver Pipeline & Finite-Volume Coupling</h2>
  <p>
    During each SIMPLE iteration, the solver dynamically extracts velocity gradients, evaluates non-dimensional invariants and tensor bases, performs neural inference, enforces Lumley realizability, and injects the anisotropic stress divergence:
  </p>
  <div class="fig-container">
    <img src="{b64_arch}" style="max-height: 220px;" alt="Solver Architecture Pipeline">
    <div class="fig-caption">Figure 1: In-situ momentum coupling pipeline inside OpenFOAM's finite volume loop.</div>
  </div>
</div>

<!-- PAGE 2: MATHEMATICAL FORMULATION -->
<div class="page">
  <h2>3. Mathematical Formulation & Physical Mechanisms</h2>
  
  <h3>3.1 Secondary Flow Generation Mechanism</h3>
  <p>
    In straight, non-circular ducts, the streamwise mean vorticity &omega;<sub>x</sub> is governed by the axial transport equation:
  </p>
  <div class="formula-box">
    <strong>U</strong> &sdot; &nabla;&omega;<sub>x</sub> &approx; &part;<sup>2</sup>/(&part;y &part;z) (&macr;w'<sup>2</sup> - &macr;v'<sup>2</sup>) + (&part;<sup>2</sup>/&part;z<sup>2</sup> - &part;<sup>2</sup>/&part;y<sup>2</sup>) &macr;v'w'
  </div>
  <p>
    Linear eddy-viscosity models enforce (&macr;w'<sup>2</sup> - &macr;v'<sup>2</sup>) = 0 everywhere. Consequently, the production of axial vorticity is identically zero. The TBNN predicts the full Reynolds stress anisotropy tensor <strong>a</strong>, capturing the alternating quadrant lobes of normal stress differences that drive secondary vortices:
  </p>
  
  <div class="fig-container">
    <img src="{b64_reynolds}" style="max-height: 180px;" alt="Reynolds Stress Mechanism">
    <div class="fig-caption">Figure 2: Normal stress difference gradients (&part;<sup>2</sup>/&part;y&part;z)(&macr;w'<sup>2</sup> - &macr;v'<sup>2</sup>) extracted from tbnnFoam simulation.</div>
  </div>

  <h3>3.2 Pope (1975) Integrity Basis Expansion</h3>
  <p>
    The deviatoric Reynolds stress anisotropy tensor <strong>a</strong> = &tau;<sup>R</sup>/(2k) - (1/3)<strong>I</strong> is expanded into 10 symmetric, trace-free tensor bases:
  </p>
  <div class="formula-box">
    <strong>a</strong> = &sum;<sub>n=1</sub><sup>10</sup> g<sup>(n)</sup>(&lambda;<sub>1</sub>, ..., &lambda;<sub>5</sub>) <strong>T</strong><sup>(n)</sup>
  </div>
  <p>
    where normalized strain rate <strong>s</strong> = (k/&epsilon;)<strong>S</strong> and rotation rate <strong>w</strong> = (k/&epsilon;)<strong>W</strong> yield the 5 scalar invariants:
  </p>
  <div class="formula-box">
    &lambda;<sub>1</sub> = tr(<strong>s</strong><sup>2</sup>), &quad; &lambda;<sub>2</sub> = tr(<strong>w</strong><sup>2</sup>), &quad; &lambda;<sub>3</sub> = tr(<strong>s</strong><sup>3</sup>), &quad; &lambda;<sub>4</sub> = tr(<strong>w</strong><sup>2</sup><strong>s</strong>), &quad; &lambda;<sub>5</sub> = tr(<strong>w</strong><sup>2</sup><strong>s</strong><sup>2</sup>)
  </div>

  <h3>3.3 Analytical Cardano Realizability Engine</h3>
  <p>
    Physical realizability requires the eigenvalues of <strong>a</strong> to satisfy the Lumley triangle: -1/3 &le; &lambda;<sub>3</sub> &le; &lambda;<sub>2</sub> &le; &lambda;<sub>1</sub> &le; 2/3. Rather than iterative numerical methods, <code>tbnnEvaluator.H</code> implements an exact Cardano closed-form solver:
  </p>
  <div class="formula-box">
    p = (1/2) tr(<strong>a</strong><sup>2</sup>), &quad; q = det(<strong>a</strong>), &quad; &lambda;<sub>k</sub> = 2&radic;(p/3) cos((1/3) arccos((3q)/(2p&radic;(p/3))) - 2&pi;k/3)
  </div>
  <p>
    If the smallest eigenvalue violates &lambda;<sub>3</sub> &ge; -1/3, the anisotropy tensor is radially scaled <strong>a</strong> &larr; s <strong>a</strong>, guaranteeing frame-invariant positive semi-definiteness in &Omicron;(1) time.
  </p>
</div>

<!-- PAGE 3: DUAL INFERENCE & ARCHITECTURE -->
<div class="page">
  <h2>4. Neural Network Architecture & Dual Inference Paths</h2>
  <div class="fig-container">
    <img src="{b64_network}" style="max-height: 200px;" alt="TBNN Network Topology">
    <div class="fig-caption">Figure 3: Neural network topology mapping 5 scalar invariants to 10 tensor expansion coefficients.</div>
  </div>

  <h3>4.1 Dual Production In-Situ Engines</h3>
  <p>
    To resolve cluster dependency conflicts while maintaining high-performance execution, the codebase provides two complementary inference engines:
  </p>
  <table>
    <thead>
      <tr>
        <th style="width:25%;">Feature</th>
        <th style="width:37%;">Embedded LibTorch (tbnnFoam)</th>
        <th style="width:38%;">Standalone C++ (tbnnSimpleFoam)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Inference Engine</strong></td>
        <td>Direct C++ <code>torch::jit</code> evaluation</td>
        <td>Header-only C++ matrix engine (<code>tbnnEvaluator.H</code>)</td>
      </tr>
      <tr>
        <td><strong>Model Source</strong></td>
        <td>Serialized TorchScript (<code>tbnn_model.pt</code>)</td>
        <td>Structured JSON weights (<code>weights.json</code>)</td>
      </tr>
      <tr>
        <td><strong>External Dependencies</strong></td>
        <td>PyTorch / LibTorch runtime C++ libraries</td>
        <td><strong>Zero third-party dependencies</strong> (Native OpenFOAM wmake)</td>
      </tr>
      <tr>
        <td><strong>Hidden Layers</strong></td>
        <td>64 units / layer (ReLU)</td>
        <td>30 units / layer (ReLU)</td>
      </tr>
      <tr>
        <td><strong>Cluster Portability</strong></td>
        <td>Requires matching CUDA/LibTorch ABI</td>
        <td><strong>100% portable across any HPC cluster</strong></td>
      </tr>
      <tr>
        <td><strong>Realizability</strong></td>
        <td>Lumley clipping via Cardano eigensolver</td>
        <td>Lumley clipping via Cardano eigensolver</td>
      </tr>
    </tbody>
  </table>

  <h3>4.2 Turbulence Anisotropy State: Barycentric Map</h3>
  <div class="fig-container">
    <img src="{b64_bary}" style="max-height: 210px;" alt="Barycentric Map">
    <div class="fig-caption">Figure 4: Barycentric map of Reynolds stress anisotropy states computed from simulation outputs.</div>
  </div>
</div>

<!-- PAGE 4: BENCHMARK HERO SHOWCASE -->
<div class="page">
  <h2>5. Primary DNS Benchmark Validation (Re_tau = 150)</h2>
  <p>
    Validation against the canonical square duct DNS benchmark of Pirozzoli et al. (2018) at Re<sub>b</sub> &approx; 4410, Re<sub>h</sub> &approx; 2205, Re<sub>&tau;</sub> = 150:
  </p>
  <div class="fig-container">
    <img src="{b64_hero}" style="max-height: 380px;" alt="TBNN Hero Showcase">
    <div class="fig-caption">Figure 5: (a) Predicted corner secondary streamlines; (b) Streamwise velocity profile along corner diagonal; (c-d) Vector and reference DNS vortex center probability density functions (blue: ACW, red: CW).</div>
  </div>

  <h3>5.1 Independent Metrics Comparison Table</h3>
  <table class="table-center">
    <thead>
      <tr>
        <th style="text-align:left;">Verification Metric</th>
        <th>Standard OpenFOAM (k-&omega; SST)</th>
        <th>tbnnFoam (Our TBNN Solver)</th>
        <th>Reference DNS (Pirozzoli 2018)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><strong>Secondary Vortex Cores</strong></td>
        <td>0 cores (Absence)</td>
        <td><strong style="color:#2b6cb0;">8 cores (Detected)</strong></td>
        <td>8 cores (Canonical)</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Peak Secondary V_sec / Ub</strong></td>
        <td>0.45% (Discretization noise)</td>
        <td><strong>1.59%</strong></td>
        <td>1.92%</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Q1 Core Coordinates (y/h, z/h)</strong></td>
        <td>Undefined (No vortices)</td>
        <td><strong>(0.91, 0.56)</strong></td>
        <td>(0.50, 0.82)</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Core Localization Error</strong></td>
        <td>N/A</td>
        <td><strong>0.481 h</strong></td>
        <td>0.000 h (Ref)</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Streamwise U NRMSE</strong></td>
        <td>6.12%</td>
        <td>15.51%</td>
        <td>0.00% (Ref)</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Streamwise U RMSE</strong></td>
        <td>0.0612 m/s</td>
        <td>0.1551 m/s</td>
        <td>0.0000 m/s</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Secondary (V, W) RMSE</strong></td>
        <td>0.0087 m/s</td>
        <td>0.0142 m/s</td>
        <td>0.0000 m/s</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Continuity Residual</strong></td>
        <td>&lt; 1e-6 (Global: 7.10e-8)</td>
        <td><strong>&lt; 1e-6 (Global: 5.56e-7)</strong></td>
        <td>N/A</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- PAGE 5: INVARIANT MANIFOLD COVERAGE -->
<div class="page">
  <h2>6. In-Situ Invariant Manifold & Operating Envelope Coverage</h2>
  <p>
    A major critique of data-driven turbulence closures is "out-of-distribution drift" during in-situ CFD coupling. To rigorously verify that the neural network operates purely in interpolation mode, <code>evaluate_timescale_distribution.py</code> computed the operating envelope of all 5 Cayley-Hamilton invariants across all 20,000 CFD cells against the DNS training manifold:
  </p>
  <div class="fig-container">
    <img src="{b64_envelope}" style="max-height: 230px;" alt="Timescale and Invariant Distribution">
    <div class="fig-caption">Figure 6: Cross-evaluation of CFD operating ranges vs. DNS training bounding envelope for invariants I<sub>1</sub> to I<sub>5</sub>.</div>
  </div>

  <table class="table-center">
    <thead>
      <tr>
        <th style="text-align:left;">Invariant / Feature</th>
        <th>DNS Training Envelope</th>
        <th>OpenFOAM CFD Operating Range (All 20k Cells)</th>
        <th>In-Distribution Coverage</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><strong>I<sub>1</sub> = tr(s<sup>2</sup>)</strong></td>
        <td>[0.000, 0.933]</td>
        <td>[1.665e-4, 0.0686]</td>
        <td><span class="badge badge-pass">100.0% In-Bounds</span></td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>I<sub>2</sub> = tr(w<sup>2</sup>)</strong></td>
        <td>[-0.932, 0.000]</td>
        <td>[-0.0682, -1.218e-8]</td>
        <td><span class="badge badge-pass">100.0% In-Bounds</span></td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>I<sub>3</sub> = tr(s<sup>3</sup>)</strong></td>
        <td>[-0.0349, 0.00426]</td>
        <td>[-9.645e-4, 1.166e-3]</td>
        <td><span class="badge badge-pass">100.0% In-Bounds</span></td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>I<sub>4</sub> = tr(w<sup>2</sup>s)</strong></td>
        <td>[-0.00142, 0.0116]</td>
        <td>[-3.909e-4, 3.180e-4]</td>
        <td><span class="badge badge-pass">100.0% In-Bounds</span></td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>I<sub>5</sub> = tr(w<sup>2</sup>s<sup>2</sup>)</strong></td>
        <td>[-0.435, 0.000]</td>
        <td>[-2.338e-3, -9.433e-12]</td>
        <td><span class="badge badge-pass">100.0% In-Bounds</span></td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Basis Norms ||T<sup>(n)</sup>||<sub>F</sub></strong></td>
        <td>[0, 1.318]</td>
        <td>[0, 0.243]</td>
        <td><span class="badge badge-pass">100.0% Bounded</span></td>
      </tr>
    </tbody>
  </table>

  <div class="highlight-card">
    <strong>Zero Extrapolation Finding:</strong> The time-scale limiter &tau; = min(5.0, 1/&omega;) naturally damps high near-wall strain rates while capping the duct core. Every single cell across the 3D duct falls strictly within the bounding manifold, ensuring that the model operates in pure interpolation mode.
  </div>
</div>

<!-- PAGE 6: MESH RESOLUTION STUDY -->
<div class="page">
  <h2>7. Cross-Sectional Mesh Resolution & Sensitivity Study</h2>
  <p>
    An independent grid sensitivity study was conducted comparing the standard base mesh against a 4x refined cross-sectional grid to examine discretization error and vortex core localization:
  </p>
  <div class="fig-container">
    <img src="{b64_mesh}" style="max-height: 250px;" alt="Mesh Sensitivity Study">
    <div class="fig-caption">Figure 7: Comparison of secondary speed contours, wall bisector profiles, and summary metrics between coarse (20k) and refined (80k) meshes.</div>
  </div>

  <table class="table-center">
    <thead>
      <tr>
        <th style="text-align:left;">Grid Resolution</th>
        <th>Cells</th>
        <th>Peak V_sec / Ub</th>
        <th>Q1 Core Centroid</th>
        <th>Core Localization Error</th>
        <th>Cores Detected</th>
        <th>Streamwise U RMSE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><strong>Coarse Base (M1)</strong></td>
        <td>20,000</td>
        <td>1.64%</td>
        <td>(0.850, 0.450)</td>
        <td>0.509 h</td>
        <td>8 cores</td>
        <td>0.1551 m/s</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>Refined Grid (M2)</strong></td>
        <td>80,000</td>
        <td>2.31%</td>
        <td>(0.825, 0.525)</td>
        <td>0.439 h</td>
        <td>8 cores</td>
        <td>0.0832 m/s</td>
      </tr>
      <tr>
        <td style="text-align:left;"><strong>DNS Benchmark Target</strong></td>
        <td>N/A</td>
        <td><strong>2.03%</strong></td>
        <td>(0.500, 0.820)</td>
        <td><strong>0.000 h (Ref)</strong></td>
        <td><strong>8 cores</strong></td>
        <td><strong>0.0000 m/s</strong></td>
      </tr>
    </tbody>
  </table>

  <ul>
    <li><strong>Streamwise Error Reduction:</strong> Refining the cross-section decreases streamwise velocity RMSE by <strong>46.4%</strong> (from 0.1551 m/s down to 0.0832 m/s).</li>
    <li><strong>Topological Stability:</strong> Both grid levels stably identify exactly <strong>8 distinct vortex cores</strong> using connected-region component labeling without heuristic fallbacks.</li>
    <li><strong>Secondary Momentum Recovery:</strong> Peak secondary circulation increases from 1.64% to 2.31%, closely bracketing the DNS reference value of 2.03%.</li>
  </ul>
</div>

<!-- PAGE 8: HIGH-REYNOLDS NUMBER CASE STUDY -->
<div class="page">
  <h2>8. High-Reynolds-Number Case Study (Re<sub>b</sub> = 17,800, Re<sub>&tau;</sub> &approx; 519)</h2>
  <p>
    To evaluate model scalability to high Reynolds numbers matching the asymptotic DNS benchmark regime of Pirozzoli et al. (2018), an independent simulation was conducted at bulk Reynolds number <strong>Re<sub>b</sub> = 17,800</strong> (Re<sub>h</sub> = 8,900, Re<sub>&tau;</sub> &approx; 519) with &nu; = 5.618 &times; 10<sup>-6</sup> m<sup>2</sup>/s and U<sub>b</sub> = 1.0 m/s on an 80k-cell mesh (50 &times; 40 &times; 40):
  </p>

  <div class="fig-container">
    <img src="{b64_re17800}" style="max-height: 250px;" alt="High-Re Re=17800 Showcase">
    <div class="fig-caption">Figure 8: High-Reynolds-number validation at Re<sub>b</sub> = 17,800: (a) Streamwise velocity contours and secondary streamlines; (b) Secondary flow velocity vectors and detected 8 vortex cores; (c) Streamwise spatial development of secondary speed and centerline velocity; (d) Cross-stream vertical velocity profile across vortex core line y/h = 0.525.</div>
  </div>

  <table class="table-center">
    <thead>
      <tr>
        <th>Station</th>
        <th>x/D<sub>h</sub></th>
        <th>U<sub>c</sub>/U<sub>b</sub></th>
        <th>Raw Max V<sub>sec</sub>/U<sub>b</sub></th>
        <th>Interior Max (|y|,|z| &le; 0.85h)</th>
        <th>Cores</th>
        <th>Q1 Core Distance to DNS</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>#1</td><td>1.1</td><td>1.019</td><td>9.63%</td><td>1.91%</td><td>0 (Rollup)</td><td>N/A</td></tr>
      <tr><td>#3</td><td>3.1</td><td>1.054</td><td>10.43%</td><td>2.39%</td><td>0 (Rollup)</td><td>N/A</td></tr>
      <tr><td>#6</td><td>6.1</td><td>1.096</td><td>9.69%</td><td>1.84%</td><td><strong>8 cores</strong></td><td>0.137 h</td></tr>
      <tr><td>#7</td><td>7.1</td><td>1.108</td><td>9.55%</td><td>1.78%</td><td><strong>8 cores</strong></td><td>0.093 h</td></tr>
      <tr><td>#9</td><td>9.1</td><td>1.132</td><td>9.39%</td><td>1.67%</td><td><strong>8 cores</strong></td><td><strong>0.060 h (1.2 cells)</strong></td></tr>
      <tr style="background:#edf2f7; font-weight:bold;">
        <td>DNS Ref</td><td>&infin;</td><td>1.33</td><td>2.03% - 2.30%</td><td>2.03% - 2.30%</td><td>8 cores</td><td>0.000 h (Ref)</td>
      </tr>
    </tbody>
  </table>

  <ul>
    <li><strong>Topological Stability:</strong> The solver converges stably in 294 SIMPLE iterations, capturing all 8 distinct corner counter-rotating vortex pairs across the downstream developing region (x/D<sub>h</sub> &ge; 6.1).</li>
    <li><strong>Core Speed Discrepancy:</strong> Downstream station interior maxima settle at 1.67% - 1.84% (mean core centroid speed 1.71%), which is 9.4% - 17.7% below the cited asymptotic periodic DNS lower bound (2.03%). This is consistent with an incompletely developed spatial duct (U<sub>c</sub>/U<sub>b</sub> = 1.132 vs DNS U<sub>c</sub>/U<sub>b</sub> &approx; 1.33).</li>
    <li><strong>Proposed Near-Wall Mechanism:</strong> The raw domain maximum (9.40%) occurs strictly in the outermost boundary-layer cell center (|z/h| = 0.975); this is framed as a proposed numerical/wall-treatment mechanism under steep velocity gradients, while the interior vortex core circulation remains smooth at &sim; 1.71%.</li>
  </ul>
</div>

<!-- PAGE 9: AUTOMATED TEST SUITE -->
<div class="page">
  <h2>9. Automated Verification & Pytest Suite Breakdown</h2>
  <p>
    The codebase includes an automated continuous test suite with <strong>17 unit and parity tests</strong> enforcing strict mathematical and numerical integrity across all modules:
  </p>

  <table>
    <thead>
      <tr>
        <th style="width:28%;">Test File</th>
        <th style="width:42%;">Verification Objective</th>
        <th style="width:15%; text-align:center;">Test Count</th>
        <th style="width:15%; text-align:center;">Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>test_parity.py</code></td>
        <td>Trace-freedom of bases, basis symmetry, Cardano eigenvalues, Lumley realizability bounds, PyTorch vs. TorchScript bitwise parity, and C++ standalone numerical parity</td>
        <td style="text-align:center;">9 tests</td>
        <td style="text-align:center;"><span class="badge badge-pass">PASSED</span></td>
      </tr>
      <tr>
        <td><code>test_coordinate_convention.py</code></td>
        <td>Non-square grid (15x23) gradient ordering, 2D interpolation order, and canonical DNS CSV (Nz, Ny) coordinate indexing</td>
        <td style="text-align:center;">3 tests</td>
        <td style="text-align:center;"><span class="badge badge-pass">PASSED</span></td>
      </tr>
      <tr>
        <td><code>test_mesh_sensitivity.py</code></td>
        <td>Stability of 8-vortex topology, secondary velocity bounds, and monotonic error reduction across coarse and fine grids</td>
        <td style="text-align:center;">1 test</td>
        <td style="text-align:center;"><span class="badge badge-pass">PASSED</span></td>
      </tr>
      <tr>
        <td><code>test_timescale_distribution.py</code></td>
        <td>100.0% in-distribution coverage of invariants I<sub>1</sub>-I<sub>5</sub> and Frobenius norms of tensor bases across all 20,000 CFD cells</td>
        <td style="text-align:center;">2 tests</td>
        <td style="text-align:center;"><span class="badge badge-pass">PASSED</span></td>
      </tr>
      <tr>
        <td><code>test_vortex_detection.py</code></td>
        <td>Zero-vorticity validation, synthetic 4-vortex test, and synthetic 8-vortex connected-component core localization</td>
        <td style="text-align:center;">2 tests</td>
        <td style="text-align:center;"><span class="badge badge-pass">PASSED</span></td>
      </tr>
    </tbody>
  </table>

  <h2>10. Physical Limitations & Benchmark Comparability</h2>
  <p>
    A critical physical distinction documented in the repository explains the observed streamwise velocity profile differences:
  </p>
  <ul>
    <li><strong>DNS Benchmark Domain:</strong> Idealized streamwise-periodic box (L<sub>x</sub> &approx; 6.28 D<sub>h</sub>) driven by an infinite-duration uniform body force (pressure gradient). The core flow is fully developed asymptotically.</li>
    <li><strong>OpenFOAM CFD Domain:</strong> Spatially developing duct of finite length L = 10 D<sub>h</sub> with a uniform fixed-velocity inlet, a zero-gradient velocity outlet with fixed-value pressure, and no-slip walls. While in-plane secondary vortices establish rapidly and achieve stationarity by x/D<sub>h</sub> &ge; 6.0, the centerline streamwise velocity is still developing toward the asymptotic state, accounting for the streamwise U RMSE.</li>
    <li><strong>Roadmap:</strong> Future iterations will configure cyclic periodic boundary conditions with OpenFOAM's <code>meanVelocityForce</code> fvOptions for one-to-one boundary equivalence.</li>
  </ul>

  <h2>11. Conclusion & Repository Access</h2>
  <p>
    <code>tbnnFoam</code> successfully bridges machine learning turbulence modeling and production CFD by solving key practical barriers: standalone zero-dependency C++ evaluation, closed-form Cardano realizability, and double-counting prevention. The complete open-source codebase, verification scripts, and simulation cases are hosted at:
  </p>
  <div style="text-align:center; padding: 10px; background: #edf2f7; border-radius: 4px; font-weight: bold; margin-top: 10px;">
    GitHub Repository: <a href="https://github.com/BLACKNIGHT-cloud12/tbnnFoam" style="color:#004c87;">https://github.com/BLACKNIGHT-cloud12/tbnnFoam</a>
  </div>
</div>

</body>
</html>
"""

out_dir = repo_root / 'docs' / 'report'
out_dir.mkdir(parents=True, exist_ok=True)
html_path = out_dir / 'tbnnFoam_Technical_Report.html'
pdf_path = out_dir / 'tbnnFoam_Technical_Report.pdf'

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"[+] Wrote HTML report to: {html_path}")

chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_exe):
    chrome_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

cmd = [
    chrome_exe,
    "--headless",
    "--no-sandbox",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file:///{html_path.as_posix()}"
]

proc = subprocess.run(cmd, capture_output=True, text=True)
if pdf_path.exists() and pdf_path.stat().st_size > 0:
    print(f"[+] Successfully compiled PDF Technical Report ({pdf_path.stat().st_size / 1024:.1f} KB) to:")
    print(f"    {pdf_path}")
else:
    print(f"[-] PDF compilation failed. Stderr: {proc.stderr}")
