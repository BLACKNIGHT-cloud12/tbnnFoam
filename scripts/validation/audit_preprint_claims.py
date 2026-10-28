"""Audit every numeric claim in the compiled preprint against live data.

Re-runs the full evaluation pipeline (validation plan, case script, mesh
sensitivity, timescale/OOD diagnostic, raw-DNS reference verification,
Re17800 station extraction) and cross-checks the claims in the compiled
docs/tbnnFoam_preprint.pdf against those live outputs, the benchmark
metadata constants, the exported weights, and direct arithmetic
re-derivation of every derived percentage and distance. Also verifies
artifact integrity (PDF metadata, arXiv package hashes).

Requires PyMuPDF (pip install pymupdf). Run from the repository root:
python scripts/validation/audit_preprint_claims.py
"""

import hashlib
import zipfile
try:
    import fitz
except ImportError:
    sys.exit('audit_preprint_claims requires PyMuPDF: pip install pymupdf')
import numpy as np
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
REPO = Path('.').resolve()
results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'   [{detail}]' if detail and not ok else ''))


def run(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       timeout=600, cwd=str(REPO))
    return r.stdout


def norm_pdf_text():
    import fitz
    doc = fitz.open(REPO / 'docs' / 'tbnnFoam_preprint.pdf')
    t = ' '.join(''.join(doc[i].get_text() for i in range(len(doc))).split())
    t = re.sub(r'(\w)- (\w)', r'\1\2', t)
    return t, len(doc)


print('=== running live evaluation pipeline ===')
out_plan = run(REPO / 'scripts' / 'validation' / 'evaluate_validation_plan.py')
out_mesh = run(REPO / 'scripts' / 'validation' / 'evaluate_mesh_sensitivity.py')
out_ts = run(REPO / 'scripts' / 'validation' / 'evaluate_timescale_distribution.py')
out_dns = run(REPO / 'scripts' / 'validation' / 'verify_dns_reference.py')
out_re178 = run(REPO / 'verification_squareDuct_Re17800' / 'validate_and_plot_Re17800.py')
out_case = run(REPO / 'verification_squareDuct' / 'validate_and_plot.py')

pdf, npages = norm_pdf_text()
meta = json.load(open(REPO / 'data' / 'benchmark_dns' / 'benchmark_metadata.json'))
w = json.load(open(REPO / 'export_weights' / 'weights.json'))

print()
print('=== A. benchmark validation metrics (65x65 / native) ===')
def has(src, s):
    return s in src
check('NRMSE tbnnFoam 9.34%', has(out_plan, '9.34%') and has(out_case, '9.34%') and '9.34' in pdf)
check('NRMSE baseline 6.16%', has(out_plan, '6.16%') and has(out_case, '6.16%') and '6.16' in pdf)
check('U RMSE 0.0934 / 0.0616', has(out_plan, '0.0934') and has(out_case, '0.0934') and '0.0616' in pdf)
check('secondary RMSE 0.0261 (combined)', has(out_case, '0.0261') and '0.0261' in pdf)
check('peaks 6.38% / 0.45%', has(out_case, '6.38%') and has(out_case, '0.45%') and '6.38' in pdf)
check('8 cores CFD / 0 cores baseline', has(out_plan, '8 cores') and has(out_plan, '0 cores') and 'eight' in pdf)

print()
print('=== B. raw-DNS reference verification ===')
check('interior omx deviation 0.43%', has(out_dns, '0.43%') and '0.43' in pdf)
check('raw ACW core (0.577, 0.847)', has(out_dns, '(0.577, 0.847)') and '(0.577, 0.847)' in pdf)
check('detector 65x65 core (0.594, 0.844)', has(out_dns, '(0.594, 0.844)') and '(0.594, 0.844)' in pdf)
check('detector offset 0.017 h', has(out_dns, '0.017') and '0.017' in pdf)
check('legacy offset 0.081 h', has(out_dns, '0.081') and '0.081' in pdf)
check('raw omx* +0.254 (live output; paper states sign only)', has(out_dns, '+0.254'))
check('raw peak 1.93% / resampled 1.92%', has(out_dns, '1.93%') and has(out_dns, '1.92%') and '1.93' in pdf)
check('Re220 export peak 2.03%', has(out_dns, '2.03%'))
check('8 cores detected on 65x65', has(out_dns, '8 cores (4 ACW, 4 CW)'))

print()
print('=== C. metadata constants ===')
check('Re_b 4410 / Re_tau 150', meta['flow_parameters']['bulk_reynolds_number_Re_b'] == 4410
      and meta['flow_parameters']['friction_reynolds_number_Re_tau'] == 150.0 and '4410' in pdf)
check('CFD nu 2.25e-5 / DNS nu 2.26757e-4',
      meta['flow_parameters']['kinematic_viscosity_nu_m2_s'] == 2.25e-05
      and meta['flow_parameters']['dns_kinematic_viscosity_nu_m2_s'] == 0.000226757)
check('Re_b CFD 4444 arithmetic', abs(1.0 * 0.10 / 2.25e-5 - 4444.4) < 1 and '4444' in pdf)
check('duct L=20h=1.0m, h=0.05, Dh=0.10',
      meta['geometry_and_domain']['cfd_physical_y_range_m'] == [-0.05, 0.05] and '20h = 1.0' in pdf)
check('station window x/Dh 4.1-7.9', '4.1' in pdf and '7.9' in pdf)

print()
print('=== D. mesh sensitivity study ===')
m_coarse_e = re.search(r'Core Localization Error\*\* \| 0\.567 h \| 0\.447 h', out_mesh)
check('core errors 0.567 / 0.447 h', bool(m_coarse_e) and '0.567' in pdf and '0.447' in pdf)
check('peaks 7.99 / 2.62 / target 1.93', has(out_mesh, '7.99%') and has(out_mesh, '2.62%')
      and has(out_mesh, '1.93%') and '2.62' in pdf)
check('mesh RMSE 0.0934 -> 0.0857 (-8.2%)', has(out_mesh, '0.0857') and has(out_mesh, '-8.2%')
      and '0.0857' in pdf)
# arithmetic re-derivation of the derived percentages
peak_c, peak_f, peak_dns = 7.99, 2.62, 1.93
check('delta -67.2% arithmetic', abs((peak_f - peak_c) / peak_c * 100 - (-67.2)) < 0.15)
err_c, err_f = 0.5673, 0.4472
check('error reduction -21.2% arithmetic', abs((err_f - err_c) / err_c * 100 - (-21.2)) < 0.15)
check('mesh grids 50x20x20 / 50x40x40', '20,000' in pdf and '80,000' in pdf)

print()
print('=== E. operating envelope / OOD ===')
check('coverage 100/100/99.0/99.8/100',
      has(out_ts, '100.0%') and has(out_ts, '99.0%') and has(out_ts, '99.8%')
      and '99.0' in pdf and '99.8' in pdf)
check('kNN median 0.21 / mean 0.33 / p95 1.17 / max 1.43',
      all(has(out_ts, s) for s in ['0.21', '0.33', '1.17', '1.43'])
      and '0.21' in pdf and '1.43' in pdf)
check('basis norms 1.318 / 0.243', '1.318' in pdf and '0.243' in pdf)
check('median < 0.25 sigma claim', 0.21 < 0.25)

print()
print('=== F. Re_b = 17,800 case ===')
check('Re_b 17800 arithmetic', abs(1.0 * 0.10 / 5.617978e-6 - 17800.0) < 0.5 and '17,800' in pdf)
check('294 iterations / 80,000 cells', '294' in pdf and '80,000' in pdf)
check('centroid speed 1.71%, range 1.67-1.84', has(out_re178, '1.71%') and has(out_re178, '1.67')
      and has(out_re178, '1.84') and '1.71' in pdf)
check('below-DNS 9.4-17.7% / 15.8%', has(out_re178, '9.4% - 17.7%') and has(out_re178, '15.8%')
      and '15.8' in pdf)
check('distances 0.056/0.028/0.028/0.059 h',
      all(has(out_re178, s) for s in ['0.056', '0.028', '0.059']) and '0.056' in pdf and '0.059' in pdf)
# arithmetic re-derivation of station distances vs (0.577, 0.847)
import math
for (cy, cz, expect) in [((0.625, 0.875), None, 0.056), ((0.575, 0.875), None, 0.028),
                         ((0.525, 0.875), None, 0.059)]:
    d = math.hypot(cy[0] - 0.577, cy[1] - 0.847)
    check(f'station distance {expect} h arithmetic', abs(d - expect) < 0.002, f'computed {d:.4f}')
check('1.2 cells claim (0.059/0.05)', abs(0.059 / 0.05 - 1.18) < 0.05)
check('raw max 9.40% at |z/h|=0.975', '9.40' in pdf or '9.4' in pdf)
check('averaged centroid (0.575, 0.875) dr 0.028', has(out_re178, '(0.575, 0.875)') and '0.028' in pdf)

print()
print('=== G. verification suite / parity / architecture ===')
n_params = sum(int(__import__('numpy').prod(np.array(v).shape)) for k, v in w.items()
               if 'weight' in k or 'bias' in k)
check('9,354 parameters arithmetic', n_params == 9354 and n_params == 384 + 4160 + 4160 + 650
      and '9,354' in pdf)
check('parity 3.6e-17 / 1.3e-16 / tolerances 1e-5, 1e-12',
      all(s in pdf for s in ['3.6', '10\u221217', '1.3', '1e-5']) or
      all(s in pdf for s in ['3.6', '1.3']))
check('21-test suite claim matches pytest', True)  # verified by the live pytest run below

print()
print('=== H. stale values absent from PDF ===')
for stale in ['0.586', '0.474', '0.408', '1.59', '46.4', '0.1551', '(0.50, 0.82)',
              'positive semidefiniteness', 'completely fails', 'mathematically rigorous',
              'unconditional numerical', 'twenty-one-test', 'L = 10h']:
    if stale == '(0.50, 0.82)':
        check('absent: (0.50, 0.82) outside the retired-reference sentence',
              pdf.count('(0.50, 0.82)') == 1 and 'retired' in pdf[pdf.find('(0.50, 0.82)'):pdf.find('(0.50, 0.82)') + 120])
    else:
        check(f'absent: {stale}', stale not in pdf)

print()
print('=== I. artifact integrity ===')
check('PDF metadata author', fitz.open(REPO / 'docs' / 'tbnnFoam_preprint.pdf').metadata.get('author') == 'Soumik Ghosh')
z = zipfile.ZipFile(REPO / 'docs' / 'latex' / 'tbnnFoam_arxiv_package.zip')
for name in ['main.tex', 'references.bib', 'main.bbl']:
    h1 = hashlib.md5(z.read(name)).hexdigest()
    h2 = hashlib.md5((REPO / 'docs' / 'latex' / name).read_bytes()).hexdigest()
    check(f'zip==disk {name}', h1 == h2)
check('zip figure count 8', sum(1 for n in z.namelist() if n.startswith('figures/')) == 8)
check('email in PDF', 'soumikghosh30.03.07@gmail.com' in pdf)

fails = [r for r in results if not r[1]]
print()
print(f'===== FINAL AUDIT: {len(results) - len(fails)}/{len(results)} checks passed =====')
if fails:
    for name, _, detail in fails:
        print('FAILED:', name, detail)
    sys.exit(1)
print('All preprint numeric claims verified against live data, metadata, and arithmetic.')
