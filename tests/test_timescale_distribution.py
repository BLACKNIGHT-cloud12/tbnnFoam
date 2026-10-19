import pytest
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.validation.evaluate_timescale_distribution import analyze_timescale_and_invariants

def test_cfd_invariants_within_training_envelope():
    repo_root = Path(__file__).resolve().parent.parent
    coverage, basis_comp = analyze_timescale_and_invariants(repo_root)
    for i in range(5):
        key = f'I_{i+1}'
        assert key in coverage
        assert coverage[key]['coverage_percent'] == 100.0
        assert coverage[key]['cfd_min'] >= coverage[key]['dns_min'] - 1e-6
        assert coverage[key]['cfd_max'] <= coverage[key]['dns_max'] + 1e-6

def test_basis_norms_bounded_by_training_distribution():
    repo_root = Path(__file__).resolve().parent.parent
    coverage, basis_comp = analyze_timescale_and_invariants(repo_root)
    for n in range(10):
        key = f'T_{n}'
        assert key in basis_comp
        assert basis_comp[key]['cfd_max'] <= basis_comp[key]['dns_max'] * 1.05

if __name__ == '__main__':
    test_cfd_invariants_within_training_envelope()
    test_basis_norms_bounded_by_training_distribution()
    print('[+] All timescale and invariant distribution tests passed!')
