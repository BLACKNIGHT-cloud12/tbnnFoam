import pytest
from pathlib import Path
import numpy as np
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.validation.evaluate_timescale_distribution import analyze_timescale_and_invariants

def _check_dns_available():
    repo_root = Path(__file__).resolve().parent.parent
    dns_env = os.environ.get('TBNN_DNS_DIR')
    if dns_env and (Path(dns_env) / 'square_duct_dns_Reb4410_Retau150.csv').exists():
        return True
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    return dns_file.exists()

def test_cfd_invariants_within_training_envelope():
    if not _check_dns_available():
        pytest.skip(
            "Benchmark DNS dataset not found in data/benchmark_dns/. "
            "To run integration test against DNS, obtain reference data per data/benchmark_dns/README.md "
            "or set TBNN_DNS_DIR."
        )
    repo_root = Path(__file__).resolve().parent.parent
    coverage, basis_comp = analyze_timescale_and_invariants(repo_root)
    for i in range(5):
        key = f'I_{i+1}'
        assert key in coverage
        assert coverage[key]['coverage_percent'] >= 98.0
        assert coverage[key]['cfd_min'] >= coverage[key]['dns_min'] - 1e-4
        assert coverage[key]['cfd_max'] <= coverage[key]['dns_max'] + 1e-3

def test_basis_norms_bounded_by_training_distribution():
    if not _check_dns_available():
        pytest.skip(
            "Benchmark DNS dataset not found in data/benchmark_dns/. "
            "To run integration test against DNS, obtain reference data per data/benchmark_dns/README.md "
            "or set TBNN_DNS_DIR."
        )
    repo_root = Path(__file__).resolve().parent.parent
    coverage, basis_comp = analyze_timescale_and_invariants(repo_root)
    for n in range(10):
        key = f'T_{n}'
        assert key in basis_comp
        assert basis_comp[key]['cfd_max'] <= basis_comp[key]['dns_max'] * 1.05

if __name__ == '__main__':
    if _check_dns_available():
        test_cfd_invariants_within_training_envelope()
        test_basis_norms_bounded_by_training_distribution()
        print('[+] All timescale and invariant distribution tests passed!')
    else:
        print('[SKIP] Benchmark DNS data not present; skipping timescale distribution test.')
