import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.validation.evaluate_mesh_sensitivity import run_mesh_sensitivity_study

def test_mesh_sensitivity_core_and_velocity():
    repo_root = Path(__file__).resolve().parent.parent
    results = run_mesh_sensitivity_study(repo_root)

    assert results['coarse']['cores_detected'] == 8
    assert results['fine']['cores_detected'] == 8

    assert results['fine']['core_dist_err'] <= 0.50
    assert results['fine']['core_dist_err'] < results['coarse']['core_dist_err']

    assert 0.010 <= results['coarse']['vmax_ratio'] <= 0.100
    assert 0.015 <= results['fine']['vmax_ratio'] <= 0.040

    assert results['fine']['rmse_u'] < results['coarse']['rmse_u']

if __name__ == '__main__':
    test_mesh_sensitivity_core_and_velocity()
    print('[+] All mesh resolution sensitivity tests passed!')
