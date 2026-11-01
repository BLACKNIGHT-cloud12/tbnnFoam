# Python Training & Experimental Pipelines

This directory contains experimental and fallback training implementations:

* **`train_tbnn.py`**:
  - **Status:** Legacy 30-neuron prototype experiment.
  - Trains an earlier 30-hidden-unit network writing to `legacy_exports/`.
* **`train_numpy_fallback.py`**:
  - **Status:** Dependency-free scikit-learn fallback experiment.
  - Trains an `MLPRegressor` using pseudo-inverse tensor basis projection without requiring PyTorch.

## Authoritative Shipped Training Pipeline
The authoritative, reproducible training pipeline for the shipped 64-width production model (9,354 parameters) is located in the repository root:
* **`tbnn_train.py`**: Trains the production model on canonical DNS data (`data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv`) with fixed random seeds (42).
* **`tbnn_export.py`**: Exports weights to TorchScript (`tbnn_model.pt`) and JSON (`export_weights/weights.json`).
* **`tbnn_tensor_math.py`**: Canonical implementation of Pope's 10-tensor integrity basis, Cayley-Hamilton invariants, and Cardano cubic realizability limiter.
