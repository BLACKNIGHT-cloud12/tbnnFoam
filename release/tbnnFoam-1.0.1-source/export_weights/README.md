# Exported Model Weights (C++ Standalone Evaluator)

This directory contains exported model weight artifacts consumed by the zero-dependency C++ standalone evaluator (`src/tbnnEvaluator.H`).

## Active Production Weights
* **`weights.json`**:
  - **Architecture:** 5 invariants -> [64, 64, 64] -> 10 basis expansion coefficients g^(1...10).
  - **Parameters:** 9,354 trainable parameters (layer counts: 384 + 4160 + 4160 + 650 = 9,354).
  - **Precision:** Single-precision float arrays.
  - **Normalization:** `inv_mean` = [0, 0, 0, 0, 0], `inv_std` = [1, 1, 1, 1, 1] (the production model trains directly on physical invariants).
  - **Parity:** Numerically identical (max parameter deviation = 0.0) to the TorchScript model `tbnn_model.pt` and PyTorch state dict `tbnn_weights.pth`.

## Legacy Artifacts
* **`tbnn_model.pt`** (in this directory):
  - **Note:** Historical 30-neuron prototype model (2,350 parameters). Retained strictly for provenance and historical reference.
  - **Active Model:** The authoritative 64-neuron production model is located at the repository root (`tbnn_model.pt`) and in the verification cases (`verification_squareDuct/constant/tbnn_model.pt`).
