import torch
import torch.nn as nn
import os

class TensorBasisNeuralNetwork(nn.Module):
    """
    Tensor Basis Neural Network (TBNN) following Ling, Kurzawski, and Templeton (2016).
    Maps 5 scalar invariants to 10 scalar coefficients g^(1..10) for Pope's basis expansion.
    """
    def __init__(self, num_invariants=5, num_bases=10, hidden_dim=64):
        super(TensorBasisNeuralNetwork, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_invariants, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, num_bases)
        )
        
        # Initialize output layer to zero (residual perturbation initialization)
        # Guarantees the solver starts from a physically stable baseline and adds Reynolds stress corrections smoothly
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, invariants: torch.Tensor, tensor_bases: torch.Tensor) -> torch.Tensor:
        # Clamp invariants to prevent numerical overflow in C++
        invariants = torch.clamp(invariants, -50.0, 50.0)
        
        # g coefficients shape: (Batch, 10)
        g = self.mlp(invariants)
        
        # Expand g for broadcasting across the 3x3 tensor components: (Batch, 10, 1, 1)
        g_expanded = g.unsqueeze(-1).unsqueeze(-1)
        
        # a_ij = sum_{n=1}^{10} g^(n) * T_ij^(n)
        anisotropy = torch.sum(g_expanded * tensor_bases, dim=1)
        
        return anisotropy

def export_torchscript_model(output_path="tbnn_model.pt"):
    print("--> Initializing Tensor Basis Neural Network (TBNN)...")
    model = TensorBasisNeuralNetwork(num_invariants=5, num_bases=10, hidden_dim=64)
    model.eval()
    
    batch_size = 1
    dummy_invariants = torch.zeros((batch_size, 5), dtype=torch.float32)
    dummy_bases = torch.zeros((batch_size, 10, 3, 3), dtype=torch.float32)
    dummy_bases[:, 0, 0, 0] = 1.0
    dummy_bases[:, 0, 1, 1] = -1.0
    
    print(f"--> Tracing model via TorchScript...")
    traced_model = torch.jit.trace(model, (dummy_invariants, dummy_bases))
    
    traced_model.save(output_path)
    print(f"--> Successfully exported TorchScript model to: {os.path.abspath(output_path)}")
    print(f"--> File size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    export_torchscript_model("tbnn_model.pt")
