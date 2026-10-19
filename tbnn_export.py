import torch
import torch.nn as nn
import os

class TensorBasisNeuralNetwork(nn.Module):

    def __init__(self, num_invariants=5, num_bases=10, hidden_dim=64):
        super(TensorBasisNeuralNetwork, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_invariants, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, num_bases))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, invariants: torch.Tensor, tensor_bases: torch.Tensor) -> torch.Tensor:
        invariants = torch.clamp(invariants, -50.0, 50.0)
        g = self.mlp(invariants)
        g_expanded = g.unsqueeze(-1).unsqueeze(-1)
        anisotropy = torch.sum(g_expanded * tensor_bases, dim=1)
        return anisotropy

def export_model(output_path='tbnn_model.pt'):
    print('--> Initializing Tensor Basis Neural Network (TBNN)...')
    model = TensorBasisNeuralNetwork(num_invariants=5, num_bases=10, hidden_dim=64)
    model.eval()
    batch_size = 1
    dummy_invariants = torch.zeros((batch_size, 5), dtype=torch.float32)
    dummy_bases = torch.zeros((batch_size, 10, 3, 3), dtype=torch.float32)
    dummy_bases[:, 0, 0, 0] = 1.0
    dummy_bases[:, 0, 1, 1] = -1.0
    print(f'--> Tracing model via TorchScript...')
    traced_model = torch.jit.trace(model, (dummy_invariants, dummy_bases))
    traced_model.save(output_path)
    print(f'--> Successfully exported TorchScript model to: {os.path.abspath(output_path)}')
    print(f'--> File size: {os.path.getsize(output_path) / 1024:.2f} KB')
if __name__ == '__main__':
    export_model('tbnn_model.pt')
