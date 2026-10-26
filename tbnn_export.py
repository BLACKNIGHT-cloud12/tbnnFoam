import os
from pathlib import Path
import json
import torch

# The shipped architecture is defined once in tbnn_train; the exporter must
# never define its own copy, or exported weights can silently diverge from
# the audited model.
from tbnn_train import TensorBasisNeuralNetwork

def export_cpp_weights(model, repo_root=None):
    """Export model weights for the standalone C++ evaluator.

    Writes the plain-JSON weights consumed by src/tbnnEvaluator.H to
    export_weights/weights.json and mirrors them into
    verification_squareDuct/constant/tbnn/weights.json. The shipped model
    is trained on unnormalized invariants, so inv_mean = 0, inv_std = 1.
    """
    if repo_root is None:
        repo_root = Path('.')
    export_dir = Path(repo_root) / 'export_weights'
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {
        'inv_mean': [0.0, 0.0, 0.0, 0.0, 0.0],
        'inv_std': [1.0, 1.0, 1.0, 1.0, 1.0],
        '0.weight': model.mlp[0].weight.detach().cpu().numpy().tolist(),
        '0.bias': model.mlp[0].bias.detach().cpu().numpy().tolist(),
        '2.weight': model.mlp[2].weight.detach().cpu().numpy().tolist(),
        '2.bias': model.mlp[2].bias.detach().cpu().numpy().tolist(),
        '4.weight': model.mlp[4].weight.detach().cpu().numpy().tolist(),
        '4.bias': model.mlp[4].bias.detach().cpu().numpy().tolist(),
        '6.weight': model.mlp[6].weight.detach().cpu().numpy().tolist(),
        '6.bias': model.mlp[6].bias.detach().cpu().numpy().tolist()
    }
    json_path = export_dir / 'weights.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f'--> Exported C++ standalone weights to: {json_path.resolve()}')
    case_json_dir = Path(repo_root) / 'verification_squareDuct' / 'constant' / 'tbnn'
    case_json_dir.mkdir(parents=True, exist_ok=True)
    with open(case_json_dir / 'weights.json', 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)

def export_model(weights_path='tbnn_weights.pth', output_path='tbnn_model.pt'):
    model = TensorBasisNeuralNetwork(num_invariants=5, num_bases=10, hidden_dim=64)
    if not os.path.exists(weights_path):
        from tbnn_train import train
        train()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    dummy_invariants = torch.zeros((1, 5), dtype=torch.float32)
    dummy_bases = torch.zeros((1, 10, 3, 3), dtype=torch.float32)
    dummy_bases[:, 0, 0, 0] = 1.0
    dummy_bases[:, 0, 1, 1] = -1.0
    traced_model = torch.jit.trace(model, (dummy_invariants, dummy_bases))
    traced_model.save(output_path)
    case_pt = Path('verification_squareDuct/constant/tbnn_model.pt')
    traced_model.save(str(case_pt))
    print(f'--> Successfully exported trained TorchScript model to: {os.path.abspath(output_path)}')
    print(f'--> Exported case model to: {case_pt.resolve()}')
    print(f'--> Model size: {os.path.getsize(output_path) / 1024:.2f} KB')
    export_cpp_weights(model)

if __name__ == '__main__':
    export_model()
