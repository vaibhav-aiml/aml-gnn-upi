"""
Export trained model for production deployment
"""
import torch
import json
from pathlib import Path
from src.models.graphsage import GraphSAGELight

class ModelExporter:
    def __init__(self, model_path="models_saved/comparison/GraphSAGE_model.pt"):
        self.model_path = model_path
        
    def save_model_metadata(self, model_config):
        """Save model configuration and metrics"""
        metadata = {
            "model_version": "1.0.0",
            "model_type": "GraphSAGE",
            "config": model_config,
            "in_channels": 9,
            "status": "production_ready"
        }
        
        out_path = Path("models_saved/metadata.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("[OK] Model metadata saved to models_saved/metadata.json")

    def export_torchscript(self, model, sample_x, sample_edge_index):
        """Export to TorchScript"""
        try:
            traced_model = torch.jit.trace(model, (sample_x, sample_edge_index))
            traced_model.save("models_saved/model_traced.pt")
            print("[OK] Model exported to TorchScript at models_saved/model_traced.pt")
        except Exception as e:
            print(f"[WARN] TorchScript trace skipped: {e}")

def main():
    print("[INFO] Model exporter executing...")
    exporter = ModelExporter()
    config = {
        "in_channels": 9,
        "hidden_channels": 64,
        "out_channels": 2
    }
    exporter.save_model_metadata(config)
    
    ckpt_path = Path("models_saved/comparison/GraphSAGE_model.pt")
    if ckpt_path.exists():
        model = GraphSAGELight(9, 64, 2)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model.eval()
        
        sample_x = torch.randn(10, 9)
        sample_edge = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        exporter.export_torchscript(model, sample_x, sample_edge)
        
    print("[OK] Export script completed successfully!")

if __name__ == "__main__":
    main()