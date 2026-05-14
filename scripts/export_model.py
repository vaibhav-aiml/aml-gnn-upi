"""
Export trained model for production deployment
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.graphsage import GraphSAGELight
import json

class ModelExporter:
    def __init__(self, model_path="models_saved/graphsage_model.pt"):
        self.model_path = model_path
        
    def export_to_onnx(self, model, sample_input):
        """Export model to ONNX format"""
        torch.onnx.export(
            model,
            sample_input,
            "models_saved/model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        print("✅ Model exported to ONNX format")
        
    def export_to_torchscript(self, model, sample_input):
        """Export to TorchScript"""
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("models_saved/model_traced.pt")
        print("✅ Model exported to TorchScript")
        
    def save_model_metadata(self, model_config):
        """Save model configuration and metrics"""
        metadata = {
            "model_version": "1.0.0",
            "model_type": "GraphSAGE",
            "config": model_config,
            "export_date": str(Path(".").stat().st_ctime)
        }
        
        with open("models_saved/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✅ Model metadata saved")
        
    def export_all_formats(self, model, sample_input, config):
        """Export model to all formats"""
        print("Exporting model for production...")
        self.export_to_onnx(model, sample_input)
        self.export_to_torchscript(model, sample_input)
        self.save_model_metadata(config)
        print("✅ Export complete!")

if __name__ == "__main__":
    print("Model exporter ready")