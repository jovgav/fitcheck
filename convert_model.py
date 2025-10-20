import torch
import torch.onnx
from model import CoolUncoolCNN
import os

def convert_pytorch_to_tfjs():
    """Convert PyTorch model to TensorFlow.js format"""
    
    # Load your trained model
    model_path = "cool_uncool_model.pth"
    if not os.path.exists(model_path):
        print("No trained model found. Please train a model first.")
        return False
    
    # Create model instance
    model = CoolUncoolCNN(num_classes=2)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = "cool_uncool_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {onnx_path}")
    print("Next steps:")
    print("1. Install tensorflowjs: pip install tensorflowjs")
    print("2. Convert ONNX to TensorFlow.js:")
    print(f"   tensorflowjs_converter --input_format=onnx {onnx_path} ./tfjs_model")
    
    return True

if __name__ == "__main__":
    convert_pytorch_to_tfjs()
