#!/usr/bin/env python3
"""Export PyTorch model to ONNX for fast inference."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b0
from lib.config import MODEL_DIR


def create_model(num_classes=2, dropout=0.4):
    """Create EfficientNet-B0 for multi-label classification."""
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def export_to_onnx(input_path: Path, output_path: Path) -> bool:
    """Convert PyTorch model to ONNX format."""
    print(f"Loading model from {input_path}...")

    checkpoint = torch.load(input_path, map_location='cpu')
    model = create_model(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch'},
            'logits': {0: 'batch'}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify
    print("Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print("Comparing PyTorch vs ONNX outputs...")
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'image': dummy_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max output difference: {max_diff:.6f}")

    if max_diff > 1e-4:
        print("Warning: Output difference exceeds tolerance!")
    else:
        print("Verification passed!")

    model_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel: EfficientNet-B0")
    print(f"Size: {model_size:.1f} MB")
    print(f"Input: (batch, 3, 224, 224)")
    print(f"Output: (batch, 2) - [biopsy, magview] logits")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--input", type=Path, default=MODEL_DIR / "model.pth")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "model.onnx")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Model not found: {args.input}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(args.input, args.output)
    print(f"\nExport complete!")


if __name__ == "__main__":
    main()
