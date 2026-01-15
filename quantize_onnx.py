"""Quantize RF-DETR ONNX model to uint8."""

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path


def quantize_model(input_path: str, output_path: str) -> None:
    """Quantize an ONNX model to uint8."""
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    input_model = Path("assets/rfdetr_seg_fp32.onnx")
    output_model = Path("assets/rfdetr_seg_uint8.onnx")

    if not input_model.exists():
        raise FileNotFoundError(
            f"{input_model} not found. Run export_onnx.py first."
        )

    quantize_model(str(input_model), str(output_model))
