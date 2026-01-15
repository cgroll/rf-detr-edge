"""Export RF-DETR segmentation model to ONNX format."""

import shutil
from pathlib import Path

from rfdetr import RFDETRSegPreview

if __name__ == "__main__":
    # Load the trained model
    model = RFDETRSegPreview(pretrain_weights="assets/checkpoint_best_regular.pth", resolution=432)

    # Export to ONNX without simplification to preserve mask output
    model.export(output_dir="assets", simplify=False)

    # Rename to indicate fp32 precision
    default_path = Path("assets/inference_model.onnx")
    renamed_path = Path("assets/rfdetr_seg_fp32.onnx")
    if default_path.exists():
        shutil.move(default_path, renamed_path)
        print(f"Model exported to {renamed_path}")
