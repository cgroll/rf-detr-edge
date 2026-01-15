"""Test ONNX inference for RF-DETR segmentation model."""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image


def load_image_with_exif_rotation(image_path: str) -> np.ndarray:
    """Load image and apply EXIF orientation correction.

    Returns:
        Image in BGR format (OpenCV convention) with correct orientation.
    """
    # Load with PIL to handle EXIF rotation
    pil_image = Image.open(image_path)

    # Apply EXIF orientation if present
    try:
        exif = pil_image._getexif()
        if exif:
            orientation = exif.get(274)  # 274 is the EXIF orientation tag
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass

    # Convert to numpy array (RGB)
    rgb_array = np.array(pil_image)

    # Convert RGB to BGR for OpenCV
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    else:
        bgr_array = rgb_array

    return bgr_array


def preprocess_image(image_path: str, input_size: int = 432) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load and preprocess image for inference.

    Returns:
        input_tensor: Preprocessed tensor ready for ONNX inference
        original_image: Original image in BGR format
        original_size: Original image size (height, width)
    """
    # Load image with EXIF rotation handling
    image = load_image_with_exif_rotation(image_path)
    original_size = (image.shape[0], image.shape[1])

    # Resize to input size
    resized = cv2.resize(image, (input_size, input_size))

    # Convert BGR to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    # Convert to NCHW format
    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    return input_tensor, image, original_size


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes: Array of shape (..., 4) with (cx, cy, w, h) format

    Returns:
        Array of shape (..., 4) with (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def postprocess_detections(
    boxes: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    original_size: tuple[int, int],
    input_size: int = 432,
    confidence_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-process model outputs.

    Args:
        boxes: Raw box outputs (1, N, 4) in normalized center format (cx, cy, w, h)
        labels: Raw label outputs (1, N, num_classes)
        masks: Raw mask outputs (1, N, mask_h, mask_w)
        original_size: Original image size (height, width)
        input_size: Model input size
        confidence_threshold: Minimum confidence for detections

    Returns:
        filtered_boxes: Boxes in corner format (x1, y1, x2, y2) scaled to original image size
        filtered_scores: Confidence scores
        filtered_classes: Class indices
        filtered_masks: Masks resized to original image size
    """
    # Get confidence scores (max across classes)
    scores = labels[0].max(axis=1)  # (N,)
    class_ids = labels[0].argmax(axis=1)  # (N,)

    # Filter by confidence
    mask = scores > confidence_threshold
    filtered_indices = np.where(mask)[0]

    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    filtered_scores = scores[filtered_indices]
    filtered_classes = class_ids[filtered_indices]
    filtered_boxes = boxes[0, filtered_indices]  # (K, 4) in (cx, cy, w, h) format
    filtered_masks = masks[0, filtered_indices]  # (K, mask_h, mask_w)

    # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    filtered_boxes = cxcywh_to_xyxy(filtered_boxes)

    # Scale boxes from normalized [0,1] to original image coordinates
    h, w = original_size
    filtered_boxes[:, [0, 2]] *= w  # x coordinates
    filtered_boxes[:, [1, 3]] *= h  # y coordinates

    # Resize masks to original image size
    resized_masks = []
    for m in filtered_masks:
        # Apply sigmoid to get probabilities
        m_prob = 1 / (1 + np.exp(-m))
        # Resize to original size
        m_resized = cv2.resize(m_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(m_resized)

    filtered_masks = np.array(resized_masks) if resized_masks else np.array([])

    return filtered_boxes, filtered_scores, filtered_classes, filtered_masks


def draw_mask_contours(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    mask_threshold: float = 0.5,
    colors: list[tuple[int, int, int]] | None = None,
    show_bboxes: bool = False
) -> np.ndarray:
    """Draw mask contours on the image.

    Args:
        image: Original image in BGR format
        masks: Binary masks (K, H, W)
        boxes: Bounding boxes (K, 4)
        scores: Confidence scores (K,)
        classes: Class indices (K,)
        mask_threshold: Threshold for mask binarization
        colors: List of BGR colors for each class
        show_bboxes: Whether to draw bounding boxes

    Returns:
        Annotated image
    """
    if colors is None:
        # Generate distinct colors for classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    output = image.copy()

    for i, (mask, box, score, cls) in enumerate(zip(masks, boxes, scores, classes)):
        color = colors[int(cls) % len(colors)]

        # Binarize mask
        binary_mask = (mask > mask_threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        cv2.drawContours(output, contours, -1, color, 2)

        if show_bboxes:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Class {int(cls)}: {score:.2f}"
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output


def run_inference(
    model_path: str,
    image_path: str,
    output_path: str,
    show_bboxes: bool = False,
    confidence_threshold: float = 0.5
) -> None:
    """Run inference with a single model and save results."""
    print(f"\n{'='*60}")
    print(f"Loading model from {model_path}")
    session = ort.InferenceSession(model_path)

    print(f"Loading image from {image_path}")
    input_tensor, original_image, original_size = preprocess_image(image_path)

    print("Running inference...")
    outputs = session.run(None, {"input": input_tensor})
    boxes, labels, masks = outputs

    print(f"Raw output shapes - Boxes: {boxes.shape}, Labels: {labels.shape}, Masks: {masks.shape}")

    # Post-process
    filtered_boxes, filtered_scores, filtered_classes, filtered_masks = postprocess_detections(
        boxes, labels, masks, original_size, confidence_threshold=confidence_threshold
    )

    print(f"Found {len(filtered_boxes)} detections above threshold")

    if len(filtered_boxes) > 0:
        # Draw contours
        output_image = draw_mask_contours(
            original_image, filtered_masks, filtered_boxes, filtered_scores, filtered_classes,
            show_bboxes=show_bboxes
        )

        # Save result
        cv2.imwrite(output_path, output_image)
        print(f"Saved output to {output_path}")
    else:
        print("No detections found. Try lowering the confidence threshold.")
        # Save original image with message
        output_image = original_image.copy()
        cv2.putText(output_image, "No detections", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, output_image)


def main():
    # Paths
    image_path = "assets/test_img_000000020014596465.jpg"

    # Model configurations: (model_path, output_suffix)
    models = [
        ("assets/rfdetr_seg_fp32.onnx", "fp32"),
        ("assets/rfdetr_seg_uint8.onnx", "uint8"),
    ]

    # Visualization options
    show_bboxes = False
    confidence_threshold = 0.5

    for model_path, suffix in models:
        if not Path(model_path).exists():
            print(f"\nSkipping {model_path} (not found)")
            continue

        output_path = f"assets/inference_output_{suffix}.jpg"
        run_inference(
            model_path=model_path,
            image_path=image_path,
            output_path=output_path,
            show_bboxes=show_bboxes,
            confidence_threshold=confidence_threshold
        )

    print(f"\n{'='*60}")
    print("Inference complete. Compare outputs:")
    print("  - assets/inference_output_fp32.jpg  (full precision)")
    print("  - assets/inference_output_uint8.jpg (quantized)")


if __name__ == "__main__":
    main()
