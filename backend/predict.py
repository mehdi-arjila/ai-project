"""
predict.py - Inference Engine for Handwriting Recognition
Handles image preprocessing, model inference, and returns structured predictions.
"""

import os
import sys
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import CLASSES, NUM_CLASSES, load_or_create_model

# Singleton model reference (loaded once, reused across requests)
_model = None


def get_model():
    """Load model once and cache it globally."""
    global _model
    if _model is None:
        _model = load_or_create_model("model/saved_model.h5")
    return _model


def reload_model():
    """Force reload the model from disk (called after retraining)."""
    global _model
    _model = None
    _model = load_or_create_model("model/saved_model.h5")
    return _model


# ─── Image Preprocessing ────────────────────────────────────────────────────

def preprocess_image(image_data):
    """
    Preprocess raw image data into a 28x28 normalized tensor.

    Steps:
      1. Decode base64 PNG from canvas
      2. Convert to grayscale
      3. Invert colors (canvas is white-on-black → model expects black-on-white... actually inverted)
      4. Crop to bounding box (center the drawing)
      5. Resize to 28x28
      6. Normalize pixels to [0, 1]

    Args:
        image_data: base64 encoded PNG string (with or without data URI prefix)

    Returns:
        np.ndarray: shape (1, 28, 28, 1), float32, values in [0, 1]
    """
    # Strip data URI prefix if present
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    # Decode base64 → PIL Image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")

    # Create white background and paste image onto it
    background = Image.new("L", img.size, 0)
    # Extract the alpha channel and use it as mask
    r, g, b, a = img.split()
    # Convert to grayscale: white drawing on black background
    gray = Image.fromarray(np.array(r).astype(np.uint8))
    background.paste(gray, mask=a)
    img = background

    # Convert to numpy for processing
    arr = np.array(img, dtype=np.float32)

    # Bounding box crop: find non-zero pixels and crop tightly
    rows = np.any(arr > 20, axis=1)
    cols = np.any(arr > 20, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding around the bounding box
        pad = 4
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1] - 1, cmax + pad)

        arr = arr[rmin:rmax+1, cmin:cmax+1]

    # Resize to 28x28 using LANCZOS for quality
    pil_cropped = Image.fromarray(arr.astype(np.uint8))
    pil_resized = pil_cropped.resize((28, 28), Image.LANCZOS)

    # Normalize to [0, 1]
    final = np.array(pil_resized, dtype=np.float32) / 255.0

    # Reshape to (1, 28, 28, 1) for model input
    return final.reshape(1, 28, 28, 1)


def preprocess_from_array(arr):
    """
    Preprocess a numpy array (H, W) directly to model input.
    Used for retraining with corrected samples.

    Args:
        arr: numpy array (H, W) uint8 or float32

    Returns:
        np.ndarray: shape (1, 28, 28, 1), float32
    """
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    result = np.array(img, dtype=np.float32) / 255.0
    return result.reshape(1, 28, 28, 1)


# ─── Prediction ─────────────────────────────────────────────────────────────

def predict(image_data):
    """
    Run inference on a base64-encoded canvas image.

    Args:
        image_data: base64 PNG string from frontend canvas

    Returns:
        dict: {
            "predicted":    str   - top predicted character
            "confidence":   float - confidence score (0.0–1.0)
            "top3": [
                {"label": str, "confidence": float},
                ...
            ]
            "all_scores": [float] * 36  - full softmax output for visualization
        }
    """
    model = get_model()

    # Preprocess the image
    processed = preprocess_image(image_data)

    # Run inference
    probs = model.predict(processed, verbose=0)[0]  # shape: (36,)

    # Top-1 prediction
    top1_idx  = int(np.argmax(probs))
    top1_char = CLASSES[top1_idx]
    top1_conf = float(probs[top1_idx])

    # Top-3 predictions
    top3_idx  = np.argsort(probs)[::-1][:3]
    top3 = [
        {"label": CLASSES[i], "confidence": float(probs[i])}
        for i in top3_idx
    ]

    return {
        "predicted":  top1_char,
        "confidence": top1_conf,
        "top3":       top3,
        "all_scores": [float(p) for p in probs],  # for 3D visualization
    }


def get_layer_activations(image_data):
    """
    Extract intermediate layer activations for 3D visualization.
    Returns activation stats for each CNN layer to drive the Three.js animation.

    Args:
        image_data: base64 PNG string

    Returns:
        dict: layer name → activation statistics
    """
    import tensorflow as tf

    model = get_model()
    processed = preprocess_image(image_data)

    # Build activation model for key layers
    layer_names = ["conv1", "pool1", "conv2", "pool2", "dense1", "output_layer"]
    available = [l.name for l in model.layers]

    activation_outputs = []
    activation_layer_names = []

    for name in layer_names:
        if name in available:
            activation_outputs.append(model.get_layer(name).output)
            activation_layer_names.append(name)

    activation_model = tf.keras.Model(inputs=model.input, outputs=activation_outputs)
    activations = activation_model.predict(processed, verbose=0)

    result = {}
    for name, act in zip(activation_layer_names, activations):
        act_flat = act.flatten()
        result[name] = {
            "mean":  float(np.mean(act_flat)),
            "max":   float(np.max(act_flat)),
            "min":   float(np.min(act_flat)),
            # Sample 64 values for visualization (normalized 0-1)
            "sample": [float(v) for v in np.interp(
                act_flat[np.random.choice(len(act_flat), min(64, len(act_flat)), replace=False)],
                [act_flat.min(), act_flat.max() + 1e-8],
                [0, 1]
            )]
        }

    return result


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tensorflow as tf
    print("Running prediction test with random input...")
    # Create a dummy white image (simulating a drawn digit)
    dummy = np.zeros((200, 200), dtype=np.uint8)
    # Draw a simple vertical line (like '1')
    dummy[40:160, 95:105] = 255

    # Save as temp PNG and encode to base64
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(dummy).save(tmp.name)
    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    result = predict(b64)
    print(f"Predicted: {result['predicted']} ({result['confidence']*100:.1f}%)")
    print(f"Top 3: {result['top3']}")
