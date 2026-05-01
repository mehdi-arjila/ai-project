"""
model.py - CNN Architecture for Handwritten Digit & Letter Recognition
Supports 36 classes: digits 0-9 and letters A-Z
Uses TensorFlow/Keras with a 4-layer CNN architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


# ─── Class Label Mapping ────────────────────────────────────────────────────
# 36 classes: 0-9 digits + A-Z uppercase letters
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUM_CLASSES = len(CLASSES)  # 36


def build_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    """
    Build the CNN model with 4 main layers as per specification:
      1. Input Layer       - 28x28 grayscale
      2. Conv Block 1      - Conv2D(32, 3x3, ReLU) + MaxPooling
      3. Conv Block 2      - Conv2D(64, 3x3, ReLU) + MaxPooling
      4. Dense Layers      - Flatten + Dense(128, ReLU) + Softmax(36)
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential(name="HandwritingCNN")

    # ── Layer 1: Input + First Conv Block ───────────────────────────────────
    model.add(layers.Input(shape=input_shape, name="input_layer"))

    # Batch normalization for better training stability
    model.add(layers.BatchNormalization(name="bn_input"))

    # First Convolutional Layer: 32 filters, 3×3 kernel, ReLU activation
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name="conv1"
    ))

    # Batch norm after conv
    model.add(layers.BatchNormalization(name="bn_conv1"))

    # MaxPooling: reduces spatial dimensions by 2x
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # Dropout for regularization
    model.add(layers.Dropout(0.25, name="dropout1"))

    # ── Layer 2: Second Conv Block ───────────────────────────────────────────
    # Second Convolutional Layer: 64 filters, 3×3 kernel, ReLU activation
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name="conv2"
    ))

    model.add(layers.BatchNormalization(name="bn_conv2"))

    # Second MaxPooling
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))

    model.add(layers.Dropout(0.25, name="dropout2"))

    # ── Layer 3 & 4: Dense Layers ────────────────────────────────────────────
    # Flatten feature maps to 1D vector
    model.add(layers.Flatten(name="flatten"))

    # Dense hidden layer with 128 neurons and ReLU
    model.add(layers.Dense(128, activation='relu', name="dense1"))

    model.add(layers.BatchNormalization(name="bn_dense"))

    model.add(layers.Dropout(0.5, name="dropout3"))

    # Output layer: 36 classes with softmax
    model.add(layers.Dense(num_classes, activation='softmax', name="output_layer"))

    # ── Compile the model ────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary_dict(model):
    """
    Returns a dictionary with model architecture info for the frontend.
    Used for the 3D visualization layer structure.
    """
    summary = {
        "total_params": model.count_params(),
        "layers": []
    }

    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A",
        }
        summary["layers"].append(layer_info)

    return summary


def load_or_create_model(model_path="model/saved_model.h5"):
    """
    Load an existing model from disk, or create a fresh one if not found.
    
    Args:
        model_path: Path to the saved .h5 model file
        
    Returns:
        keras.Model: Loaded or freshly created model
    """
    import os

    if os.path.exists(model_path):
        print(f"[Model] Loading existing model from: {model_path}")
        try:
            model = keras.models.load_model(model_path)
            print("[Model] Model loaded successfully.")
            return model
        except Exception as e:
            print(f"[Model] Failed to load model: {e}. Creating new model.")

    print("[Model] No saved model found. Building new model.")
    model = build_model()
    return model


if __name__ == "__main__":
    # Quick test: build and display model summary
    model = build_model()
    model.summary()
    print(f"\nTotal Classes: {NUM_CLASSES}")
    print(f"Class Labels: {CLASSES}")
