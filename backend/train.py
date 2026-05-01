"""
train.py - Training Pipeline for Handwritten Digit & Letter Recognition
Combines MNIST (digits 0-9) and EMNIST (letters A-Z) into a unified dataset.
Trains for at least 10 epochs with validation split and saves model.h5
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import build_model, CLASSES, NUM_CLASSES

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "model/saved_model.h5"
PLOT_DIR        = "model/plots"
EPOCHS          = 15       # Train for at least 10 epochs
BATCH_SIZE      = 128
VALIDATION_SPLIT = 0.15


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_mnist_digits():
    """
    Load MNIST digit dataset (0-9).
    Labels: 0-9 map directly to class indices 0-9.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("[Data] Loading MNIST digits...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Labels stay as 0-9
    print(f"[Data] MNIST: {len(x_train)} train, {len(x_test)} test samples")
    return x_train, y_train, x_test, y_test


def load_emnist_letters():
    """
    Load EMNIST letters dataset (A-Z = 26 classes).
    EMNIST letters labels are 1-26 (a=1 ... z=26).
    We remap to class indices 10-35 (A=10 ... Z=35).
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("[Data] Loading EMNIST letters...")

    try:
        # Try tensorflow_datasets first (preferred)
        import tensorflow_datasets as tfds

        # Load EMNIST letters split
        ds_train = tfds.load('emnist/letters', split='train', as_supervised=True, shuffle_files=True)
        ds_test  = tfds.load('emnist/letters', split='test',  as_supervised=True)

        def extract_emnist(ds):
            images, labels = [], []
            for img, lbl in ds:
                img_np = img.numpy().astype("float32") / 255.0
                # EMNIST images need to be transposed (rotated)
                img_np = np.transpose(img_np.reshape(28, 28))
                images.append(img_np.reshape(28, 28, 1))
                # Remap label: 1-26 → 10-35 (after digits 0-9)
                labels.append(int(lbl.numpy()) + 9)  # 1→10, 2→11, ..., 26→35
            return np.array(images), np.array(labels)

        x_train, y_train = extract_emnist(ds_train)
        x_test,  y_test  = extract_emnist(ds_test)
        print(f"[Data] EMNIST Letters: {len(x_train)} train, {len(x_test)} test samples")
        return x_train, y_train, x_test, y_test

    except Exception as e:
        print(f"[Data] tensorflow_datasets not available ({e}). Using synthetic letter data.")
        return _generate_synthetic_letters()


def _generate_synthetic_letters():
    """
    Fallback: generate synthetic letter data using PIL when EMNIST
    is not available. Draws A-Z characters on 28x28 canvas.
    """
    from PIL import Image, ImageDraw, ImageFont
    import random

    print("[Data] Generating synthetic letter data...")
    samples_per_class = 1000
    x_list, y_list = [], []

    for class_idx in range(26):
        letter = chr(ord('A') + class_idx)
        label  = class_idx + 10  # A=10, B=11, ..., Z=35

        for _ in range(samples_per_class):
            img = Image.new('L', (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            # Random size and position for augmentation
            font_size = random.randint(16, 22)
            x_off = random.randint(2, 6)
            y_off = random.randint(2, 6)

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            draw.text((x_off, y_off), letter, fill=255, font=font)

            arr = np.array(img).astype("float32") / 255.0
            x_list.append(arr.reshape(28, 28, 1))
            y_list.append(label)

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    # Shuffle
    idx = np.random.permutation(len(x_arr))
    x_arr, y_arr = x_arr[idx], y_arr[idx]

    split = int(0.85 * len(x_arr))
    return x_arr[:split], y_arr[:split], x_arr[split:], y_arr[split:]


def load_combined_dataset():
    """
    Combine MNIST digits + EMNIST letters into a unified dataset.
    
    Returns:
        Tuple: (x_train, y_train, x_test, y_test)
        Labels: 0-9 for digits, 10-35 for A-Z letters
    """
    # Load digits
    x_digit_train, y_digit_train, x_digit_test, y_digit_test = load_mnist_digits()

    # Load letters
    x_letter_train, y_letter_train, x_letter_test, y_letter_test = load_emnist_letters()

    # Combine train sets
    x_train = np.concatenate([x_digit_train, x_letter_train], axis=0)
    y_train = np.concatenate([y_digit_train, y_letter_train], axis=0)

    # Combine test sets
    x_test = np.concatenate([x_digit_test, x_letter_test], axis=0)
    y_test = np.concatenate([y_digit_test, y_letter_test], axis=0)

    # Shuffle
    train_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    test_idx = np.random.permutation(len(x_test))
    x_test, y_test = x_test[test_idx], y_test[test_idx]

    print(f"[Data] Combined: {len(x_train)} train, {len(x_test)} test samples")
    print(f"[Data] Labels range: {y_train.min()} to {y_train.max()} ({NUM_CLASSES} classes)")
    return x_train, y_train, x_test, y_test


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(
    model=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_save_path=MODEL_SAVE_PATH,
    plot_dir=PLOT_DIR,
    additional_data=None
):
    """
    Full training pipeline: load data, train, save model, plot results.
    
    Args:
        model:            Optional pre-existing model (for retraining)
        epochs:           Number of training epochs
        batch_size:       Training batch size
        model_save_path:  Path to save the trained model
        plot_dir:         Directory to save training plots
        additional_data:  Optional tuple (x_extra, y_extra) for user correction data
        
    Returns:
        dict: Training history metrics
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load dataset
    x_train, y_train, x_test, y_test = load_combined_dataset()

    # Optionally inject user correction data
    if additional_data is not None:
        x_extra, y_extra = additional_data
        x_train = np.concatenate([x_train, x_extra], axis=0)
        y_train = np.concatenate([y_train, y_extra], axis=0)
        print(f"[Train] Added {len(x_extra)} user correction samples.")

    # Build or reuse model
    if model is None:
        model = build_model()
    model.summary()

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        # Save best model during training
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[Train] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Train] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # ── Save Training Plots ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_training_plots(history, plot_dir, timestamp)
    _save_confusion_matrix(model, x_test, y_test, plot_dir, timestamp)

    # ── Build History Dict for API ────────────────────────────────────────────
    hist_dict = {
        "accuracy":     [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in history.history.get("loss", [])],
        "val_loss":     [float(v) for v in history.history.get("val_loss", [])],
        "test_accuracy": float(test_acc),
        "test_loss":     float(test_loss),
        "epochs":        len(history.history.get("accuracy", [])),
        "timestamp":     timestamp,
    }

    print(f"\n[Train] Model saved to: {model_save_path}")
    return hist_dict


# ─── Plotting ────────────────────────────────────────────────────────────────

def _save_training_plots(history, plot_dir, timestamp):
    """Save accuracy and loss curves as PNG files."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0a1a')

    epochs_range = range(1, len(history.history['accuracy']) + 1)

    # Accuracy plot
    ax1.set_facecolor('#0d0d2b')
    ax1.plot(epochs_range, history.history['accuracy'],     color='#00e5ff', linewidth=2, label='Train Accuracy')
    ax1.plot(epochs_range, history.history['val_accuracy'], color='#ff6b6b', linewidth=2, label='Val Accuracy',  linestyle='--')
    ax1.set_title('Model Accuracy', color='white', fontsize=14, pad=10)
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.set_ylabel('Accuracy', color='#aaa')
    ax1.legend(facecolor='#1a1a3a', labelcolor='white')
    ax1.tick_params(colors='#aaa')
    ax1.spines[:].set_color('#333')
    ax1.grid(True, color='#222', alpha=0.5)

    # Loss plot
    ax2.set_facecolor('#0d0d2b')
    ax2.plot(epochs_range, history.history['loss'],     color='#00e5ff', linewidth=2, label='Train Loss')
    ax2.plot(epochs_range, history.history['val_loss'], color='#ff6b6b', linewidth=2, label='Val Loss',  linestyle='--')
    ax2.set_title('Model Loss', color='white', fontsize=14, pad=10)
    ax2.set_xlabel('Epoch', color='#aaa')
    ax2.set_ylabel('Loss', color='#aaa')
    ax2.legend(facecolor='#1a1a3a', labelcolor='white')
    ax2.tick_params(colors='#aaa')
    ax2.spines[:].set_color('#333')
    ax2.grid(True, color='#222', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"training_curves_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()

    # Also save a "latest" version for the API to serve
    latest = os.path.join(plot_dir, "training_curves_latest.png")
    plt.figure(figsize=(14, 5))
    fig.savefig(latest, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()

    print(f"[Plot] Training curves saved: {path}")


def _save_confusion_matrix(model, x_test, y_test, plot_dir, timestamp, max_samples=2000):
    """Save a confusion matrix heatmap for test predictions."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Limit samples to prevent memory issues
    if len(x_test) > max_samples:
        idx = np.random.choice(len(x_test), max_samples, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    sns.heatmap(
        cm_norm,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        linewidths=0.1,
        linecolor='#222'
    )

    ax.set_title('Confusion Matrix (Normalized)', color='white', fontsize=16, pad=15)
    ax.set_xlabel('Predicted', color='#aaa', fontsize=12)
    ax.set_ylabel('Actual',    color='#aaa', fontsize=12)
    ax.tick_params(colors='#aaa', labelsize=9)

    path = os.path.join(plot_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0a0a1a')

    # Also save latest
    latest = os.path.join(plot_dir, "confusion_matrix_latest.png")
    plt.savefig(latest, dpi=120, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()

    print(f"[Plot] Confusion matrix saved: {path}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Handwriting Recognition - Training Pipeline")
    print("=" * 60)
    history = train_model()
    print("\n[Done] Training complete!")
    print(f"  Final Test Accuracy: {history['test_accuracy']:.4f}")
