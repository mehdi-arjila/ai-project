"""
api.py - Flask REST API for Handwriting Recognition
Endpoints:
  GET  /health          → server status check
  POST /predict         → run inference on drawn image
  POST /train           → retrain model (with optional user corrections)
  GET  /plots/<name>    → serve training plot images
  GET  /               → serve frontend index.html
"""

import os
import sys
import json
import threading
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predict import predict, reload_model, get_layer_activations
from backend.model import CLASSES

# ─── App Setup ───────────────────────────────────────────────────────────────

# Serve frontend files from the /frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
PLOTS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "plots")
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "user_drawings")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)  # Enable CORS for frontend requests

# Training state (non-blocking background training)
training_state = {
    "is_training": False,
    "progress":    0,
    "message":     "idle",
    "last_result": None,
}

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static frontend assets (JS, CSS, etc.)."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    Returns server status and whether a trained model is available.
    """
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "saved_model.h5"
    )
    model_exists = os.path.exists(model_path)

    return jsonify({
        "status":        "ok",
        "model_loaded":  model_exists,
        "classes":       CLASSES,
        "num_classes":   len(CLASSES),
        "timestamp":     datetime.now().isoformat(),
        "is_training":   training_state["is_training"],
    })


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    Body: { "image": "<base64 PNG string>" }
    Returns: { "predicted", "confidence", "top3", "all_scores" }
    """
    try:
        data = request.get_json(force=True)

        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field in request body"}), 400

        image_data = data["image"]
        if not image_data:
            return jsonify({"error": "Empty image data"}), 400

        # Run prediction
        result = predict(image_data)

        # Optionally get layer activations for 3D visualization
        try:
            activations = get_layer_activations(image_data)
            result["activations"] = activations
        except Exception:
            result["activations"] = {}

        return jsonify(result)

    except Exception as e:
        print(f"[API] Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/save_drawing", methods=["POST"])
def save_drawing():
    """
    POST /save_drawing
    Body: { "image": "<base64>", "label": "A", "correction": true }
    Saves the drawing to disk for future retraining.
    """
    try:
        data  = request.get_json(force=True)
        image = data.get("image", "")
        label = data.get("label", "unknown").upper()
        
        if not image:
            return jsonify({"error": "No image provided"}), 400

        # Decode and save image
        import base64
        from PIL import Image
        from io import BytesIO

        if "," in image:
            image = image.split(",", 1)[1]

        img_bytes = base64.b64decode(image)
        img = Image.open(BytesIO(img_bytes)).convert("L")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename  = f"{label}_{timestamp}.png"
        save_path = os.path.join(DATA_DIR, label)
        os.makedirs(save_path, exist_ok=True)

        img.save(os.path.join(save_path, filename))

        return jsonify({"status": "saved", "filename": filename})

    except Exception as e:
        print(f"[API] Save drawing error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    POST /train
    Body (optional): { "epochs": 10, "use_user_data": true }
    Starts training in a background thread. Returns immediately.
    Poll /train_status for progress.
    """
    if training_state["is_training"]:
        return jsonify({"error": "Training already in progress"}), 409

    data   = request.get_json(force=True) or {}
    epochs = int(data.get("epochs", 15))

    def run_training():
        training_state["is_training"] = True
        training_state["progress"]    = 0
        training_state["message"]     = "Loading datasets..."

        try:
            # Import here to avoid circular imports at module load
            from backend.train import train_model

            # Check for user correction data
            additional_data = None
            if data.get("use_user_data") and os.path.exists(DATA_DIR):
                additional_data = _load_user_drawings()

            training_state["message"] = f"Training for {epochs} epochs..."
            result = train_model(
                epochs=epochs,
                additional_data=additional_data
            )

            # Reload model after training
            reload_model()

            training_state["last_result"] = result
            training_state["message"]     = f"Done! Accuracy: {result['test_accuracy']*100:.1f}%"
            training_state["progress"]    = 100

        except Exception as e:
            training_state["message"] = f"Training failed: {str(e)}"
            print(f"[Train] Error: {e}")
        finally:
            training_state["is_training"] = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        "status":  "started",
        "message": f"Training started for {epochs} epochs",
    })


@app.route("/train_status", methods=["GET"])
def train_status():
    """GET /train_status → current training progress."""
    return jsonify({
        "is_training": training_state["is_training"],
        "progress":    training_state["progress"],
        "message":     training_state["message"],
        "last_result": training_state["last_result"],
    })


@app.route("/plots/<filename>")
def serve_plot(filename):
    """GET /plots/<filename> → serve training plot images."""
    return send_from_directory(PLOTS_DIR, filename)


@app.route("/plots", methods=["GET"])
def list_plots():
    """GET /plots → list available plot files."""
    try:
        files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
        files.sort(reverse=True)
        return jsonify({"plots": files})
    except Exception:
        return jsonify({"plots": []})


# ─── Helper ──────────────────────────────────────────────────────────────────

def _load_user_drawings():
    """
    Load user-saved drawings from DATA_DIR.
    Each subfolder is named by class label (e.g. 'A', '3').
    Returns (x_array, y_array) ready for training.
    """
    import base64
    from PIL import Image
    from backend.model import CLASSES

    x_list, y_list = [], []

    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        label_upper = class_name.upper()
        if label_upper not in CLASSES:
            continue

        label_idx = CLASSES.index(label_upper)

        for fname in os.listdir(class_path):
            if not fname.endswith(".png"):
                continue
            try:
                img = Image.open(os.path.join(class_path, fname)).convert("L")
                img = img.resize((28, 28))
                arr = np.array(img, dtype=np.float32) / 255.0
                x_list.append(arr.reshape(28, 28, 1))
                y_list.append(label_idx)
            except Exception:
                continue

    if not x_list:
        return None

    return np.array(x_list), np.array(y_list)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Handwriting Recognition API - Starting...")
    print("  Frontend: http://localhost:5000")
    print("  Health:   http://localhost:5000/health")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
