# Neural Vision — Handwriting Recognition AI

Full-stack AI app: CNN-based recognition of handwritten digits (0–9) and letters (A–Z) with a live 3D neural network visualization built with Three.js.

---

## Project Structure

```
preject.ai.pfe/
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── backend/
│   ├── model.py             ← CNN architecture (4 layers)
│   ├── train.py             ← Training pipeline (MNIST + EMNIST)
│   ├── predict.py           ← Inference engine
│   └── api.py               ← Flask REST API
├── frontend/
│   ├── index.html           ← UI
│   ├── style.css            ← Futuristic dark theme
│   ├── app.js               ← Canvas + API calls
│   └── three_scene.js       ← Three.js 3D visualization
├── model/
│   ├── saved_model.h5       ← Saved after training
│   └── plots/               ← Training curves + confusion matrix
└── data/
    └── user_drawings/       ← User-saved drawings for retraining
```

---

## Quick Start

### Step 1 — Install Python dependencies

```powershell
cd c:\preject.ai.pfe
pip install -r requirements.txt
```

### Step 2a — Train the model first (≈10–20 min)

```powershell
python main.py --train
```

This trains on MNIST + EMNIST for 15 epochs, saves `model/saved_model.h5`, and generates accuracy/confusion plots.

### Step 2b — OR: Start server without training (if model already exists)

```powershell
python main.py
```

### Step 3 — Open the app

Browser opens automatically. Or go to: **http://localhost:5000**

---

## API Endpoints

| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/health`       | Server + model status                |
| POST   | `/predict`      | Run inference on base64 canvas image |
| POST   | `/save_drawing` | Save drawing with label for retraining|
| POST   | `/train`        | Start background retraining          |
| GET    | `/train_status` | Poll training progress               |
| GET    | `/plots/<file>` | Serve training plot images           |

---

## How to Use the App

1. **Draw** a digit or letter on the black canvas (use mouse or touch)
2. Click **Predict** (or press `Enter`) — watch the 3D animation fire
3. See the **predicted character**, confidence %, top-3 results, and per-class bar chart
4. **Correct** wrong predictions: type the right label and click Save
5. **Retrain** with your saved drawings using the Train panel
6. View **training plots** (accuracy curves + confusion matrix) via the buttons

### Keyboard Shortcuts
- `Enter` — Predict
- `Delete` / `Escape` — Clear canvas

---

## CNN Architecture

```
Input (28×28×1)
  ↓
Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool(2×2) + Dropout(0.25)
  ↓
Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2) + Dropout(0.25)
  ↓
Flatten → Dense(128, ReLU) + BatchNorm + Dropout(0.5)
  ↓
Dense(36, Softmax)   ← 0–9 + A–Z
```

---

## Troubleshooting

**"No trained model found"** → Run `python main.py --train` first.

**EMNIST not loading** → The trainer falls back to synthetic letter generation automatically. For full EMNIST data: `pip install tensorflow-datasets`

**Port already in use** → `python main.py --port 5001`
