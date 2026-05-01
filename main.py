"""
main.py - Application Entry Point
Starts the Flask API server. Run this file to launch the full application.

Usage:
    python main.py [--train]

Flags:
    --train   Train the model before starting the server
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Handwriting Recognition AI - Full Stack Application"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model before starting the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    args = parser.parse_args()

    # Ensure model directory exists
    os.makedirs("model/plots", exist_ok=True)
    os.makedirs("data/user_drawings", exist_ok=True)

    # ── Optional: train first ────────────────────────────────────────────────
    if args.train:
        print("\n" + "=" * 60)
        print("  Starting Training Pipeline...")
        print("=" * 60 + "\n")
        from backend.train import train_model
        history = train_model()
        print(f"\n✓ Training complete! Test accuracy: {history['test_accuracy']*100:.2f}%\n")

    # ── Start the API server ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Handwriting Recognition AI")
    print(f"  → Open browser: http://localhost:{args.port}")
    print("=" * 60 + "\n")

    from backend.api import app
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
