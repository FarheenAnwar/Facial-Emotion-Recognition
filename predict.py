"""
predict.py — Single-image facial emotion prediction.

Loads the trained model and class-index map, detects faces with Haar Cascade,
and prints a colour-coded confidence report for every detected face.

Usage:
    python predict.py <image_path>
    python predict.py <image_path> --no-detect   # skip face detection; use whole image
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ── Paths ──────────────────────────────────────────────────────────────────────
IMG_SIZE           = 48
MODEL_PATH         = "models/emotion_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
CASCADE_PATH       = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ANSI colour codes for terminal output
_ANSI = ["\033[91m", "\033[95m", "\033[93m", "\033[92m",
         "\033[94m", "\033[96m", "\033[97m"]
_RESET = "\033[0m"

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_resources():
    for path in [MODEL_PATH, CLASS_INDICES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: '{path}'\n"
                "Run train.py first to generate the model and class indices."
            )
    model = load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH) as fh:
        class_indices = json.load(fh)              # {emotion: index}

    # Invert to {index: emotion} for prediction lookup
    idx_to_emotion = {v: k for k, v in class_indices.items()}
    return model, idx_to_emotion


# ── Pre/post-processing ────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """BGR face crop → normalised (1, 48, 48, 1) float32 tensor."""
    gray      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm      = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def detect_faces(image: np.ndarray) -> list:
    """Return list of (x, y, w, h) bounding boxes for detected faces."""
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces   = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return list(faces) if len(faces) > 0 else []


def predict_emotion(model, face_tensor: np.ndarray, idx_to_emotion: dict):
    """Return (top_emotion, top_confidence, {emotion: score}) for a face."""
    probs     = model.predict(face_tensor, verbose=0)[0]
    top_idx   = int(np.argmax(probs))
    scores    = {idx_to_emotion[i]: float(probs[i]) for i in range(len(probs))}
    return idx_to_emotion[top_idx], float(probs[top_idx]), scores


# ── Display ────────────────────────────────────────────────────────────────────

def print_report(label: str, confidence: float, scores: dict, face_num: int):
    emoji = EMOTION_EMOJIS.get(label, "")
    print(f"\n{'─' * 46}")
    print(f"  Face #{face_num}  →  {emoji} {label.capitalize()}  ({confidence * 100:.1f}%)")
    print(f"{'─' * 46}")

    # Sort by confidence descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (emotion, score) in enumerate(sorted_scores):
        bar_len  = int(score * 32)
        bar      = "█" * bar_len + "░" * (32 - bar_len)
        color    = _ANSI[i % len(_ANSI)]
        emoji_e  = EMOTION_EMOJIS.get(emotion, "")
        print(f"  {color}{emoji_e} {emotion:<9}{_RESET} {bar}  {score * 100:5.1f}%")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def predict_image(image_path: str, use_detection: bool = True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, idx_to_emotion = load_resources()
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"OpenCV could not read: {image_path}")

    if use_detection:
        faces = detect_faces(image)
        if not faces:
            print("⚠  No faces detected — running prediction on the full image.")
            faces = [(0, 0, image.shape[1], image.shape[0])]
    else:
        faces = [(0, 0, image.shape[1], image.shape[0])]

    results = []
    for i, (x, y, w, h) in enumerate(faces):
        face_bgr    = image[y : y + h, x : x + w]
        face_tensor = preprocess_face(face_bgr)
        label, conf, scores = predict_emotion(model, face_tensor, idx_to_emotion)
        print_report(label, conf, scores, face_num=i + 1)
        results.append({"bbox": (x, y, w, h), "emotion": label,
                         "confidence": conf, "scores": scores})
    return results


def main():
    parser = argparse.ArgumentParser(description="Facial emotion prediction — single image.")
    parser.add_argument("image_path", help="Path to input image (jpg / png)")
    parser.add_argument(
        "--no-detect", action="store_true",
        help="Skip face detection and run on the full image",
    )
    args = parser.parse_args()
    predict_image(args.image_path, use_detection=not args.no_detect)


if __name__ == "__main__":
    main()
