"""
webcam.py — Real-time facial emotion recognition via webcam.

Controls:
    q  — quit
    s  — save screenshot  →  screenshots/<timestamp>.png

Features:
    • Haar Cascade face detection every frame
    • CNN emotion prediction per detected face
    • Bounding box + emotion label with emoji (rendered via PIL)
    • Confidence % shown next to each label
    • FPS counter in top-left corner
"""

import os
import json
import time
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

# ── Paths & constants ──────────────────────────────────────────────────────────
IMG_SIZE           = 48
MODEL_PATH         = "models/emotion_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
CASCADE_PATH       = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SCREENSHOT_DIR     = "screenshots"

EMOTION_EMOJIS = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# BGR colour per emotion for bounding boxes
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   160, 0  ),
    "fear":     (160, 0,   160),
    "happy":    (0,   220, 0  ),
    "neutral":  (190, 190, 190),
    "sad":      (220, 80,  0  ),
    "surprise": (0,   220, 220),
}

# ── Font loader ────────────────────────────────────────────────────────────────
# Try platform emoji fonts so glyphs render correctly; fall back gracefully.
_EMOJI_FONT_CANDIDATES = [
    "seguiemj.ttf",           # Windows Segoe UI Emoji
    "NotoColorEmoji.ttf",     # Linux
    "Apple Color Emoji.ttc",  # macOS
    "arial.ttf",              # last resort (emojis become boxes but text works)
]


def _load_pil_font(size: int) -> ImageFont.FreeTypeFont:
    for name in _EMOJI_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


_FONT_LABEL = _load_pil_font(20)
_FONT_FPS   = _load_pil_font(18)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_resources():
    for path in [MODEL_PATH, CLASS_INDICES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: '{path}'\n"
                "Run train.py first."
            )
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as fh:
        class_indices = json.load(fh)
    idx_to_emotion = {v: k for k, v in class_indices.items()}
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    return model, idx_to_emotion, cascade


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def draw_text_pil(
    frame: np.ndarray,
    text: str,
    xy: tuple,
    font: ImageFont.FreeTypeFont,
    color_rgb: tuple = (255, 255, 255),
    shadow: bool = True,
) -> np.ndarray:
    """
    Overlay Unicode / emoji text onto a BGR OpenCV frame using PIL.
    Returns the modified frame (BGR).
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    if shadow:
        # Thin drop-shadow for readability
        draw.text((xy[0] + 1, xy[1] + 1), text, font=font, fill=(0, 0, 0))
    draw.text(xy, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_webcam():
    model, idx_to_emotion, cascade = load_resources()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam — check that a camera is connected and not in use."
        )

    print("Webcam running.  q = quit   s = screenshot")

    fps_counter = 0
    fps_display = 0.0
    t_prev      = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face_gray   = gray[y : y + h, x : x + w]
            face_tensor = preprocess_face(face_gray)
            probs       = model.predict(face_tensor, verbose=0)[0]
            top_idx     = int(np.argmax(probs))
            emotion     = idx_to_emotion[top_idx]
            confidence  = float(probs[top_idx])
            color_bgr   = EMOTION_COLORS.get(emotion, (0, 255, 0))
            emoji       = EMOTION_EMOJIS.get(emotion, "")

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

            # Banner background above the box
            banner_h = 30
            cv2.rectangle(frame, (x, y - banner_h), (x + w, y), color_bgr, cv2.FILLED)

            # Emotion label + emoji rendered with PIL (supports Unicode)
            label_text  = f"{emoji} {emotion.capitalize()}  {confidence * 100:.1f}%"
            color_tuple = (
                color_bgr[2], color_bgr[1], color_bgr[0]
            )   # BGR → RGB for PIL
            frame = draw_text_pil(
                frame, label_text,
                (x + 4, y - banner_h + 4),
                _FONT_LABEL,
                color_rgb=(255, 255, 255),
                shadow=True,
            )

        # FPS overlay (plain cv2 text — no emoji needed)
        fps_counter += 1
        t_now = time.perf_counter()
        if t_now - t_prev >= 1.0:
            fps_display = fps_counter / (t_now - t_prev)
            fps_counter = 0
            t_prev      = t_now

        cv2.putText(
            frame, f"FPS: {fps_display:.1f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        face_count = len(faces) if hasattr(faces, "__len__") else 0
        cv2.putText(
            frame, f"Faces: {face_count}",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, "q: quit   s: screenshot",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
        )

        cv2.imshow("Emotion Recognition — Real-Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting …")
            break
        elif key == ord("s"):
            ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
            cv2.imwrite(filepath, frame)
            print(f"Screenshot saved: {filepath}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
