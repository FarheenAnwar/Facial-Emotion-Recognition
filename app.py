"""
app.py — Streamlit dashboard for facial emotion recognition.

Four tabs:
  1.  Live Webcam    — real-time inference, confidence chart, CSV logging
  2.  Image Upload   — per-image prediction, Grad-CAM overlay, Plotly chart
  3.  Analytics      — history CSV analysis with Plotly charts
  4.  Model Performance — confusion matrix, training history, architecture

Run:
    streamlit run app.py
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Paths & constants ──────────────────────────────────────────────────────────
IMG_SIZE           = 48
MODEL_PATH         = "models/emotion_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
CASCADE_PATH       = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HISTORY_CSV        = "emotion_history.csv"
SCREENSHOT_DIR     = "screenshots"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}
EMOTION_COLORS = {
    "angry":    "#e74c3c", "disgust": "#27ae60", "fear":    "#8e44ad",
    "happy":    "#f39c12", "neutral": "#95a5a6", "sad":     "#2980b9",
    "surprise": "#1abc9c",
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("model", None), ("idx_to_emotion", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Resource loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading emotion model …")
def get_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None
    if not os.path.exists(CLASS_INDICES_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    # Warm up the model with a dummy input so layers have defined outputs
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    model.predict(dummy, verbose=0)
    with open(CLASS_INDICES_PATH) as fh:
        class_indices = json.load(fh)
    idx_to_emotion = {v: k for k, v in class_indices.items()}
    return model, idx_to_emotion


def get_cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)


# ── Inference helpers ──────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def detect_and_predict(frame: np.ndarray, model, idx_to_emotion: dict, cascade):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results = []
    for (x, y, w, h) in faces:
        face_bgr    = frame[y : y + h, x : x + w]
        face_tensor = preprocess_face(face_bgr)
        probs       = model.predict(face_tensor, verbose=0)[0]
        top_idx     = int(np.argmax(probs))
        emotion     = idx_to_emotion[top_idx]
        confidence  = float(probs[top_idx])
        scores      = {idx_to_emotion[i]: float(probs[i]) for i in range(len(probs))}

        color = (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, cv2.FILLED)
        cv2.putText(
            frame,
            f"{emotion.capitalize()}  {confidence * 100:.1f}%",
            (x + 4, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        results.append({
            "emotion":    emotion,
            "confidence": confidence,
            "scores":     scores,
            "bbox":       (x, y, w, h),
            "top_idx":    top_idx,
        })
    return frame, results


# ── CSV logging ────────────────────────────────────────────────────────────────

def log_prediction(emotion: str, confidence: float, scores: dict):
    row = {
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "emotion":    emotion,
        "confidence": round(confidence * 100, 2),
    }
    for label in EMOTION_LABELS:
        row[label] = round(scores.get(label, 0.0), 4)

    df_row = pd.DataFrame([row])
    header = not os.path.exists(HISTORY_CSV)
    df_row.to_csv(HISTORY_CSV, mode="a", header=header, index=False)


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def plotly_confidence_bar(scores: dict, title: str = "Confidence Scores") -> go.Figure:
    emotions = sorted(scores.keys(), key=lambda e: scores[e], reverse=True)
    values   = [scores[e] * 100 for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#888") for e in emotions]

    fig = go.Figure(go.Bar(
        x=values, y=emotions,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 110]),
        height=300,
        margin=dict(l=0, r=40, t=40, b=0),
    )
    return fig


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def compute_gradcam(model, img_array, class_idx):
    """
    Compute Grad-CAM heatmap.
    img_array : shape (1, 48, 48, 1), float32 [0,1]
    class_idx : int — predicted class index
    Returns   : float32 heatmap (H, W) normalised [0,1], or None on failure
    """
    try:
        # Find last Conv2D layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            return None

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[:, class_idx]

        grads        = tape.gradient(loss, conv_outputs)          # (1, h, w, filters)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))     # (filters,)

        conv_outputs = conv_outputs[0]                            # (h, w, filters)
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap      = tf.squeeze(heatmap)                        # (h, w)

        # ReLU + normalise
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            return None
        heatmap = (heatmap / max_val).numpy().astype(np.float32)
        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return None


def apply_gradcam_overlay(face_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    Overlay a Grad-CAM heatmap on a BGR face crop.
    Returns an RGB image ready for st.image().
    """
    h, w = face_bgr.shape[:2]

    # Resize heatmap to face size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to uint8 and apply JET colormap → BGR
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # Blend
    face_float    = face_bgr.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    blended_bgr   = cv2.addWeighted(face_float, 1 - alpha, heatmap_float, alpha, 0)
    blended_bgr   = np.clip(blended_bgr, 0, 255).astype(np.uint8)

    # Return RGB versions
    face_rgb    = cv2.cvtColor(face_bgr,       cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(blended_bgr,    cv2.COLOR_BGR2RGB)

    return face_rgb, heatmap_rgb, overlay_rgb


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

model, idx_to_emotion = get_model_and_labels()

with st.sidebar:
    st.title("Emotion Recognition")
    st.markdown("---")

    if model is None:
        st.error("Model not found.\nRun `python train.py` first.")
    else:
        st.success("Model loaded ✅")
        st.metric("Parameters", f"{model.count_params():,}")

    st.markdown("---")
    st.markdown("**Emotion Classes**")
    for emo in EMOTION_LABELS:
        st.write(f"{EMOTION_EMOJIS[emo]}  {emo.capitalize()}")

    st.markdown("---")
    if os.path.exists(HISTORY_CSV):
        try:
            df_side = pd.read_csv(HISTORY_CSV)
            st.metric("Total Predictions", len(df_side))
        except Exception:
            pass

    st.caption("Emotion Detection System")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Webcam",
    "Image Upload",
    "Analytics Dashboard",
    "Model Performance",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Webcam
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Live Webcam")
    st.caption("Check the box to start streaming. Uncheck to stop.")

    if model is None:
        st.warning("Train the model first to enable webcam inference.")
    else:
        run_webcam = st.checkbox("▶ Start / Stop Webcam", key="webcam_toggle")

        col_feed, col_chart = st.columns([3, 2])
        frame_placeholder  = col_feed.empty()
        chart_placeholder  = col_chart.empty()
        status_placeholder = st.empty()

        if run_webcam:
            cascade = get_cascade()
            cap     = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Cannot open webcam — check that a camera is connected.")
                st.session_state.webcam_toggle = False
            else:
                os.makedirs(SCREENSHOT_DIR, exist_ok=True)

                while st.session_state.get("webcam_toggle", False):
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam feed lost.")
                        break

                    annotated, preds = detect_and_predict(
                        frame.copy(), model, idx_to_emotion, cascade
                    )
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

                    if preds:
                        pred = preds[0]
                        log_prediction(pred["emotion"], pred["confidence"], pred["scores"])

                        fig = plotly_confidence_bar(
                            pred["scores"],
                            title=f"{EMOTION_EMOJIS.get(pred['emotion'],'')} "
                                  f"{pred['emotion'].capitalize()}  "
                                  f"{pred['confidence']*100:.1f}%",
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        status_placeholder.info(
                            f"Detected: **{pred['emotion'].capitalize()}** — "
                            f"{pred['confidence']*100:.1f}% confidence"
                        )

                cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Image Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Image Upload")

    if model is None:
        st.warning("Train the model first.")
    else:
        uploaded = st.file_uploader(
            "Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_np  = np.array(pil_img)
            frame   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cascade = get_cascade()

            annotated, preds = detect_and_predict(
                frame.copy(), model, idx_to_emotion, cascade
            )
            rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col_img, col_info = st.columns(2)
            col_img.image(rgb_annotated, caption="Detection Result", use_column_width=True)

            if not preds:
                col_info.info("No faces detected in this image.")
            else:
                pred = preds[0]
                log_prediction(pred["emotion"], pred["confidence"], pred["scores"])

                with col_info:
                    st.markdown(
                        f"### {EMOTION_EMOJIS.get(pred['emotion'],'')} "
                        f"{pred['emotion'].capitalize()}  "
                        f"— {pred['confidence']*100:.1f}%"
                    )
                    fig = plotly_confidence_bar(pred["scores"])
                    st.plotly_chart(fig, use_container_width=True)

                # ── Grad-CAM ──────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("🔍 Grad-CAM Visualisation")
                st.caption(
                    "Highlights which face regions most influenced the prediction. "
                    "Warmer colours (red/yellow) = higher influence."
                )

                x, y, w, h = pred["bbox"]
                face_bgr    = frame[y : y + h, x : x + w]
                face_tensor = preprocess_face(face_bgr)
                top_idx     = pred["top_idx"]

                heatmap = compute_gradcam(model, face_tensor, top_idx)

                if heatmap is not None:
                    face_rgb, heatmap_rgb, overlay_rgb = apply_gradcam_overlay(face_bgr, heatmap)

                    # Resize to uniform display size
                    disp_size = (200, 200)
                    face_disp    = cv2.resize(face_rgb,    disp_size)
                    heatmap_disp = cv2.resize(heatmap_rgb, disp_size)
                    overlay_disp = cv2.resize(overlay_rgb, disp_size)

                    c1, c2, c3 = st.columns(3)
                    c1.image(face_disp,    caption="Face Crop",        use_column_width=True)
                    c2.image(heatmap_disp, caption="Grad-CAM Heatmap", use_column_width=True)
                    c3.image(overlay_disp, caption="Overlay",           use_column_width=True)
                else:
                    st.info("Grad-CAM could not be generated for this image.")

            # Show all faces if more than one detected
            if len(preds) > 1:
                st.markdown("---")
                st.subheader(f"All {len(preds)} Detected Faces")
                cols = st.columns(min(len(preds), 4))
                for i, p in enumerate(preds):
                    x, y, w, h = p["bbox"]
                    face_crop   = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    cols[i % 4].image(
                        face_crop,
                        caption=f"{EMOTION_EMOJIS.get(p['emotion'],'')} "
                                f"{p['emotion'].capitalize()} {p['confidence']*100:.1f}%",
                        use_column_width=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Analytics Dashboard")

    if not os.path.exists(HISTORY_CSV):
        st.info(
            "No prediction history yet.\n\n"
            "Use the **Webcam** or **Image Upload** tabs to generate data."
        )
    else:
        try:
            df = pd.read_csv(HISTORY_CSV)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as exc:
            st.error(f"Failed to load history CSV: {exc}")
            st.stop()

        if df.empty:
            st.info("The history CSV is empty — no predictions logged yet.")
        else:
            total_preds  = len(df)
            top_emotion  = df["emotion"].mode()[0]
            avg_conf     = df["confidence"].mean()
            duration_str = "—"
            if total_preds > 1:
                delta = df["timestamp"].max() - df["timestamp"].min()
                secs  = int(delta.total_seconds())
                duration_str = (
                    f"{secs // 3600}h {(secs % 3600) // 60}m {secs % 60}s"
                    if secs >= 3600
                    else f"{secs // 60}m {secs % 60}s"
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Predictions",    total_preds)
            m2.metric("Most Frequent Emotion",
                      f"{EMOTION_EMOJIS.get(top_emotion,'')} {top_emotion.capitalize()}")
            m3.metric("Avg Confidence",        f"{avg_conf:.1f}%")
            m4.metric("Session Duration",      duration_str)

            st.markdown("---")
            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.subheader("Emotion Distribution")
                counts = df["emotion"].value_counts().reset_index()
                counts.columns = ["emotion", "count"]
                fig_pie = px.pie(
                    counts, values="count", names="emotion",
                    color="emotion", color_discrete_map=EMOTION_COLORS, hole=0.35,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                st.subheader("Average Confidence by Emotion")
                avg_by_emo = (
                    df.groupby("emotion")["confidence"]
                    .mean().reset_index()
                    .sort_values("confidence", ascending=False)
                )
                fig_bar = px.bar(
                    avg_by_emo, x="emotion", y="confidence",
                    color="emotion", color_discrete_map=EMOTION_COLORS,
                    labels={"confidence": "Avg Confidence (%)"},
                    text_auto=".1f",
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.subheader("Confidence Over Time")
            fig_line = px.line(
                df, x="timestamp", y="confidence", color="emotion",
                color_discrete_map=EMOTION_COLORS, markers=True,
                labels={"confidence": "Confidence (%)", "timestamp": "Time"},
            )
            fig_line.update_layout(height=350)
            st.plotly_chart(fig_line, use_container_width=True)

            st.subheader("Emotion Sequence Over Time")
            df_seq  = df.copy()
            emo_order = {e: i for i, e in enumerate(EMOTION_LABELS)}
            df_seq["emotion_id"] = df_seq["emotion"].map(emo_order)
            fig_seq = px.scatter(
                df_seq, x="timestamp", y="emotion_id", color="emotion",
                color_discrete_map=EMOTION_COLORS,
                labels={"emotion_id": "Emotion", "timestamp": "Time"},
            )
            fig_seq.update_yaxes(
                tickvals=list(emo_order.values()),
                ticktext=[e.capitalize() for e in EMOTION_LABELS],
            )
            fig_seq.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_seq, use_container_width=True)

            st.markdown("---")
            st.subheader("Last 20 Predictions")
            display_cols = ["timestamp", "emotion", "confidence"] + [
                c for c in EMOTION_LABELS if c in df.columns
            ]
            st.dataframe(
                df[display_cols].tail(20).iloc[::-1].reset_index(drop=True),
                use_container_width=True,
            )
            st.download_button(
                "⬇ Download Full History CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="emotion_history.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Model Performance")

    col_cm, col_hist = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        if os.path.exists("models/confusion_matrix.png"):
            st.image("models/confusion_matrix.png", use_column_width=True)
        else:
            st.info("Run `python train.py` to generate the confusion matrix.")

    with col_hist:
        st.subheader("Training History")
        if os.path.exists("models/training_history.png"):
            st.image("models/training_history.png", use_column_width=True)
        else:
            st.info("Run `python train.py` to generate training plots.")

    st.markdown("---")
    st.subheader("Model Architecture")
    if model is not None:
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines), language="text")
    else:
        st.info("Model not loaded.")

    st.markdown("---")
    st.subheader("Dataset Class Distribution")
    for split in ["train", "test"]:
        split_dir = f"dataset/{split}"
        if os.path.isdir(split_dir):
            rows = []
            for emo in sorted(os.listdir(split_dir)):
                emo_dir = os.path.join(split_dir, emo)
                if os.path.isdir(emo_dir):
                    count = len([
                        f for f in os.listdir(emo_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
                    rows.append({"Emotion": emo.capitalize(), "Images": count})
            if rows:
                df_dist = pd.DataFrame(rows)
                total   = df_dist["Images"].sum()
                df_dist["Share (%)"] = (df_dist["Images"] / total * 100).round(1)
                st.markdown(f"**{split.capitalize()} set** — {total:,} images total")
                st.dataframe(df_dist, use_container_width=True, hide_index=True)