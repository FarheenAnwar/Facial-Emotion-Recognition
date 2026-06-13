# Facial-Emotion-Recognition
Real-time facial emotion recognition using CNN, TensorFlow, and OpenCV
# Real-Time Facial Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green?logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-blueviolet?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep-learning pipeline that classifies **7 facial emotions** in real time using a custom CNN trained on the FER2013 dataset. Ships with a full Streamlit dashboard, standalone webcam inference, Grad-CAM visualisation, and an analytics dashboard backed by logged prediction history.

---

## Features

- **7-class emotion recognition** — angry, disgust, fear, happy, neutral, sad, surprise
- **Custom CNN** — 3 convolutional blocks (64 → 128 → 256 filters) with BatchNorm & Dropout
- **Real-time webcam inference** — bounding boxes, emoji labels, confidence %, FPS counter
- **Streamlit dashboard** — 4-tab UI with live feed, image upload, analytics, model performance
- **Grad-CAM heatmap** — visualises which face region drove each prediction
- **Prediction history CSV** — logs every prediction with all 7 confidence scores
- **Analytics charts** — Plotly pie, line, bar charts over logged history
- **Screenshot capture** — press `s` in the webcam window to save a timestamped PNG

---

## Project Structure

```
Emotion_Recognition/
├── dataset/
│   ├── train/
│   │   ├── angry/        ← ~3,995 training images
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/        ← ~1,018 test images
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
├── models/                    ← created automatically during training
│   ├── emotion_model.h5
│   ├── class_indices.json
│   ├── confusion_matrix.png
│   └── training_history.png
├── screenshots/               ← created automatically
├── app.py                     ← Streamlit dashboard
├── train.py                   ← model training
├── predict.py                 ← single-image CLI prediction
├── webcam.py                  ← standalone OpenCV webcam
├── requirements.txt
├── emotion_history.csv        ← auto-created on first prediction
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
# If using git
git clone <your-repo-url>
cd Emotion_Recognition
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Replace `tensorflow>=2.10.0` in requirements.txt with `tensorflow[and-cuda]` and ensure your CUDA toolkit matches the TF version.

---

## Dataset Setup

The project uses the **FER2013** dataset in folder form (images already extracted per class).

### Download from Kaggle

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API key in `~/.kaggle/`
3. Download the dataset:

```bash
kaggle datasets download -d msambare/fer2013
```

This downloads a ZIP containing `train/` and `test/` subfolders, each with 7 emotion subdirectories.

### Alternative: Manual download

1. Go to [kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click **Download** → extract the ZIP
3. Copy the `train/` and `test/` folders into `dataset/`:

```
dataset/
├── train/
│   ├── angry/      ← JPEG images
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

> **Note:** The folder names must be lowercase (angry, disgust, fear, happy, neutral, sad, surprise) to match what `flow_from_directory` expects.

---

## Training

```bash
python train.py
```

What this does:
1. Loads images from `dataset/train/` and `dataset/test/` using `ImageDataGenerator.flow_from_directory()`
2. Applies augmentation on training data (horizontal flip, rotation ±10°, zoom ±10%, shifts)
3. Builds and trains the CNN for up to 50 epochs with EarlyStopping (patience=10)
4. Saves the best model to `models/emotion_model.h5`
5. Saves `models/class_indices.json` — the label→index mapping used by all other scripts
6. Prints a full classification report (precision, recall, F1 per class)
7. Saves `models/confusion_matrix.png` and `models/training_history.png`

**Expected training time:** ~15–30 min on a GPU, several hours on CPU  
**Expected test accuracy:** ~65–70% (FER2013 human-level accuracy is ~65%)

---

## Running Each Component

### Streamlit Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

| Tab | What you get |
|-----|-------------|
| **Live Webcam** | Start/stop webcam stream, real-time bounding boxes, Plotly confidence chart, auto CSV logging |
| **Image Upload** | Upload any photo, face detection, confidence chart, Grad-CAM heatmap overlay |
| **Analytics** | Pie chart, line chart, bar chart, last-20 table, CSV download — from `emotion_history.csv` |
| **Model Performance** | Confusion matrix, training curves, architecture summary, dataset distribution |

### Standalone Webcam (OpenCV window)

```bash
python webcam.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot to `screenshots/<timestamp>.png` |

### Single-Image Prediction (CLI)

```bash
# With face detection (default)
python predict.py path/to/photo.jpg

# Skip face detection — run on full image
python predict.py path/to/photo.jpg --no-detect
```

Prints the top emotion + a colour-coded bar for all 7 confidence scores.

---

## CNN Architecture

```
Input (48 × 48 × 1)
│
├── Block 1:  Conv2D(64) → BN → Conv2D(64)  → BN → MaxPool(2×2) → Dropout(0.25)
├── Block 2:  Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2×2) → Dropout(0.25)
├── Block 3:  Conv2D(256) → BN → Conv2D(256) → BN → MaxPool(2×2) → Dropout(0.25)
│
├── Flatten
├── Dense(512, relu) → BN → Dropout(0.5)
├── Dense(256, relu) → Dropout(0.3)
└── Dense(7, softmax)
```

- Optimizer: Adam (lr=0.001, decays via ReduceLROnPlateau)
- Loss: Categorical cross-entropy
- Total parameters: ~8.5M

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python train.py` first |
| `class_indices.json not found` | Run `python train.py` — it is saved automatically |
| `Cannot open webcam` | Check camera is connected and not used by another app |
| Low accuracy (~50%) | Ensure dataset images are in the correct subfolders; check class balance |
| Grad-CAM blank/black | Model must have at least one Conv2D layer; retrain if needed |
| `tf-keras-vis` install fails | Safe to ignore — Grad-CAM uses a manual `tf.GradientTape` fallback |
| Emoji boxes in webcam window | Install Segoe UI Emoji font (Windows) or Noto Color Emoji (Linux) |
| OOM during training | Reduce `BATCH_SIZE` in `train.py` (try 32) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep learning | TensorFlow / Keras |
| Face detection | OpenCV Haar Cascade |
| Dashboard | Streamlit |
| Charts | Plotly |
| Training plots | Matplotlib + Seaborn |
| Grad-CAM | tf.GradientTape (manual) |
| Image processing | OpenCV + Pillow |
| Data | pandas + NumPy |
| Evaluation | scikit-learn |

---

## Resume Bullets

- Built an end-to-end **real-time facial emotion recognition** pipeline using TensorFlow/Keras, achieving ~67% accuracy on FER2013 (7 classes, 35k images) — matching published baselines
- Designed a custom **9-layer CNN** with BatchNorm and Dropout regularisation; used Grad-CAM to produce interpretable heatmaps highlighting attention regions per prediction
- Developed a **4-tab Streamlit dashboard** with live webcam inference, image upload, Grad-CAM overlay, and a Plotly analytics dashboard backed by logged prediction history
- Implemented **real-time OpenCV webcam loop** with Haar Cascade face detection, PIL-rendered emoji labels, FPS counter, and screenshot capture

---

## License

MIT — free to use, modify, and distribute.
