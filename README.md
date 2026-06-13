# 🎭 Real-Time Facial Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning system that detects and classifies human emotions from facial expressions in real-time using a custom CNN trained on the FER2013 dataset.

---

## 📸 Demo

> Live webcam inference with bounding boxes, confidence scores, and a Streamlit analytics dashboard.

---

## ✨ Features

- 🎥 **Real-Time Webcam Detection** — live emotion recognition with bounding boxes and confidence scores
- 🖼️ **Image Upload** — upload any photo and get instant emotion prediction
- 🔍 **Grad-CAM Visualization** — highlights which facial regions influenced the prediction
- 📊 **Analytics Dashboard** — emotion history charts, distribution pie chart, confidence trends
- 🧠 **Model Performance Tab** — confusion matrix, training history, model architecture
- 💾 **Emotion History Logging** — all predictions saved to CSV with timestamps
- 📷 **Screenshot Capture** — press `S` in webcam mode to save a screenshot

---

## 🎯 Emotions Detected

| Emotion  | Emoji |
|----------|-------|
| Happy    | 😊    |
| Sad      | 😢    |
| Angry    | 😠    |
| Fear     | 😨    |
| Surprise | 😲    |
| Disgust  | 🤢    |
| Neutral  | 😐    |

---

## 🗂️ Project Structure

```
Emotion_Recognition/
│
├── dataset/                  # FER2013 dataset (not included, see Setup)
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       └── ...
│
├── models/                   # Saved model and plots
│   ├── emotion_model.h5
│   ├── class_indices.json
│   ├── confusion_matrix.png
│   └── training_history.png
│
├── screenshots/              # Webcam screenshots
├── app.py                    # Streamlit dashboard
├── train.py                  # Model training script
├── predict.py                # Single image prediction
├── webcam.py                 # Standalone OpenCV webcam script
├── requirements.txt
├── emotion_history.csv       # Auto-generated prediction log
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/FarheenAnwar/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### 2. Create Virtual Environment (Python 3.11 required)
```bash
py -3.11 -m venv venv311
venv311\Scripts\activate        # Windows
# source venv311/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Dataset
- Go to [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Download and extract
- Place the folders so your structure looks like:
```
dataset/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

---

## 🚀 Usage

### Step 1: Train the Model
```bash
python train.py
```
- Training takes 1–3 hours on CPU
- Expected accuracy: ~65–70% (normal for FER2013)
- Saves `models/emotion_model.h5` and evaluation plots

### Step 2: Test on a Single Image
```bash
python predict.py path/to/image.jpg
```

### Step 3: Run Webcam Detection
```bash
python webcam.py
```
- Press `S` to save a screenshot
- Press `Q` to quit

### Step 4: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🏗️ Model Architecture

```
Input (48×48×1)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(7, Softmax)
```

---

## 📦 Tech Stack

| Category         | Tools                          |
|------------------|--------------------------------|
| Deep Learning    | TensorFlow, Keras, CNN         |
| Computer Vision  | OpenCV, Haar Cascade           |
| Dashboard        | Streamlit, Plotly              |
| Data Processing  | NumPy, Pandas, scikit-learn    |
| Visualization    | Matplotlib, Seaborn, Grad-CAM  |

---

## 📝 Note on Accuracy

FER2013 is a challenging dataset — even state-of-the-art models achieve ~72–75%. An accuracy of 65–70% on this dataset is completely normal and acceptable. The "disgust" class has very few samples and is the hardest to classify correctly.

---

## 📄 Resume Description

> Developed a Real-Time Facial Emotion Recognition System using CNN, TensorFlow, and OpenCV. Trained on the FER2013 dataset (35,000+ images) to classify seven human emotions with ~65–70% accuracy. Integrated a webcam-based inference pipeline and built a Streamlit dashboard featuring live emotion analytics, confidence visualization, Grad-CAM heatmaps, and prediction history tracking.

**Skills demonstrated:** Computer Vision · Deep Learning · CNN · TensorFlow · OpenCV · Streamlit · Real-Time Video Processing · Grad-CAM · Data Visualization

---

## 👩‍💻 Author

**Farheen Anwar**  
[GitHub](https://github.com/FarheenAnwar)
