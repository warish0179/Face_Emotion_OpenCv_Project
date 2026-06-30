# Real-Time Face Emotion Detector 🧠📸

A robust, real-time Facial Emotion Recognition system built using **Python**, **OpenCV**, and **DeepFace**. The application captures live video from your webcam, automatically isolates the primary user's face while filtering background noise, and displays recognized emotions instantly with dynamic text and vibrant matching emojis.

---

## ✨ Features
* **Real-Time Detection:** Smooth, low-latency facial expression tracking via standard webcam input.
* **Anti-Vibration Bounding Box:** Fine-tuned OpenCV configuration parameters to completely eliminate frame flickering and rapid box shaking.
* **Single Face Targeting:** Automatically filters out background ghost boxes to lock onto the largest, most prominent face (the primary user).
* **Full Expression Tracking:** Dynamically tracks Happy, Sad, Angry, Surprise, Neutral, Fear, and Disgust.
* **Mood-Adaptive UI:** Bounding box text and emoji colors shift in real-time according to the detected mood (e.g., Aggressive Red for Angry, Calm Blue for Sad).

---

## 🛠️ Tech Stack & Dependencies
* **Python 3.11+**
* **OpenCV (opencv-python):** For computer vision operations, face coordinates cropping, and frame rendering.
* **DeepFace:** For robust, abstract neural network-based facial landmark extraction.
* **TF-Keras & TensorFlow 2.21+:** Core machine learning framework supporting background weight configurations.

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/warish0179/Face_Emotion_OpenCv_Project.git](https://github.com/warish0179/Face_Emotion_OpenCv_Project.git)
cd Face_Emotion_OpenCv_Project

### 2. Isolate Environment via Virtualenv
To prevent strict dependency conflicts between modern TensorFlow updates and legacy global media packages, establish a localized sandbox environment:
# Generate the virtual environment
python -m venv emotion_env
# Activation for Windows (PowerShell)
.\emotion_env\Scripts\Activate.ps1

### 3. Install Package Prerequisites
Install dependencies directly inside your newly activated isolated workspace:
```bash
pip install opencv-python deepface tf-keras tensorflow

### 4. Execute the Tracker
```bash
python emotion_detector.py
Note: Upon initial launch, the system will securely cache the background evaluation weights. Press 'q' on your keyboard to gracefully terminate the video pipeline at any point.
