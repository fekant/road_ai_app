# 🛣️ Road AI - Streamlit App

This Streamlit application performs **automatic detection and annotation** of:

- 🛑 Traffic Signs (using YOLOv8 + GTSRB CNN)
- 🕳️ Road Surface Damage (cracks, potholes, etc.)
- 📍 GPS mapping (if image contains EXIF location)

---

## 🚀 Features

- Upload one or multiple road images
- Detect and classify:
  - Traffic signs (e.g., Stop, Speed Limit 30, Yield)
  - Damage types (e.g., Pothole, Longitudinal Crack)
- Draw annotated bounding boxes directly on the image
- Generate a downloadable **CSV summary**
- Display GPS markers on map (if GPS metadata exists)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run the app

```bash
streamlit run app.py
```

---

## 📁 Requirements

Make sure you place the following models in the root directory:

- `yolov8s_gtsdb.pt` – for traffic sign detection
- `yolov8s_rdd.pt` – for road damage detection
- `gtsrb_cnn_model.h5` – for classifying traffic sign types

You can upload them to your Google Drive and download them inside `app.py`.

---

## 📍 Example Screenshot

![demo](https://user-images.githubusercontent.com/demo-path/screenshot.jpg)

---

## 🙌 Credits

Developed by [Your Name] as part of a traffic analysis and road maintenance project.

