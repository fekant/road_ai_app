import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
import cv2

# Download from Google Drive if missing
@st.cache_resource
def load_models():
    if not os.path.exists("yolov8s_gtsdb.pt"):
        gdown.download("https://drive.google.com/uc?id=141uXNq3hfhTBlxzGBZBzVMLrFs8cA5XX", "yolov8s_gtsdb.pt", quiet=False)
    if not os.path.exists("yolov8s_rdd.pt"):
        gdown.download("https://drive.google.com/file/d/1YOGQ1GU-gyt-zN6nqTdV8T_kD4WlN-qV", "yolov8s_rdd.pt", quiet=False)
    if not os.path.exists("gtsrb_cnn_model.h5"):
        gdown.download("https://drive.google.com/file/d/1sSa7cFrPZuPPL85LrQKJ8hpP5aypOKJI", "gtsrb_cnn_model.h5", quiet=False)

    model_signs = YOLO("yolov8s_gtsdb.pt")
    model_damage = YOLO("yolov8s_rdd.pt")
    classifier = load_model("gtsrb_cnn_model.h5")
    return model_signs, model_damage, classifier

model_signs, model_damage, classifier = load_models()

label_map_rdd = {
    "D00": "Longitudinal Crack",
    "D01": "Transverse Crack",
    "D10": "Alligator Crack",
    "D11": "Severe Crack",
    "D20": "White Line Degradation",
    "D40": "Pothole"
}

label_map_gtsrb = {
    i: name for i, name in enumerate([
        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
        "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
        "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 tons", "Right-of-way at intersection",
        "Priority road", "Yield", "Stop", "No vehicles", "Vehicles > 3.5 tons prohibited", "No entry", "General caution",
        "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road",
        "Road narrows on right", "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
        "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
        "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
        "Roundabout mandatory", "End of no passing", "End of no passing by vehicles > 3.5 tons"
    ])
}

st.title("🛣️ Road AI: Traffic Sign + Damage Detection")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
detect_signs = st.checkbox("Detect Traffic Signs", value=True)
detect_damage = st.checkbox("Detect Road Damage", value=True)

results_list = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        annotated = img_np.copy()
        filename = uploaded_file.name

        if detect_signs:
            signs_result = model_signs(img_np, save=False)[0]
            for box in signs_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = img_np[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (32, 32))
                arr = img_to_array(roi_resized) / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = classifier.predict(arr)[0]
                class_id = np.argmax(pred)
                label = label_map_gtsrb.get(class_id, str(class_id))
                conf = pred[class_id]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                results_list.append({
                    "Filename": filename, "Type": "Sign", "Label": label,
                    "Confidence": round(conf, 3), "Box": f"{x1},{y1},{x2},{y2}"
                })

        if detect_damage:
            damage_result = model_damage(img_np, save=False)[0]
            for box in damage_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                conf = float(box.conf)
                raw_label = model_damage.names[class_id]
                label = label_map_rdd.get(raw_label, raw_label)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                results_list.append({
                    "Filename": filename, "Type": "Damage", "Label": label,
                    "Confidence": round(conf, 3), "Box": f"{x1},{y1},{x2},{y2}"
                })

        st.image(annotated, caption=f"🔍 {filename}", use_column_width=True)

    # Show results table and allow CSV download
    df = pd.DataFrame(results_list)
    st.subheader("📋 Detection Results")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", csv, "detection_results.csv", "text/csv")
# Placeholder for final app.py
# Includes:
# - Traffic sign detection (YOLO + CNN)
# - Road damage detection
# - GPS EXIF reading
# - Display on map (if EXIF exists)
