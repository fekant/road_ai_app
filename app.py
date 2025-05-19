import streamlit as st
import os
import io
from PIL import Image
import exifread
import numpy as np
import cv2
import pandas as pd
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torch
import logging
import gc
from concurrent.futures import ThreadPoolExecutor
import requests
import json

# Ρύθμιση logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Allow Ultralytics DetectionModel in PyTorch safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Caching μοντέλων
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_cnn_model(model_path):
    return load_model(model_path)

# Φόρτωση μοντέλων
try:
    yolo_damages = load_yolo_model("yolov8s_rdd.pt")
    yolo_signs = load_yolo_model("yolov8s_gtsdb.pt")
    cnn_model = load_cnn_model("gtsrb_cnn_model.h5")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Δημιουργία φακέλου για αποθήκευση αποτελεσμάτων
os.makedirs("outputs", exist_ok=True)

# Συνάρτηση για εξαγωγή GPS από EXIF
def extract_gps_from_image(file_like):
    file_like.seek(0)
    try:
        tags = exifread.process_file(file_like, details=False)
        lat_ref = tags["GPS GPSLatitudeRef"].values
        lon_ref = tags["GPS GPSLongitudeRef"].values
        lat = tags["GPS GPSLatitude"].values
        lon = tags["GPS GPSLongitude"].values

        def dms_to_dd(dms, ref):
            d = float(dms[0].num) / dms[0].den
            m = float(dms[1].num) / dms[1].den
            s = float(dms[2].num) / dms[2].den
            dd = d + m/60 + s/3600
            if ref in ['S', 'W']:
                dd = -dd
            return dd

        return dms_to_dd(lat, lat_ref), dms_to_dd(lon, lon_ref)
    except Exception as e:
        logging.warning(f"No GPS data: {e}")
        return None, None

# Συνάρτηση επεξεργασίας εικόνας
def process_image(uploaded_file, mode, yolo_damages, yolo_signs, cnn_model):
    result = {}
    try:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)
        filename = uploaded_file.name
        lat, lon = extract_gps_from_image(io.BytesIO(file_bytes))

        # Έλεγχος μεγέθους αρχείου
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(f"File {filename} exceeds 10MB limit")

        # Ανίχνευση με YOLO
        results = yolo_damages.predict(img_array)[0] if mode == "Detect Damages" else yolo_signs.predict(img_array)[0]
        annotated_img = img_array.copy()
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[cls]

            # Σχεδίαση box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Ενσωμάτωση CNN για σήματα
            cnn_label = None
            if mode == "Detect Traffic Signs":
                roi = img_array[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (32, 32))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)
                prediction = cnn_model.predict(roi_input, verbose=0)
                cnn_label = np.argmax(prediction, axis=1)[0]

            result = {
                "Filename": filename,
                "Type": "Damage" if mode == "Detect Damages" else "Sign",
                "Label": label,
                "CNN_Label": cnn_label,
                "Confidence": round(conf, 3),
                "Box": f"{x1},{y1},{x2},{y2}",
                "Latitude": lat,
                "Longitude": lon,
                "Annotated_Path": os.path.join("outputs", f"annotated_{filename}")
            }

        # Αποθήκευση annotated εικόνας
        Image.fromarray(annotated_img).save(result["Annotated_Path"])
        logging.info(f"Processed {filename}: {len(results.boxes)} detections")
    except Exception as e:
        logging.error(f"Error processing {uploaded_file.name}: {e}")
        return None
    return result

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Road AI – Εντοπισμός Φθορών & Σημάτων με GPS")

# Προσαρμοσμένη θεματολογία
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

# Αρχικοποίηση session_state
if 'results_list' not in st.session_state:
    st.session_state.results_list = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'csv_file' not in st.session_state:
    st.session_state.csv_file = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = []

# Φόρμα για είσοδο
with st.form(key="analysis_form"):
    uploaded_files = st.file_uploader("Ανέβασε εικόνες", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    mode = st.selectbox("Επιλογή Λειτουργίας", ["Detect Damages", "Detect Traffic Signs"])
    run_button = st.form_submit_button("🚀 Εκκίνηση Ανάλυσης")

if run_button and uploaded_files:
    st.session_state.results_list = []
    st.session_state.df = None
    st.session_state.csv_file = None
    st.session_state.annotated_images = []

    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda f: process_image(f, mode, yolo_damages, yolo_signs, cnn_model), uploaded_files))
    st.session_state.results_list = [r for r in results if r]
    for i, _ in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / total_files)
    st.success(f"✅ Επεξεργάστηκαν {len(st.session_state.results_list)} εντοπισμοί!")
    gc.collect()  # Διαχείριση μνήμης

# Εμφάνιση αποτελεσμάτων
if st.session_state.results_list:
    st.session_state.df = pd.DataFrame(st.session_state.results_list)
    
    # Φιλτράρισμα αποτελεσμάτων
    st.subheader("🔍 Φίλτρα Αποτελεσμάτων")
    confidence_threshold = st.slider("Ελάχιστη Εμπιστοσύνη", 0.0, 1.0, 0.5)
    filtered_df = st.session_state.df[st.session_state.df["Confidence"] >= confidence_threshold]
    st.dataframe(filtered_df)

    # Εμφάνιση εικόνων
    st.subheader("📸 Επεξεργασμένες Εικόνες")
    cols = st.columns(2)
    for result in st.session_state.results_list:
        with cols[0]:
            st.image(Image.open(result["Filename"]), caption="Αρχική Εικόνα", use_column_width=True)
        with cols[1]:
            st.image(Image.open(result["Annotated_Path"]), caption="Επεξεργασμένη Εικόνα", use_column_width=True)
        st.session_state.annotated_images.append(result["Annotated_Path"])

    # Export CSV
    st.session_state.csv_file = "outputs/detections.csv"
    st.session_state.df.to_csv(st.session_state.csv_file, index=False)
    with open(st.session_state.csv_file, "rb") as f:
        st.download_button("📥 Κατέβασε CSV Αποτελεσμάτων", f, file_name="detections.csv")

    # Χάρτης
    if "Latitude" in st.session_state.df.columns and st.session_state.df["Latitude"].notnull().any():
        st.subheader("🗺️ Χάρτης Εντοπισμών")
        df = st.session_state.df.dropna(subset=["Latitude", "Longitude"])
        if not df.empty:
            m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=14)
            for _, row in df.iterrows():
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{row['Label']} ({row['Confidence']}) - {row['Filename']}",
                    icon=folium.Icon(color="red" if row["Type"] == "Damage" else "blue")
                ).add_to(m)
            st_folium(m, width=700, key="folium_map")
        else:
            st.warning("No valid GPS coordinates available for mapping.")
else:
    if run_button and uploaded_files:
        st.warning("Δεν εντοπίστηκαν αντικείμενα στις εικόνες που ανέβηκαν.")


