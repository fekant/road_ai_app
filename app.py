import streamlit as st
import os
from PIL import Image
import exifread
import numpy as np
import cv2
import pandas as pd
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Φόρτωση μοντέλων
yolo_damages = YOLO("yolov8s_rdd.pt")
yolo_signs = YOLO("yolov8s_gtsdb.pt")
cnn_model = load_model("gtsrb_cnn_model.h5")

# Συνάρτηση για εξαγωγή GPS από EXIF
def extract_gps_from_image(file):
    file.seek(0)
    tags = exifread.process_file(file, details=False)
    try:
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
    except Exception:
        return None, None

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Road AI – Εντοπισμός Φθορών & Σημάτων με GPS")

uploaded_files = st.file_uploader("Ανέβασε εικόνες", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
mode = st.selectbox("Επιλογή Λειτουργίας", ["Detect Damages", "Detect Traffic Signs"])
run_button = st.button("🚀 Εκκίνηση Ανάλυσης")

results_list = []

if run_button and uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        filename = uploaded_file.name
        lat, lon = extract_gps_from_image(uploaded_file)

        if mode == "Detect Damages":
            results = yolo_damages.predict(img_array)[0]
        else:
            results = yolo_signs.predict(img_array)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[cls]

            results_list.append({
                "Filename": filename,
                "Type": "Damage" if mode == "Detect Damages" else "Sign",
                "Label": label,
                "Confidence": round(conf, 3),
                "Box": f"{x1},{y1},{x2},{y2}",
                "Latitude": lat,
                "Longitude": lon
            })

    df = pd.DataFrame(results_list)
    st.dataframe(df)

    if "Latitude" in df.columns and df["Latitude"].notnull().any():
        st.subheader("🗺️ Χάρτης Εντοπισμών")
        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=14)

        for _, row in df.dropna(subset=["Latitude", "Longitude"]).iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"{row['Label']} ({row['Confidence']}) - {row['Filename']}",
                icon=folium.Icon(color="red" if row["Type"] == "Damage" else "blue")
            ).add_to(m)

        st_folium(m, width=700)
