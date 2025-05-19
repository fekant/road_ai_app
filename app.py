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

# Allow Ultralytics DetectionModel in PyTorch safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Φόρτωση μοντέλων
try:
    yolo_damages = YOLO("yolov8s_rdd.pt")
    yolo_signs = YOLO("yolov8s_gtsdb.pt")
    cnn_model = load_model("gtsrb_cnn_model.h5")  # Remove if unused
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
        st.warning(f"No GPS data found: {e}")
        return None, None

# Αρχικοποίηση session_state
if 'results_list' not in st.session_state:
    st.session_state.results_list = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'csv_file' not in st.session_state:
    st.session_state.csv_file = None

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Road AI – Εντοπισμός Φθορών & Σημάτων με GPS")

uploaded_files = st.file_uploader("Ανέβασε εικόνες", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
mode = st.selectbox("Επιλογή Λειτουργίας", ["Detect Damages", "Detect Traffic Signs"])
run_button = st.button("🚀 Εκκίνηση Ανάλυσης")

if run_button and uploaded_files:
    # Καθαρισμός προηγούμενων αποτελεσμάτων
    st.session_state.results_list = []
    st.session_state.df = None
    st.session_state.csv_file = None

    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img_array = np.array(image)
            filename = uploaded_file.name
            lat, lon = extract_gps_from_image(io.BytesIO(file_bytes))

            # Ανίχνευση με YOLO
            results = yolo_damages.predict(img_array)[0] if mode == "Detect Damages" else yolo_signs.predict(img_array)[0]

            # Annotated εικόνα
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

                st.session_state.results_list.append({
                    "Filename": filename,
                    "Type": "Damage" if mode == "Detect Damages" else "Sign",
                    "Label": label,
                    "Confidence": round(conf, 3),
                    "Box": f"{x1},{y1},{x2},{y2}",
                    "Latitude": lat,
                    "Longitude": lon
                })

            # Αποθήκευση annotated εικόνας
            output_filename = os.path.join("outputs", f"annotated_{filename}")
            Image.fromarray(annotated_img).save(output_filename)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

# Εμφάνιση αποτελεσμάτων από session_state
if st.session_state.results_list:
    st.session_state.df = pd.DataFrame(st.session_state.results_list)
    st.dataframe(st.session_state.df)

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
