import streamlit as st
import os
import io
from PIL import Image
import exifread
import numpy as np
import logging

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Road AI Simplified", page_icon="🛣️")

# Debug and logging setup
print("Streamlit app initialized successfully")
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models (cached to ensure they are used)
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    logging.info(f"Loaded YOLO model from {model_path}")
    return model

@st.cache_resource
def load_cnn_model(model_path):
    model = load_model(model_path)
    logging.info(f"Loaded CNN model from {model_path}")
    return model

try:
    yolo_damages = load_yolo_model("yolov8s_rdd.pt")
    yolo_signs = load_yolo_model("yolov8s_gtsdb.pt")
    cnn_model = load_cnn_model("gtsrb_cnn_model.h5")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# EXIF GPS extraction (simplified)
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

# Image processing function
def process_image(uploaded_file, mode, yolo_damages, yolo_signs, cnn_model):
    result = {"success": False}
    try:
        # Save the uploaded file
        file_path = os.path.join("outputs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        logging.info(f"Saved file to {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found after saving: {file_path}")

        # Load and process image
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        img_array = np.array(image)
        lat, lon = extract_gps_from_image(io.BytesIO(uploaded_file.getvalue()))

        # Use the appropriate YOLO model
        yolo_model = yolo_damages if mode == "Detect Damages" else yolo_signs
        results = yolo_model.predict(img_array, conf=0.1)[0]  # Lowered threshold for testing
        logging.info(f"Applied {mode} model, found {len(results.boxes)} detections")

        if not results.boxes:
            st.warning(f"No detections in {mode} mode for {uploaded_file.name}")
            return result

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[cls]
            logging.info(f"Detected {label} with confidence {conf}")

            cnn_label = None
            if mode == "Detect Traffic Signs":
                try:
                    roi = img_array[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi_resized = np.resize(roi, (48, 48, 3))  # Simplified resize
                        roi_normalized = roi_resized / 255.0
                        roi_input = np.expand_dims(roi_normalized, axis=0)
                        prediction = cnn_model.predict(roi_input, verbose=0)
                        cnn_label = np.argmax(prediction, axis=1)[0]
                        logging.info(f"CNN classified as {cnn_label}")
                except Exception as e:
                    logging.error(f"CNN failed: {e}")
                    st.warning(f"CNN classification failed for {uploaded_file.name}: {e}")

            detections.append({
                "Label": label,
                "Confidence": round(conf, 3),
                "CNN_Label": cnn_label,
                "Box": f"{x1},{y1},{x2},{y2}",
                "Latitude": lat,
                "Longitude": lon
            })

        result["success"] = True
        result["detections"] = detections
        return result

    except Exception as e:
        logging.error(f"Error processing {uploaded_file.name}: {e}")
        st.error(f"Error processing image: {e}")
        return result

# Streamlit UI
st.title("Road AI Simplified")

with st.form(key="analysis_form"):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    mode = st.selectbox("Mode", ["Detect Damages", "Detect Traffic Signs"])
    run_button = st.form_submit_button("Run Analysis")

if run_button and uploaded_file:
    result = process_image(uploaded_file, mode, yolo_damages, yolo_signs, cnn_model)
    if result["success"]:
        st.success(f"Processed {uploaded_file.name} with {len(result['detections'])} detections")
        for detection in result["detections"]:
            st.write(f"Label: {detection['Label']}, Confidence: {detection['Confidence']}")
            if detection["CNN_Label"] is not None:
                st.write(f"CNN Classification: {detection['CNN_Label']}")
            st.write(f"Box: {detection['Box']}")
            if detection["Latitude"] and detection["Longitude"]:
                st.write(f"GPS: {detection['Latitude']}, {detection['Longitude']}")
    else:
        st.warning("No detections or processing failed.")
