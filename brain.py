# 🧠 Brain Tumor Classification — Streamlit

import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

MODEL_PATH = r"D:\Guvi Projects\6. Brain Tumour Classification\venv\models_outputs\MobileNetV2_best.pkl"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload a **brain MRI image** below to classify the tumor type.")

# -----------------------------
# 🧩 LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

with st.sidebar:
    st.header("Model Info")
    st.caption("Ensure the path below is correct:")
    st.code(MODEL_PATH)

try:
    assert os.path.exists(MODEL_PATH), f"Model file not found:\n{MODEL_PATH}"
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model:\n{e}")
    st.stop()

# -----------------------------
# 🔧 HELPERS
# -----------------------------
def preprocess_image(pil_img: Image.Image, target_hw=(224, 224)) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, target_hw)
    arr = tf.cast(arr, tf.float32) / 255.0
    arr = tf.expand_dims(arr, 0)
    return arr.numpy()

def safe_softmax(vec: np.ndarray) -> np.ndarray:
    x = vec - np.max(vec, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# -----------------------------
# 📤 IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("👆 Upload a brain MRI image to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Error loading image: {e}")
    st.stop()

st.image(image, caption="🩺 Uploaded MRI Image", use_container_width=True)

# -----------------------------
# 🔮 PREDICTION
# -----------------------------
img_batch = preprocess_image(image, IMG_SIZE)

try:
    preds = model.predict(img_batch)
    preds = np.array(preds)

    if preds.ndim == 2 and (preds.min() < 0 or preds.max() > 1):
        preds = safe_softmax(preds)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

probs = preds[0] if preds.ndim == 2 else preds
pred_class = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_class]
confidence = float(probs[pred_class])

# -----------------------------
# 🎯 RESULTS
# -----------------------------
st.subheader("🧾 Prediction Results")
st.success(f"**Predicted Tumor Type:** {pred_label}")
st.write(f"**Confidence:** {confidence * 100:.2f}%")

st.markdown("### 🔍 Model Confidence by Class")
conf_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
st.bar_chart(conf_data)

result = {
    "predicted_label": pred_label,
    "confidence": confidence,
    "per_class": conf_data
}

st.download_button(
    "⬇️ Download Result (JSON)",
    data=json.dumps(result, indent=2),
    file_name="prediction_result.json",
    mime="application/json"
)

# -----------------------------
# 📚 ABOUT
# -----------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
This application classifies brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

**Developed by:** Naren krish  
**Framework:** TensorFlow + Streamlit  
**Model Used:** MobileNetV2_best.pkl
""")
