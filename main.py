import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # pastikan Keras pakai TF

import streamlit as st
from PIL import Image
import numpy as np

# Import TF dengan pesan ramah kalau gagal
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error("TensorFlow belum terpasang. Pastikan `tensorflow-cpu==2.16.1` ada di requirements.txt lalu redeploy.")
    st.stop()

MODEL_PATH = "64B30E-ENB0-tanamanHias-v3.keras"

CLASS_LABELS = [
    "Aglaonema", "Daisy", "Dandelion", "Jasmine", "Lavender",
    "Lily Flower", "Rose", "Sunflower", "Tulip"
]

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"File model tidak ditemukan: `{path}`. Pastikan nama file persis & ada di root repo.")
        st.stop()
    return tf.keras.models.load_model(path, compile=False)

def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx] * 100.0)

# ---------- UI ----------
st.markdown('<h2 style="text-align:center;color:#4a7c59;">🌱 Klasifikasi Tanaman Hias</h2>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Foto Tanaman Hias", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Silakan upload gambar dulu.")
    st.stop()

# Load model SETELAH ada kebutuhan (lebih stabil)
model = load_model(MODEL_PATH)

img = Image.open(uploaded)
st.image(img, caption="📷 Preview", width=320)

with st.spinner("🔍 Menganalisis gambar..."):
    label, conf = predict(model, img)
    st.success(f"🌿 Jenis Tanaman: {label}")
    st.info(f"🔎 Tingkat Keyakinan: {conf:.2f}%")

if st.button("🔄 Analisis Ulang"):
    with st.spinner("🔁 Mengulang analisis..."):
        label, conf = predict(model, img)
        st.success(f"🌿 Jenis Tanaman: {label}")
        st.info(f"🔎 Tingkat Keyakinan: {conf:.2f}%")
