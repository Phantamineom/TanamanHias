import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # hindari backend mismatch

import streamlit as st
from PIL import Image
import numpy as np

# --- Coba import TF dengan pesan ramah kalau gagal
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
        st.error(f"File model tidak ditemukan: `{path}`. Pastikan nama file benar dan ada di repo.")
        st.stop()
    # compile False untuk inferensi, lebih cepat dan aman
    model = tf.keras.models.load_model(path, compile=False)
    return model

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx] * 100.0)

# ---------- UI ----------
st.markdown(
    """
    <style>
    .title { text-align:center;color:#4a7c59;font-size:2.2em;margin:6px 0 16px }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="title">🌱 Klasifikasi Tanaman Hias</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Foto Tanaman Hias", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("Silakan upload gambar dulu.")
    st.stop()

# Load model saat dibutuhkan (setelah user upload)
model = load_model(MODEL_PATH)

img = Image.open(uploaded)
st.image(img, caption="📷 Preview", width=320)

placeholder = st.empty()
with st.spinner("🔍 Menganalisis gambar..."):
    label, conf = predict(model, img)
    with placeholder.container():
        st.success(f"🌿 **Jenis Tanaman:** {label}")
        st.info(f"🔎 **Tingkat Keyakinan:** {conf:.2f}%")

if st.button("🔄 Analisis Ulang"):
    with st.spinner("🔁 Mengulang analisis..."):
        label, conf = predict(model, img)
        with placeholder.container():
            st.success(f"🌿 **Jenis Tanaman:** {label}")
            st.info(f"🔎 **Tingkat Keyakinan:** {conf:.2f}%")
