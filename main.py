import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Pastikan Keras pakai TF

import streamlit as st
from PIL import Image
import numpy as np

# Import TensorFlow
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error("TensorFlow belum terpasang. Pastikan `tensorflow-cpu==2.16.1` ada di requirements.txt lalu redeploy.")
    st.stop()

MODEL_PATH = "64B20E-ENB0-tanamanHias-v3.keras"

CLASS_LABELS = [
    "Aglaonema", "Daisy", "Dandelion", "Jasmine", "Lavender",
    "Lily Flower", "Rose", "Sunflower", "Tulip"
]

def load_model_safe(path):
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print("✅ Model loaded successfully!")
        print("ℹ️ Model input shape:", model.input_shape)
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        raise e

def preprocess_image(pil_img: Image.Image):
    """
    Convert image to RGB, resize to 224x224, normalize, and expand dims.
    Ensures shape is always (1, 224, 224, 3)
    """
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    print(f"🔍 DEBUG - Preprocessed image shape: {arr.shape}")
    return arr

def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx] * 100.0)

# ---------- UI ----------
st.markdown('<h2 style="text-align:center;color:#4a7c59;">🌱 Klasifikasi Tanaman Hias</h2>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Foto Tanaman Hias", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Silakan upload gambar dulu.")
    st.stop()

# ----- PERUBAIKAN UTAMA DI SINI -----
# 1. Buka gambar dari file yang di-upload
raw_img = Image.open(uploaded)

# 2. Langsung konversi ke RGB. Ini memastikan gambar selalu punya 3 channel.
#    Ini adalah langkah paling krusial untuk memperbaiki error.
rgb_img = raw_img.convert("RGB")

# 3. Tampilkan gambar yang sudah pasti RGB ke pengguna
st.image(rgb_img, caption="📷 Preview", width=320)
# ------------------------------------

# Load model setelah ada kebutuhan (lebih stabil di Streamlit Cloud)
try:
    model = load_model_safe(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()


with st.spinner("🔍 Menganalisis gambar..."):
    # 4. Gunakan gambar yang sudah dikonversi (rgb_img) untuk prediksi
    label, conf = predict(model, rgb_img)
    st.success(f"🌿 Jenis Tanaman: {label}")
    st.info(f"🔎 Tingkat Keyakinan: {conf:.2f}%")