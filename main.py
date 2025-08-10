import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Memastikan Keras menggunakan TensorFlow

import streamlit as st
from PIL import Image
import numpy as np
import requests

# Import TensorFlow dengan penanganan error
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "Paket TensorFlow tidak ditemukan. "
        "Pastikan `tensorflow` ada di file requirements.txt, "
        "lalu deploy ulang aplikasi Anda."
    )
    st.stop()

# --- Konfigurasi ---
MODEL_PATH = "64B30E-ENB0-tanamanHias-v3.keras"
MODEL_URL = "https://huggingface.co/Phantamineom/TanamanHias/resolve/main/64B30E-ENB0-tanamanHias-v3.keras"
CLASS_LABELS = [
    "Aglaonema", "Daisy", "Dandelion", "Jasmine", "Lavender",
    "Lily Flower", "Rose", "Sunflower", "Tulip"
]

# --- Fungsi-Fungsi ---

def download_model(url, path):
    """
    Mengunduh file model dari URL jika belum ada di path lokal.
    """
    if not os.path.exists(path):
        with st.spinner(f"Mengunduh model dari Hugging Face... (ini hanya terjadi sekali)"):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()  # Akan error jika status code bukan 2xx
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model berhasil diunduh!")
            except requests.exceptions.RequestException as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()

def load_model_safe(path):
    """
    Memuat model Keras dengan aman.
    """
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari file: {e}")
        st.stop()

def preprocess_image(pil_img: Image.Image):
    """
    Mengubah ukuran, normalisasi, dan memastikan gambar SELALU 3 channel.
    Ini adalah fungsi "anti gagal" untuk mengatasi ValueError.
    """
    # 1. Coba konversi ke RGB dan resize
    img = pil_img.convert("RGB").resize((224, 224))
    
    # 2. Konversi ke array NumPy
    arr = np.asarray(img, dtype=np.float32)

    # 3. Lapisan Pertahanan: Pastikan array memiliki 3 channel
    if arr.ndim == 2:  # Jika gambar grayscale (hanya 2 dimensi)
        arr = np.expand_dims(arr, axis=-1) # Ubah ke (H, W, 1)
    
    if arr.shape[-1] == 1:  # Jika gambar punya 1 channel
        # Duplikasi channel tunggal menjadi 3 channel
        arr = np.concatenate([arr, arr, arr], axis=-1)

    # 4. Normalisasi piksel ke rentang [0, 1]
    arr /= 255.0

    # 5. Tambahkan dimensi batch untuk input model
    arr = np.expand_dims(arr, axis=0)
    
    # Dijamin berbentuk (1, 224, 224, 3)
    return arr

def predict(model, pil_img: Image.Image):
    """
    Melakukan prediksi pada gambar yang sudah diproses.
    """
    processed_image = preprocess_image(pil_img)
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    label = CLASS_LABELS[predicted_index]
    confidence = float(predictions[predicted_index] * 100.0)
    return label, confidence

# --- Antarmuka Pengguna (UI) Streamlit ---

st.markdown('<h2 style="text-align:center;color:#4a7c59;">🌱 Klasifikasi Tanaman Hias</h2>', unsafe_allow_html=True)

# Pertama, pastikan model sudah terunduh
download_model(MODEL_URL, MODEL_PATH)

# Komponen untuk upload file
uploaded_file = st.file_uploader(
    "Upload Foto Tanaman Hias",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Silakan upload sebuah gambar untuk memulai analisis.")
    st.stop()

# Buka dan tampilkan gambar
raw_image = Image.open(uploaded_file)
st.image(raw_image, caption="📷 Gambar yang Di-upload", width=320)

# Muat model dari file lokal
model = load_model_safe(MODEL_PATH)

# Lakukan prediksi
with st.spinner("🔍 Menganalisis gambar..."):
    predicted_label, prediction_confidence = predict(model, raw_image)
    
    # Tampilkan hasil
    st.success(f"🌿 Jenis Tanaman: **{predicted_label}**")
    st.info(f"🔎 Tingkat Keyakinan: **{prediction_confidence:.2f}%**")
