import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

MODEL_PATH = "64B30E-ENB0-tanamanHias-v3.keras"
MODEL_URL = "https://huggingface.co/Phantamineom/TanamanHias/resolve/main/64B30E-ENB0-tanamanHias-v3.keras"
CLASS_NAMES = ["Aglaonema", "Daisy", "Dandelion", "Jasmine", 
               "Lavender", "Lily Flower","Rose", "Sunflower", "Tulip"]
CONFIDENCE_THRESHOLD = 50.0

def download_model(url, path):
    """
    Mengunduh model dari URL jika file belum ada secara lokal.
    """
    if not os.path.exists(path):
        with st.spinner("Mengunduh model dari Hugging Face (hanya sekali)..."):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()  # Cek jika ada error HTTP
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()

def load_model(path):
    """
    Memuat keseluruhan model (arsitektur + bobot) dari file.
    Fungsi ini bisa menangani format .keras maupun .h5.
    """
    try:
        # Menggunakan compile=False karena kita hanya butuh untuk inferensi
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari file: {e}")
        st.stop()

def preprocess_image(img: Image.Image):
    """
    Fungsi preprocessing "anti gagal".
    Memastikan gambar selalu berukuran (224, 224) dengan 3 channel warna (RGB).
    """
    # 1. Konversi ke array NumPy terlebih dahulu untuk memeriksa channel
    img_array = np.array(img).astype(np.float32)
    
    # 2. PERBAIKAN KUNCI: Tangani gambar RGBA (4 channel)
    if img_array.shape[-1] == 4:
        # Buang channel Alpha, ambil hanya 3 channel pertama (RGB)
        img_array = img_array[:, :, :3]

    # 3. Konversi kembali ke PIL Image untuk resize dan normalisasi yang konsisten
    img = Image.fromarray(np.uint8(img_array))
    
    # 4. Pastikan gambar berwarna (RGB) dan ubah ukuran
    img = img.convert('RGB').resize((224, 224))
    
    # 5. Konversi lagi ke array setelah diproses
    img_array = np.array(img).astype(np.float32)
    
    # 6. Normalisasi manual (penting untuk akurasi model Anda)
    img_array = img_array / 255.0
    
    # 7. Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, img: Image.Image):
    """
    Melakukan prediksi pada gambar yang sudah diproses.
    """
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)[0]

    probabilities = np.argmax(preds)
    
    pred_class = CLASS_NAMES[probabilities]
    pred_conf = preds[probabilities] * 100
    
    return pred_class, pred_conf

st.set_page_config(page_title="Klasifikasi Tanaman Hias", layout="centered")
st.title("Klasifikasi Tanaman Hias")

# Langkah 1: Pastikan model sudah ada (diunduh jika perlu)
download_model(MODEL_URL, MODEL_PATH)

# Langkah 2: Muat model
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload gambar Tanaman)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(img, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Prediksi"):
        pred_class, pred_conf = predict(model, img)
        st.success(f"**{pred_class}** — {pred_conf:.2f}%")
else:
    st.info("Silakan upload gambar untuk memulai.")
