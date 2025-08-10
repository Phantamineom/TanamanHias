import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Memastikan Keras menggunakan TensorFlow

import streamlit as st
from PIL import Image
import numpy as np
import requests

# Import TensorFlow dengan penanganan error jika modul tidak ditemukan
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "Paket TensorFlow tidak ditemukan. "
        "Pastikan `tensorflow-cpu==2.16.1` ada di file requirements.txt, "
        "lalu deploy ulang aplikasi Anda."
    )
    st.stop()

# --- Konfigurasi ---

MODEL_PATH = "64B30E-ENB0-tanamanHias-v3.keras"
MODEL_URL = "https://huggingface.co/Phantamineom/TanamanHias/resolve/main/64B30E-ENB0-tanamanHias-v3.keras"
CLASS_LABELS = [
    "Aglaonema", "Daisy", "Dandelion", "Jasmine", "Lavender",
    "Lily Flower", "Rose", "Sunflower", "Tulip"]

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model dari Hugging Face..."):
            r = requests.get(MODEL_URL, stream=True)
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Gagal mengunduh model.")
                st.stop()




# --- Fungsi-Fungsi ---

def load_model_safe(path):
    """
    Memuat model Keras dengan aman dan menampilkan pesan status.
    """
    try:
        # Menggunakan compile=False karena model hanya untuk inferensi
        model = tf.keras.models.load_model(path, compile=False)
        print("✅ Model berhasil dimuat!")
        # Menampilkan input shape yang diharapkan model untuk debugging
        print(f"ℹ️ Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{path}' ada di direktori yang sama.")
        st.error(f"Detail Error: {e}")
        # Menghentikan eksekusi jika model gagal dimuat
        st.stop()

def preprocess_image(pil_img: Image.Image):
    """
    Mengubah ukuran gambar menjadi 224x224, normalisasi, dan menyesuaikan dimensi.
    Fungsi ini sekarang menerima gambar yang sudah dijamin RGB.
    """
    # Resize gambar
    img = pil_img.resize((224, 224))
    # Konversi ke array NumPy dan normalisasi piksel ke rentang [0, 1]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Tambahkan dimensi batch (dari (224, 224, 3) menjadi (1, 224, 224, 3))
    arr = np.expand_dims(arr, axis=0)
    
    print(f"🔍 DEBUG - Preprocessed image shape: {arr.shape}") # Untuk debugging di log
    return arr

def predict(model, pil_img: Image.Image):
    """
    Melakukan prediksi menggunakan model pada gambar yang sudah diproses.
    """
    # Pra-pemrosesan gambar
    processed_image = preprocess_image(pil_img)
    
    # Lakukan prediksi (verbose=0 agar tidak mencetak log prediksi ke konsol)
    predictions = model.predict(processed_image, verbose=0)[0]
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_index = int(np.argmax(predictions))
    
    # Dapatkan label kelas dan tingkat keyakinan
    label = CLASS_LABELS[predicted_index]
    confidence = float(predictions[predicted_index] * 100.0)
    
    return label, confidence

# --- Antarmuka Pengguna (UI) Streamlit ---

# Judul Aplikasi
st.markdown('<h2 style="text-align:center;color:#4a7c59;">🌱 Klasifikasi Tanaman Hias</h2>', unsafe_allow_html=True)

# Komponen untuk upload file
uploaded_file = st.file_uploader(
    "Upload Foto Tanaman Hias",
    type=["jpg", "jpeg", "png"]
)

# Cek apakah pengguna sudah mengupload file
if uploaded_file is None:
    st.info("Silakan upload sebuah gambar untuk memulai analisis.")
    st.stop()

# ======================================================================
# BAGIAN UTAMA PERBAIKAN: Konversi Gambar Dilakukan di Muka
# ======================================================================

# 1. Buka gambar dari file yang di-upload menggunakan PIL
raw_image = Image.open(uploaded_file)

# 2. SEGERA konversi gambar ke format RGB. Ini adalah langkah kunci.
#    Langkah ini memastikan semua gambar (termasuk PNG atau grayscale)
#    memiliki 3 channel warna yang konsisten sebelum diproses lebih lanjut.
rgb_image = raw_image.convert("RGB")

# 3. Tampilkan gambar yang sudah pasti RGB ke pengguna sebagai preview
st.image(rgb_image, caption="📷 Gambar yang Di-upload", width=320)

# ======================================================================

# Memuat model AI (lazy loading, hanya dimuat saat dibutuhkan)
download_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# Proses prediksi saat pengguna meng-upload gambar
with st.spinner("🔍 Menganalisis gambar..."):
    # Gunakan gambar yang sudah dikonversi (rgb_image) untuk prediksi
    predicted_label, prediction_confidence = predict(model, rgb_image)
    
    # Tampilkan hasil prediksi
    st.success(f"🌿 Jenis Tanaman: **{predicted_label}**")
    st.info(f"🔎 Tingkat Keyakinan: **{prediction_confidence:.2f}%**")