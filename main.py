import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Fungsi load dan compile ulang model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('64B30E-ENB0-TanamanHias-v3.keras', compile=False)  # ganti sesuai nama filemu
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),         # optimizer default
    #     loss='categorical_crossentropy',              # sesuaikan jika sparse_categorical_crossentropy
    #     metrics=['accuracy']
    # )
    return model

model = load_model()

# Label klasifikasi
# class_labels = [
#     "Aglaonema", "Alyssum", "Aster", "Azalea", "Bergamot", "Cosmos", "Dahlia",
#     "Daisy", "Dandelion", "Dieffenbacia", "Euphorbia", "Eustoma", "Gerbera",
#     "Iris", "Ixora", "Jasmine", "Lavender", "Lily Flower", "Orchid", "Pansy",
#     "Peony", "Polyanthus", "Rose", "Sage", "Snapdragon", "Sunflower",
#     "Tuberose", "Tulip", "Viola"
# ]

class_labels = [
    "Aglaonema", "Daisy", "Dandelion", "Jasmine", "Lavender", 
    "Lily Flower","Rose", "Sunflower", "Tulip"
]
# CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        color: #4a7c59;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .upload-area {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<div class="title">🌱 Klasifikasi Tanaman Hias</div>', unsafe_allow_html=True)

# Upload gambar
# st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Foto Tanaman Hias", type=['jpg', 'jpeg', 'png'])
# st.markdown('</div>', unsafe_allow_html=True)

# Fungsi untuk preprocess gambar
def preprocess_image(image):
    img = image.convert('RGB')                    
    img = img.resize((224, 224))                   
    img_array = np.array(img).astype(np.float32)   
    img_array = img_array / 255.0                  # Normalisasi pixel 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    return img_array

# Fungsi analisis gambar
def analyze_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    top_label = class_labels[top_idx]
    confidence = predictions[top_idx] * 100
    return top_label, confidence

# Kalau ada gambar di-upload
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Preview Gambar", width=300)

    prediction_placeholder = st.empty()

    # Analisis gambar saat upload
    with st.spinner('🔍 Menganalisis gambar...'):
        top_label, confidence = analyze_image(image)
        with prediction_placeholder.container():
            st.success(f"🌿 **Jenis Tanaman:** {top_label}")
            st.info(f"🔎 **Tingkat Keyakinan:** {confidence:.2f}%")

    # Tombol Analisis Ulang
    if st.button('🔄 Analisis Ulang'):
        with st.spinner('🔍 Menganalisis ulang...'):
            top_label, confidence = analyze_image(image)
            with prediction_placeholder.container():
                st.success(f"🌿 **Jenis Tanaman:** {top_label}")
                st.info(f"🔎 **Tingkat Keyakinan:** {confidence:.2f}%")
