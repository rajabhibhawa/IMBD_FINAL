import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import time

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FIX: Custom Embedding layer untuk mengatasi konflik versi
# Keras (menangani parameter 'quantization_config' yang
# tidak dikenali oleh versi Keras yang lebih lama)
# =========================================================
class CompatibleEmbedding(Embedding):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return cls(**config)


@st.cache_resource
def load_keras_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={'Embedding': CompatibleEmbedding}
    )


class_names = ['Negative', 'Positive']

model_paths = {
    "GRU - Config 1": "GRU_Config1_SeqB.h5",
    "GRU - Config 2": "best_model_imdb_GRU_Config2_SeqB.h5"
}

MAX_LEN = 250


@st.cache_resource
def load_tokenizer(path="tokenizer.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def encode_review(text, max_len):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded


with st.sidebar:
    st.header("Pengaturan")
    model_choice = st.selectbox("Pilih Model", list(model_paths.keys()))
    st.write("---")
    st.header("Tentang Model")
    st.write(f"Model aktif: **{model_choice}**")
    st.write("Dataset: **IMDB Movie Reviews**")


def predict(padded_review):
    model = load_keras_model(model_paths[model_choice])
    start = time.time()
    prediction = model.predict(padded_review, verbose=0)
    elapsed = (time.time() - start) * 1000
    return prediction[0][0], elapsed


if 'history' not in st.session_state:
    st.session_state.history = []


tab1, tab2 = st.tabs(["🔍 Prediksi", "📊 Tentang Model"])

with tab1:
    st.title("Klasifikasi Sentimen Review Film IMDB")
    st.write(f"Menggunakan model: **{model_choice}**")

    user_review = st.text_area(
        "Tulis review film (dalam Bahasa Inggris)",
        height=150,
        placeholder="Contoh: This movie was absolutely fantastic, great acting and story..."
    )

    analyze_clicked = st.button("Analisis Sentimen", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_review.strip():
            st.warning("Silakan masukkan review terlebih dahulu.")
        else:
            try:
                padded_review = encode_review(user_review, MAX_LEN)

                with st.spinner('Menganalisis review...'):
                    raw_score, inference_time = predict(padded_review)

                raw_score = float(raw_score)
                predicted_class = class_names[int(raw_score > 0.5)]
                confidence = raw_score if raw_score > 0.5 else 1 - raw_score
                prob_positive = raw_score
                prob_negative = 1 - raw_score

                emoji = "😊" if predicted_class == "Positive" else "😞"

                col_left, col_right = st.columns([1, 2])

                with col_left:
                    st.markdown(f"""
                    <div style="width: 100%; height: 220px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background-color: var(--secondary-background-color, #E8EAED); display: flex; align-items: center; justify-content: center;">
                        <p style="font-size: 90px; margin: 0;">{emoji}</p>
                    </div>
                    <p style="text-align: center; color: #666; font-size: 13px; margin-top: 6px;">Review dianalisis</p>
                    """, unsafe_allow_html=True)

                with col_right:
                    st.markdown(f"""
                    <div style="background-color: var(--secondary-background-color, #E8EAED); border-radius: 12px; padding: 20px; border: 1px solid #E0E0E0; height: 220px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                        <p style="color: #666; margin: 0; font-size: 18px;">Prediksi</p>
                        <h2 style="margin: 8px 0; font-size: 36px;">{predicted_class} {emoji}</h2>
                        <p style="color: #666; margin: 0; font-size: 18px;">Confidence: {confidence*100:.1f}%</p>
                        <p style="color: #999; margin: 4px 0 0 0; font-size: 13px;">⏱️ Inference: {inference_time:.2f} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")

                if confidence > 0.8:
                    st.success(f"Model cukup yakin: **{predicted_class}** ({confidence*100:.1f}%)")
                    st.progress(float(confidence))
                elif confidence > 0.5:
                    st.warning(f"Model kurang yakin: **{predicted_class}** ({confidence*100:.1f}%)")
                    st.progress(float(confidence))

                st.write("### Perbandingan Probabilitas")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(label="😞 Negative", value=f"{prob_negative*100:.1f}%")
                with col_b:
                    st.metric(label="😊 Positive", value=f"{prob_positive*100:.1f}%")

                st.session_state.history.append({
                    "Model": model_choice,
                    "Review (potongan)": user_review[:40] + ("..." if len(user_review) > 40 else ""),
                    "Prediksi": predicted_class,
                    "Confidence": f"{confidence*100:.1f}%",
                    "Waktu (ms)": f"{inference_time:.2f}"
                })

            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

    if st.session_state.history:
        st.write("### Riwayat Prediksi")
        st.table(st.session_state.history)

        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()

with tab2:
    st.header("Tentang Model")
    st.write(f"**Model aktif:** {model_choice}")
    st.write("**Kelas:** Negative, Positive")
    st.write("**Dataset:** IMDB Movie Reviews (50K, Kaggle)")
    st.write("---")
    if model_choice == "GRU - Config 1":
        st.write("Model GRU dengan konfigurasi 1 (arsitektur dasar), dilatih pada sequence review IMDB.")
    else:
        st.write("Model GRU dengan konfigurasi 2 (hasil tuning tambahan), merupakan model terbaik pada eksperimen.")