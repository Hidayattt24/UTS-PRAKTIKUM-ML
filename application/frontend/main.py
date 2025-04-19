# Frontend - Streamlit

import requests
import streamlit as st
import pandas as pd
import numpy as np

# mendefinisikan header
st.title('Bot or Not?')

# Sidebar untuk memilih model
st.sidebar.title("Model Selection")

# Dapatkan daftar model yang tersedia dari backend
try:
    models_response = requests.get('http://localhost:8000/models/')
    if models_response.status_code == 200:
        available_models = models_response.json().get('available_models', [])
        model_choice = st.sidebar.selectbox(
            'Pilih model untuk prediksi:',
            available_models,
            format_func=lambda x: x.replace('_', ' ').title()  # Format tampilan
        )
    else:
        st.sidebar.error("Tidak dapat mendapatkan daftar model")
        model_choice = "random_forest"  # Default jika gagal mendapatkan daftar
except requests.exceptions.ConnectionError:
    st.sidebar.error('❌ Tidak dapat terhubung ke server backend.')
    model_choice = "random_forest"  # Default jika server tidak berjalan

# Informasi tentang model
st.sidebar.markdown("### Informasi Model")
if model_choice == "random_forest":
    st.sidebar.info("""
    **Random Forest** adalah model ensemble yang menggabungkan banyak decision tree untuk menghasilkan prediksi yang lebih akurat dan stabil.
    """)
elif model_choice == "bagging":
    st.sidebar.info("""
    **Bagging (Bootstrap Aggregating)** adalah teknik ensemble yang mengurangi variance dan membantu mencegah overfitting dengan mengambil sampel dari data training dengan penggantian.
    """)

# menampilkan form input
with st.form(key='user_input_form'):
    # Form untuk Nama
    name = st.text_input('Nama user', 'name')

    # Form untuk Gender (pilihan antara Male atau Female)
    gender = st.selectbox('Jenis kelamin', ['Male', 'Female'])

    # Form untuk Email (harus menggunakan email yang valid)
    email_id = st.text_input('Alamat email user', 'email@example.com')

    # Form untuk Google Login (True atau False)
    is_glogin = st.checkbox('Apakah akun menggunakan google login untuk register akun atau tidak', value=True)

    # Form untuk Jumlah Follower
    follower_count = st.number_input('Jumlah follower', min_value=0)

    # Form untuk Jumlah Following
    following_count = st.number_input('Jumlah following', min_value=0)

    # Form untuk Jumlah Dataset yang Dibuat
    dataset_count = st.number_input('Jumlah dataset yang dimiliki', min_value=0)

    # Form untuk Jumlah Notebooks yang Dibuat
    code_count = st.number_input('Jumlah notebook kode yang dimiliki', min_value=0)

    # Form untuk Jumlah Diskusi yang Dikutip
    discussion_count = st.number_input('Jumlah diskusi yang pernah diikuti', min_value=0)

    # Form untuk Rata-rata Waktu Membaca Notebook dalam menit
    avg_nb_read_time_min = st.number_input('Rata-rata waktu yang dihabiskan untuk menggunakan notebook kaggle (dalam menit)', min_value=0.0)

    # Form untuk Total Vote yang Diberikan pada Notebook
    total_votes_gave_nb = st.number_input('Total jumlah vote yang pernah diberikan pada sebuah notebook', min_value=0)

    # Form untuk Total Vote yang Diberikan pada Dataset
    total_votes_gave_ds = st.number_input('Total jumlah vote yang pernah diberikan pada sebuah dataset', min_value=0)

    # Form untuk Total Vote yang Diberikan pada Diskusi
    total_votes_gave_dc = st.number_input('Total jumlah vote yang pernah diberikan pada sebuah discussion', min_value=0)

    # Tombol submit untuk form
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Mengubah gender menjadi kolom one-hot encoding
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0

        # Lengkapi dictionary yang akan dikirim ke backend API
        user_input = {
            "name": name,
            "gender_female": gender_female,
            "gender_male": gender_male, 
            "email_id": email_id,
            "is_glogin": is_glogin,
            "follower_count": follower_count,
            "following_count": following_count,
            "dataset_count": dataset_count,
            "code_count": code_count,
            "discussion_count": discussion_count,
            "avg_nb_read_time_min": avg_nb_read_time_min,
            "total_votes_gave_nb": total_votes_gave_nb,
            "total_votes_gave_ds": total_votes_gave_ds,
            "total_votes_gave_dc": total_votes_gave_dc,
            "model_choice": model_choice  # Mengirim pilihan model ke backend
        }

        try:
            # Kirim data ke backend untuk prediksi
            response = requests.post('http://localhost:8000/predict/', json=user_input)

            # Menampilkan hasil prediksi
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction')
                model_used = result.get('model_used', model_choice)

                st.write(f"Model yang digunakan: **{model_used.replace('_', ' ').title()}**")
                
                if prediction == 1:
                    st.error('🛑 User terdeteksi BOT.')
                else:
                    st.success('✅ User tidak terdeteksi BOT.')
            else:
                error_message = "Unknown error"
                try:
                    error_detail = response.json()
                    error_message = error_detail.get('detail', error_message)
                except:
                    pass
                st.error(f'Terjadi kesalahan dari server: {error_message}')
        except requests.exceptions.ConnectionError:
            st.error('❌ Tidak dapat terhubung ke server backend. Pastikan FastAPI sudah berjalan.')