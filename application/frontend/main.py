import requests
import streamlit as st
import pandas as pd
import numpy as np

# mendefinisikan header
st.title('Bot or Not?')

# menampilkan form input
with st.form(key='user_input_form'):
    # Form untuk Nama
    name = st.text_input('Nama user', 'name')

    # Form untuk Gender (pilihan antara Male atau Female)
    gender = st.selectbox('Jenis kelamin', ['Male', 'Female'])

    # Form untuk Email (harus menggunakan email yang valid)
    email_id = st.text_input('Alamat email user', 'email')

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
        # Ketika submit, siapkan data dalam format yang akan dikirim ke backend API
        user_input = {...}  ## lengkapi dengan data yang akan dikirim ke backend
        
        # Kirim data ke backend untuk prediksi
        # Misalnya, Anda bisa menggunakan requests untuk mengirim data ke API FastAPI
        response = requests.post('http://localhost:8000/predict/', json=user_input)
        
        # Menampilkan hasil prediksi
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            
            if prediction == 1:
                st.write('User terdeteksi BOT.')
            else:
                st.write('User tidak terdeteksi BOT.')
        else:
            st.write('Terjadi kesalahan dalam prediksi.')