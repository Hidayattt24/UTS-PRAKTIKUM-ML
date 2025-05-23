
# 🤖 Kaggle Bot Account Detection Dataset

## 📝 Dataset Description
Dataset ini **mensimulasikan perilaku pengguna Kaggle** untuk mendeteksi **akun bot**. Meskipun bukan berasal dari data dunia nyata, data ini dibuat menggunakan **library Faker** yang menghasilkan data sintetis menyerupai aktivitas pengguna sebenarnya. Tujuan utama dari dataset ini adalah untuk menyajikan skenario realistis mengenai bagaimana akun bot bisa berperilaku dalam kompetisi data science, termasuk aktivitas seperti voting, komentar, dan interaksi lainnya. Hal ini sangat relevan untuk menjaga **integritas kompetisi**, **evaluasi yang adil**, dan **kepercayaan komunitas** di platform seperti Kaggle.


## 🎯 Problem Context

- **Competition Integrity**: Bots dapat memanipulasi sistem voting.
- **Fair Evaluation**: Menjamin keterlibatan asli di notebook dan diskusi.
- **Community Trust**: Menjaga interaksi pengguna yang autentik.

---

## 📁 Project Structure

```
UTS/
├── application/
│   ├── backend/
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── main.py
│       └── requirements.txt
├── dataset/
│   └── kaggle_bot_accounts.csv
├── model/
│   ├── model_rf_tuned.pkl
│   └── model_bagging_tuned.pkl
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8 atau lebih tinggi
- Git
- [Kaggle API](https://www.kaggle.com/docs/api) (opsional, jika ingin unduh otomatis)

---

### 🔧 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/kaggle-bot-detection.git
   cd kaggle-bot-detection
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv env
   pip install virtualenv

   # Aktifkan environment
   # Windows:
   env\Scripts\activate
   # macOS/Linux:
   source env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Running the Application

### ▶️ Start Backend API (FastAPI)

```bash
cd application/backend
uvicorn main:app --reload
```

📡 Akses API di: `http://localhost:8000`  
🧾 Dokumentasi API: `http://localhost:8000/docs`

---

### ▶️ Start Frontend Application (Streamlit)

Buka terminal baru:

```bash
cd application/frontend
streamlit run main.py
```

🌐 Frontend tersedia di: `http://localhost:8501`

---

## 📥 Dataset Setup

### 1. **Download Dataset**

Kunjungi: [Kaggle Bot Account Detection Dataset](https://www.kaggle.com/datasets/shriyashjagtap/kaggle-bot-account-detection)  
Klik tombol **Download**, dan simpan file ZIP ke komputer kamu.

### 2. **Extract Dataset ke Folder `data/`**

```bash
# Buat folder data
mkdir data
cd data

# Pindahkan file ZIP yang telah diunduh (Windows)
move "%USERPROFILE%\Downloads\kaggle-bot-account-detection.zip" .

# Ekstrak file (Windows)
tar -xf kaggle-bot-account-detection.zip

# Atau (Linux/Mac)
unzip kaggle-bot-account-detection.zip
```

Setelah ekstraksi, file `kaggle_bot_accounts.csv` akan tersedia di dalam folder `data/`.

---

## 🛠️ Technologies Used

- **FastAPI** – Backend API framework
- **Streamlit** – Frontend web app framework
- **Scikit-learn** – Machine Learning models
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations

---

## 📊 Model Performance

| Model              | Accuracy |
| ------------------ | -------- |
| Random Forest      | 98.41%   |
| Bagging Classifier | 98.41%   |

---

## 📊 Dataset Features

- **Total Records**: 1,321,188
- **File Format**: CSV
- **Size**: ~150MB

---

## 🗂️ Project Structure After Setup

```
project/
├── data/
│   ├── kaggle-bot-account-detection.zip
│   └── kaggle_bot_accounts.csv
├── application/
│   ├── backend/
│   └── frontend/
└── README.md
```

---

## 👥 Authors

- Maulana Fikri
- Hidayat Nur Hakim
- M. Syahidal Akbar Zas

---

## 🙏 Acknowledgments

- Terima kasih untuk **Kaggle** atas inspirasinya
