# Backend - FastAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = FastAPI()

# Skema input dari pengguna
class UserInput(BaseModel):
    name: str
    gender_female: int  # One-hot encoding untuk female
    gender_male: int  # One-hot encoding untuk male
    email_id: EmailStr
    is_glogin: bool
    follower_count: int
    following_count: int
    dataset_count: int
    code_count: int
    discussion_count: int
    avg_nb_read_time_min: float
    total_votes_gave_nb: int
    total_votes_gave_ds: int
    total_votes_gave_dc: int
    model_choice: str  # Untuk memilih model yang akan digunakan

# Fungsi preprocessing pipeline
def preprocess_pipeline():
    numeric_features = [
        'follower_count', 'following_count', 'dataset_count',
        'code_count', 'discussion_count', 'avg_nb_read_time_min',
        'total_votes_gave_nb', 'total_votes_gave_ds', 'total_votes_gave_dc'
    ]
    
    categorical_boolean_features = ['is_glogin']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_boolean_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_boolean_transformer, categorical_boolean_features)
    ])

    return preprocessor

# Load model yang telah dilatih
rf_model = joblib.load('../../model/model_rf_tuned.pkl')
bagging_model = joblib.load('../../model/model_bagging_tuned.pkl')

# Dictionary model untuk akses model berdasarkan nama
models = {
    'random_forest': rf_model,
    'bagging': bagging_model
}

@app.post("/predict/")
async def predict(user_input: UserInput):
    # Mengubah input menjadi format yang benar
    data = {
        'name': user_input.name,
        'gender_female': user_input.gender_female,
        'gender_male': user_input.gender_male,
        'is_glogin': user_input.is_glogin,
        'follower_count': user_input.follower_count,
        'following_count': user_input.following_count,
        'dataset_count': user_input.dataset_count,
        'code_count': user_input.code_count,
        'discussion_count': user_input.discussion_count,
        'avg_nb_read_time_min': user_input.avg_nb_read_time_min,
        'total_votes_gave_nb': user_input.total_votes_gave_nb,
        'total_votes_gave_ds': user_input.total_votes_gave_ds,
        'total_votes_gave_dc': user_input.total_votes_gave_dc
    }
    
    # Mengubah menjadi DataFrame
    df = pd.DataFrame([data])
    
    # Hapus kolom name karena tidak digunakan untuk prediksi
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    # Pilih model berdasarkan pilihan pengguna
    model_choice = user_input.model_choice
    if model_choice not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_choice}' tidak tersedia")
    
    selected_model = models[model_choice]
    
    try:
        # Prediksi langsung menggunakan model yang sudah terlatih
        prediction = selected_model.predict(df)
        return {"prediction": int(prediction[0]), "model_used": model_choice}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error prediksi: {str(e)}")

# Endpoint untuk mendapatkan daftar model yang tersedia
@app.get("/models/")
async def get_models():
    return {"available_models": list(models.keys())}