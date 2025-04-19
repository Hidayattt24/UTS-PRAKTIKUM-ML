from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import joblib
import pandas as pd

app = FastAPI()

# Skema input dari pengguna
class UserInput(BaseModel):
    name: str
    gender_female: int  # One-hot encoding untuk female
    gender_male: int    # One-hot encoding untuk male
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

# Load model yang telah dilatih
rf_model = joblib.load('../../model/model_rf_tuned.pkl')
bagging_model = joblib.load('../../model/model_bagging_tuned.pkl')

# Dictionary model untuk akses model berdasarkan nama
models = {
    'random_forest': rf_model,
    'bagging': bagging_model
}

# Kolom yang sesuai dengan urutan saat pelatihan model
column_order = [
    'IS_GLOGIN',
    'FOLLOWER_COUNT',
    'FOLLOWING_COUNT',
    'DATASET_COUNT',
    'CODE_COUNT',
    'DISCUSSION_COUNT',
    'AVG_NB_READ_TIME_MIN',
    'TOTAL_VOTES_GAVE_NB',
    'TOTAL_VOTES_GAVE_DS',
    'TOTAL_VOTES_GAVE_DC',
    'GENDER_Female',
    'GENDER_Male'
]

@app.post("/predict/")
async def predict(user_input: UserInput):
    # Menyusun data dengan kolom yang sesuai urutan pelatihan
    data = {
        'IS_GLOGIN': user_input.is_glogin,
        'FOLLOWER_COUNT': user_input.follower_count,
        'FOLLOWING_COUNT': user_input.following_count,
        'DATASET_COUNT': user_input.dataset_count,
        'CODE_COUNT': user_input.code_count,
        'DISCUSSION_COUNT': user_input.discussion_count,
        'AVG_NB_READ_TIME_MIN': user_input.avg_nb_read_time_min,
        'TOTAL_VOTES_GAVE_NB': user_input.total_votes_gave_nb,
        'TOTAL_VOTES_GAVE_DS': user_input.total_votes_gave_ds,
        'TOTAL_VOTES_GAVE_DC': user_input.total_votes_gave_dc,
        'GENDER_Female': user_input.gender_female,
        'GENDER_Male': user_input.gender_male
    }

    df = pd.DataFrame([data])[column_order]

    # Pilih model berdasarkan input
    model_choice = user_input.model_choice
    if model_choice not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_choice}' tidak tersedia")

    selected_model = models[model_choice]

    try:
        # Prediksi menggunakan model
        prediction = selected_model.predict(df)
        return {"prediction": int(prediction[0]), "model_used": model_choice}
    except Exception as e:
        error_message = f"Error prediksi: {str(e)}"
        raise HTTPException(status_code=400, detail=error_message)

@app.get("/models/")
async def get_models():
    return {"available_models": list(models.keys())}
