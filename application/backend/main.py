import uuid
import uvicorn
import joblib
from fastapi import File
from fastapi import UploadFile
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import EmailStr
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# mmebuat instance FastAPI
app = FastAPI()

# mendefinisikan schema input 
class UserInput(BaseModel):
    ... ### lengkapi dengan atribut-atribut yang dibutuhkan


def preprocess_pipeline():
    numeric_features = [...]  # masukkan kolom-kolom numerik

    categorical_boolean_features = [...] # masukkan kolom kategorikal dan boolean

    numeric_transformer = Pipeline(steps=[
        ...  ## lengkapi dengan pipeline yang dibutuhkan
    ])

    categorical_boolean_transfomer = Pipeline(steps=[
        ... ## lengkapi dengan pipeline yang dibutuhkan
    ])

    preprocessor = ColumnTransformer(
        ... ## lengkapi dengan transformer yang dibutuhkan
    )

    return preprocessor

# load model
model = joblib.load('../../model/model.pkl')    ## SESUAIKAN DENGAN LOKASI MODEL YANG ANDA GUNAKAN

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    # Ubah input menjadi format yang sesuai (pandas DataFrame)
    data = pd.DataFrame(...)  ## lengkapi dengan data yang akan diproses

    # Terapkan pipeline untuk preprocessing
    preprocessing_pipeline = preprocess_pipeline()
    processed_data = preprocessing_pipeline.fit_transform(data) 
    
    # Prediksi dengan model yang sudah dilatih
    prediction = ... ## lengkapi kode untuk melakukan prediksi
    
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)