import uuid
import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, EmailStr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create FastAPI instance
app = FastAPI(title="ML Bot Prediction API",
              description="API for predicting whether a user is a bot based on usage metrics",
              version="1.0.0")

# User input schema
class UserInput(BaseModel):
    name: str
    email_id: EmailStr
    is_glogin: bool
    gender_female: int  # One-hot encoding for female
    gender_male: int    # One-hot encoding for male
    follower_count: int
    following_count: int
    dataset_count: int
    code_count: int
    discussion_count: int
    avg_nb_read_time_min: float
    total_votes_gave_nb: int
    total_votes_gave_ds: int
    total_votes_gave_dc: int
    model_choice: str = "random_forest"  # Default model choice

# Load trained models
models = {
    "random_forest": joblib.load('../../model/model_rf_tuned.pkl'),
    "bagging": joblib.load('../../model/model_bagging_tuned.pkl')
}

# Normalization parameters
scaler_mean = {
    'TOTAL_VOTES_GAVE_NB': 17.535584640490224,
    'TOTAL_VOTES_GAVE_DS': 6.530441542006134,
    'TOTAL_VOTES_GAVE_DC': 1.5298814400372998,
    'DATASET_COUNT': 2.529110164488324,
    'FOLLOWING_COUNT': 44.69164343000391,
    'FOLLOWER_COUNT': 26.80728556420434,
    'CODE_COUNT': 10.36182435807773,
    'AVG_NB_READ_TIME_MIN': 12.715438521996866,
    'DISCUSSION_COUNT': 65.79288564534343
}

scaler_std = {
    'TOTAL_VOTES_GAVE_NB': 4.475612457559885,
    'TOTAL_VOTES_GAVE_DS': 2.225461628724728,
    'TOTAL_VOTES_GAVE_DC': 1.0909238003275576,
    'DATASET_COUNT': 2.4280590832682174,
    'FOLLOWING_COUNT': 38.3139331407359,
    'FOLLOWER_COUNT': 22.329235354928624,
    'CODE_COUNT': 8.001625654143362,
    'AVG_NB_READ_TIME_MIN': 9.277711153987854,
    'DISCUSSION_COUNT': 46.123942659824834
}

# Columns in order used during model training
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

# Preprocessing pipeline
def preprocess_pipeline():
    numeric_features = [
        'FOLLOWER_COUNT',
        'FOLLOWING_COUNT',
        'DATASET_COUNT',
        'CODE_COUNT',
        'DISCUSSION_COUNT',
        'AVG_NB_READ_TIME_MIN',
        'TOTAL_VOTES_GAVE_NB',
        'TOTAL_VOTES_GAVE_DS',
        'TOTAL_VOTES_GAVE_DC'
    ]

    categorical_boolean_features = [
        'IS_GLOGIN',
        'GENDER_Female',
        'GENDER_Male'
    ]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('passthrough', 'passthrough')  # Already in numeric form (0/1)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_boolean_features)
    ])

    return preprocessor

# Manual normalization function
def normalize_data(data):
    cols_to_scale = [
        'TOTAL_VOTES_GAVE_NB', 'TOTAL_VOTES_GAVE_DS', 'TOTAL_VOTES_GAVE_DC',
        'DATASET_COUNT', 'FOLLOWING_COUNT', 'FOLLOWER_COUNT',
        'CODE_COUNT', 'AVG_NB_READ_TIME_MIN', 'DISCUSSION_COUNT'
    ]
    
    for col in cols_to_scale:
        if col in data.columns:
            data[col] = (data[col] - scaler_mean[col]) / scaler_std[col]
    return data

@app.post("/predict/", summary="Predict whether a user is a bot based on usage metrics")
async def predict(user_input: UserInput):
    try:
        # Prepare data in correct format and column order
        data = {
            'IS_GLOGIN': int(user_input.is_glogin),
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

        df = pd.DataFrame([data])
        
        # Select model
        model_choice = user_input.model_choice
        if model_choice not in models:
            raise HTTPException(status_code=400, detail=f"Model '{model_choice}' not available")
        
        selected_model = models[model_choice]
        
        # Process data based on model requirements
        if hasattr(selected_model, 'feature_names_in_'):
            # For scikit-learn models with feature_names_in_ attribute
            df_ordered = df[column_order]
            processed_data = preprocess_pipeline().fit_transform(df_ordered)
            prediction = selected_model.predict(processed_data)
            
            # Calculate probabilities if available
            probabilities = None
            if hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba(processed_data)
                bot_probability = float(probabilities[0][1])  # Probability of class 1 (bot)
            else:
                bot_probability = None
                
            # Get feature importance if available
            feature_importance = None
            if hasattr(selected_model, 'feature_importances_'):
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(column_order, selected_model.feature_importances_)
                }
                
            result = {
                "prediction": int(prediction[0]),
                "model_used": model_choice,
                "user_name": user_input.name,
                "user_email": user_input.email_id
            }
            
            if bot_probability is not None:
                result["probability"] = bot_probability
                
            if feature_importance is not None:
                result["feature_importance"] = feature_importance
                
            return result
        else:
            # Fallback to manual normalization method
            normalized_data = normalize_data(df)
            prediction = selected_model.predict(normalized_data)
            
            return {
                "prediction": int(prediction[0]),
                "model_used": model_choice,
                "user_name": user_input.name,
                "user_email": user_input.email_id
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction process: {str(e)}")

@app.get("/models/", summary="Get list of available models")
async def get_models():
    return {"available_models": list(models.keys())}

@app.get("/health", summary="Check API health status")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)