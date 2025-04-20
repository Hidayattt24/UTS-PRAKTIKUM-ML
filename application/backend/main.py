import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI instance
app = FastAPI(
    title="Bot Detection API",
    description="API for predicting whether a user is a bot based on their activity metrics",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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

# Model paths - update these with your actual model paths
MODEL_DIR = "../../model"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_rf_tuned.pkl")
BAGGING_MODEL_PATH = os.path.join(MODEL_DIR, "model_bagging_tuned.pkl")

# Load models
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    bagging_model = joblib.load(BAGGING_MODEL_PATH)
    
    # Dictionary of available models
    models = {
        "random_forest": rf_model,
        "bagging": bagging_model
    }
except Exception as e:
    # Use placeholder models for testing if real models are not available
    # In a production environment, you would handle this differently
    print(f"Error loading models: {str(e)}")
    print("Using placeholder models for testing purposes")
    
    # Create dummy models (only for testing - remove in production)
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    models = {
        "random_forest": RandomForestClassifier(),
        "bagging": BaggingClassifier()
    }

# Order of features expected by the models
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

# Define preprocessing pipeline
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

    categorical_features = [
        'IS_GLOGIN',
        'GENDER_Female',
        'GENDER_Male'
    ]

    # Standard scaler for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pass-through for categorical features (already binary)
    categorical_transformer = Pipeline(steps=[
        ('passthrough', 'passthrough')
    ])

    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

@app.get("/health", summary="Check API health status")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.get("/models/", summary="Get list of available models")
async def get_models():
    """Get the list of available prediction models."""
    return {"available_models": list(models.keys())}

@app.post("/predict/", summary="Predict if a user is a bot based on their activity metrics")
async def predict(user_input: UserInput):
    """
    Predict whether a user is a bot based on their activity metrics.
    
    The prediction is made using the selected model (random_forest or bagging).
    """
    try:
        # Prepare input data with the correct column format
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
        
        # Create DataFrame with the correct column order
        df = pd.DataFrame([data])[column_order]
        
        # Select the requested model
        model_choice = user_input.model_choice
        if model_choice not in models:
            raise HTTPException(status_code=400, detail=f"Model '{model_choice}' not available")
        
        selected_model = models[model_choice]
        
        # Apply preprocessing
        preprocessor = preprocess_pipeline()
        processed_data = preprocessor.fit_transform(df)
        
        # Make prediction
        prediction = selected_model.predict(processed_data)
        
        # Get prediction probability if available
        probability = None
        if hasattr(selected_model, 'predict_proba'):
            probabilities = selected_model.predict_proba(processed_data)
            probability = float(probabilities[0][1])  # Probability of class 1 (bot)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(selected_model, 'feature_importances_'):
            feature_importance = {
                feature: float(importance)
                for feature, importance in zip(column_order, selected_model.feature_importances_)
            }
        
        # Build response
        result = {
            "prediction": int(prediction[0]),
            "model_used": model_choice,
            "user_name": user_input.name,
            "user_email": user_input.email_id,
        }
        
        if probability is not None:
            result["probability"] = probability
        
        if feature_importance is not None:
            result["feature_importance"] = feature_importance
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)