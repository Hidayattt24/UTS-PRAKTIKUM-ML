import requests
import streamlit as st
import pandas as pd
import numpy as np
import json
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Bot or Not? Detector",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .feature-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🤖 Bot or Not?</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict whether a user is likely a bot based on their activity metrics</p>", unsafe_allow_html=True)

# Backend API URL - change this to your actual API endpoint
API_URL = "http://localhost:8000"

# Check backend connection
with st.sidebar:
    st.title("Server Status")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend server is connected")
            
            # Display available models
            try:
                models_response = requests.get(f"{API_URL}/models/")
                if models_response.status_code == 200:
                    available_models = models_response.json().get("available_models", [])
                    model_choice = st.selectbox(
                        "Select prediction model:",
                        available_models,
                        format_func=lambda x: x.replace("_", " ").title()
                    )
                else:
                    st.warning("⚠️ Could not fetch model list")
                    model_choice = "random_forest"  # Default
            except:
                st.warning("⚠️ Could not fetch model list")
                model_choice = "random_forest"  # Default
        else:
            st.error("❌ Backend server is not healthy")
            model_choice = "random_forest"  # Default
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to backend server")
        model_choice = "random_forest"  # Default
        st.warning("Please make sure the backend server is running")

    # Model information
    st.title("Model Information")
    if model_choice == "random_forest":
        st.info("""
        **Random Forest** is an ensemble model that combines multiple decision trees to produce more accurate predictions.
        
        Features:
        - Handles non-linear relationships
        - Resistant to overfitting
        - Provides feature importance
        """)
    elif model_choice == "bagging":
        st.info("""
        **Bagging** (Bootstrap Aggregating) reduces variance by training models on random subsets of data.
        
        Features:
        - Reduces model variance
        - Improves stability
        - Works well with high-dimensional data
        """)

# Create input form with two columns
st.markdown("<h2 class='feature-header'>User Input Form</h2>", unsafe_allow_html=True)

with st.form(key="user_input_form"):
    col1, col2 = st.columns(2)
    
    # User identification (not used for prediction)
    with col1:
        st.subheader("User Information")
        name = st.text_input("Name", value="John Doe")
        email_id = st.text_input("Email address", value="user@example.com")
        gender = st.selectbox("Gender", ["Male", "Female"])
        is_glogin = st.checkbox("Uses Google Login", value=True)
    
    # User activity metrics (used for prediction)
    with col2:
        st.subheader("Activity Metrics")
        follower_count = st.number_input("Follower count", min_value=0, value=25)
        following_count = st.number_input("Following count", min_value=0, value=45)
        dataset_count = st.number_input("Dataset count", min_value=0, value=3)
        code_count = st.number_input("Code notebooks count", min_value=0, value=10)
        discussion_count = st.number_input("Discussion count", min_value=0, value=65)
    
    # Engagement metrics
    st.subheader("Engagement Metrics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_nb_read_time_min = st.number_input("Avg. notebook read time (min)", 
                                            min_value=0.0, value=12.5, step=0.5)
    with col4:
        total_votes_gave_nb = st.number_input("Total votes on notebooks", min_value=0, value=17)
        total_votes_gave_ds = st.number_input("Total votes on datasets", min_value=0, value=6)
    with col5:
        total_votes_gave_dc = st.number_input("Total votes on discussions", min_value=0, value=2)
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict", use_container_width=True)

# Handle prediction
if submit_button:
    with st.spinner("Processing prediction..."):
        # Prepare gender one-hot encoding
        gender_female = 1 if gender == "Female" else 0
        gender_male = 1 if gender == "Male" else 0
        
        # Prepare API request data
        user_input = {
            "name": name,
            "email_id": email_id,
            "is_glogin": is_glogin,
            "gender_female": gender_female,
            "gender_male": gender_male,
            "follower_count": follower_count,
            "following_count": following_count,
            "dataset_count": dataset_count,
            "code_count": code_count,
            "discussion_count": discussion_count,
            "avg_nb_read_time_min": avg_nb_read_time_min,
            "total_votes_gave_nb": total_votes_gave_nb,
            "total_votes_gave_ds": total_votes_gave_ds,
            "total_votes_gave_dc": total_votes_gave_dc,
            "model_choice": model_choice
        }
        
        try:
            # Make prediction request
            response = requests.post(f"{API_URL}/predict/", json=user_input)
            
            # Show prediction results
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                
                # Display results
                st.header("Prediction Results")
                
                # Show model used
                st.info(f"Model used: **{result.get('model_used', '').replace('_', ' ').title()}**")
                
                # Show prediction with styling
                if prediction == 1:
                    st.error("### ⚠️ BOT DETECTED")
                    st.markdown("""
                    <div style="padding: 15px; border-radius: 5px; background-color: #FFEBEE; border-left: 5px solid #F44336;">
                        <h3 style="color: #D32F2F; margin-top: 0;">This account shows bot-like behavior</h3>
                        <p>The model has determined that this user's activity patterns match those of automated accounts.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("### ✅ HUMAN USER")
                    st.markdown("""
                    <div style="padding: 15px; border-radius: 5px; background-color: #E8F5E9; border-left: 5px solid #4CAF50;">
                        <h3 style="color: #2E7D32; margin-top: 0;">This account shows human-like behavior</h3>
                        <p>The model has determined that this user's activity patterns match those of legitimate human users.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability if available
                if "probability" in result:
                    prob = result["probability"]
                    st.subheader("Prediction Probability")
                    
                    # Create columns for probability display
                    prob_col1, prob_col2 = st.columns([1, 3])
                    
                    with prob_col1:
                        st.metric("Bot Probability", f"{prob:.2%}")
                    
                    with prob_col2:
                        # Progress bar for probability
                        st.progress(prob)
                        
                        # Confidence level text
                        if prob >= 0.9:
                            conf_text = "Very high confidence"
                        elif prob >= 0.75:
                            conf_text = "High confidence"
                        elif prob >= 0.6:
                            conf_text = "Moderate confidence"
                        elif prob >= 0.4:
                            conf_text = "Low confidence"
                        else:
                            conf_text = "Very low confidence"
                        
                        st.caption(f"Confidence level: {conf_text}")
                
                # Show feature importance if available
                if "feature_importance" in result:
                    st.subheader("Feature Importance")
                    feature_imp = result["feature_importance"]
                    
                    # Process feature names for better display
                    processed_features = {}
                    for feature, importance in feature_imp.items():
                        # Clean up feature names for display
                        display_name = feature.replace("_", " ").title()
                        if display_name.startswith("Is "):
                            display_name = display_name.replace("Is ", "Uses ")
                        if display_name == "Avg Nb Read Time Min":
                            display_name = "Avg. Reading Time"
                        processed_features[display_name] = importance
                    
                    # Create DataFrame and sort by importance
                    feat_df = pd.DataFrame({
                        "Feature": list(processed_features.keys()),
                        "Importance": list(processed_features.values())
                    }).sort_values("Importance", ascending=False)
                    
                    # Display feature importance chart
                    st.bar_chart(feat_df.set_index("Feature"))
                    
                    # Display top features as text
                    top_features = feat_df.head(3)["Feature"].tolist()
                    st.info(f"Top influencing features: **{', '.join(top_features)}**")
                
                # Show input data summary
                with st.expander("View Input Data Summary"):
                    # Create separate DataFrames for each category
                    personal_df = pd.DataFrame({
                        "Field": ["Name", "Email", "Gender", "Google Login"],
                        "Value": [name, email_id, gender, "Yes" if is_glogin else "No"]
                    })
                    
                    activity_df = pd.DataFrame({
                        "Metric": ["Followers", "Following", "Datasets", "Code Notebooks", "Discussions"],
                        "Count": [follower_count, following_count, dataset_count, code_count, discussion_count]
                    })
                    
                    engagement_df = pd.DataFrame({
                        "Metric": ["Avg. Reading Time", "Notebook Votes", "Dataset Votes", "Discussion Votes"],
                        "Value": [
                            f"{avg_nb_read_time_min:.1f} min", 
                            total_votes_gave_nb, 
                            total_votes_gave_ds, 
                            total_votes_gave_dc
                        ]
                    })
                    
                    # Display tables
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Personal Information")
                        st.table(personal_df)
                    
                    with col2:
                        st.subheader("Activity Metrics")
                        st.table(activity_df)
                    
                    st.subheader("Engagement Metrics")
                    st.table(engagement_df)
            
            else:
                # Handle API error
                error_message = "Unknown error"
                try:
                    error_detail = response.json()
                    error_message = error_detail.get("detail", error_message)
                except:
                    pass
                st.error(f"Server error: {error_message}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server. Please ensure it is running.")

# Add explanatory information
with st.expander("ℹ️ About This Bot Detection Tool"):
    st.write("""
    ### How This Tool Works
    
    This system uses machine learning to analyze user activity patterns and identify potential bot accounts based on their behavior. The models have been trained on labeled data of known bot and human accounts.
    
    ### Key Features Used in Detection
    
    Bot accounts often exhibit different patterns than human users in several key areas:
    
    1. **Social Interactions**: Differences in follower-to-following ratios
    2. **Content Creation**: Frequency and type of content created (datasets, code, discussions)
    3. **Engagement Patterns**: Time spent on content, voting behavior, and interaction patterns
    
    ### Why Detecting Bots Matters
    
    Identifying bot accounts helps maintain platform integrity by:
    
    - Preventing manipulation of ratings and content visibility
    - Ensuring fair competition among users
    - Maintaining accurate analytics and metrics
    - Protecting users from spam and automated harassment
    
    ### Interpreting Results
    
    The prediction is based on comparing the input metrics to patterns learned from known bot and human accounts. A higher probability indicates stronger confidence in the prediction. Feature importance values show which metrics most influenced the prediction.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9E9E9E;'>Bot Detection API v1.0.0 | Powered by ML & FastAPI</p>", unsafe_allow_html=True)