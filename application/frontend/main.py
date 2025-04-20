import requests
import streamlit as st
import pandas as pd
import numpy as np
import json

# Configure page
st.set_page_config(
    page_title="Bot or Not? Detector",
    page_icon="🤖",
    layout="wide"
)

# Define header with styling
st.title('🤖 Bot or Not?')
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-text {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="sub-header">Predict whether a user is likely a bot based on their usage metrics</p>', 
            unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"

# Sidebar for model selection
st.sidebar.title("Model Selection")

# Get available models from backend
try:
    models_response = requests.get(f'{API_URL}/models/')
    if models_response.status_code == 200:
        available_models = models_response.json().get('available_models', [])
        model_choice = st.sidebar.selectbox(
            'Select prediction model:',
            available_models,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    else:
        st.sidebar.error("Unable to fetch model list")
        model_choice = "random_forest"  # Default if unable to fetch
except requests.exceptions.ConnectionError:
    st.sidebar.error('❌ Cannot connect to backend server. Please ensure it is running.')
    model_choice = "random_forest"  # Default if server is not running

# Model information
st.sidebar.markdown("### Model Information")
if model_choice == "random_forest":
    st.sidebar.info("""
    **Random Forest** is an ensemble model that combines multiple decision trees to produce more accurate and stable predictions.
    
    It works well with high-dimensional data and can handle both numerical and categorical features.
    """)
elif model_choice == "bagging":
    st.sidebar.info("""
    **Bagging (Bootstrap Aggregating)** is an ensemble technique that reduces variance and helps prevent overfitting by taking samples from the training data with replacement.
    
    It's effective for reducing model variance and improving stability.
    """)

# Health check for backend
try:
    health_response = requests.get(f'{API_URL}/health')
    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend server is connected and healthy")
    else:
        st.sidebar.warning("⚠️ Backend server may have issues")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Backend server is not reachable")

# Create two columns for the form
col1, col2 = st.columns(2)

# Create the input form
with st.form(key='user_input_form'):
    # Left column - Personal Information
    with col1:
        st.subheader("User Information")
        name = st.text_input('Name', 'John Doe')
        email_id = st.text_input('Email address', 'email@example.com')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        is_glogin = st.checkbox('Uses Google login', value=True)
        
    # Right column - Activity Metrics
    with col2:
        st.subheader("User Activity Metrics")
        follower_count = st.number_input('Follower count', min_value=0, value=10)
        following_count = st.number_input('Following count', min_value=0, value=20)
        dataset_count = st.number_input('Dataset count', min_value=0, value=2)
        code_count = st.number_input('Code notebooks count', min_value=0, value=5)
        discussion_count = st.number_input('Discussion count', min_value=0, value=15)
        
    # Additional metrics that might need more space
    st.subheader("Engagement Metrics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_nb_read_time_min = st.number_input('Avg. notebook read time (min)', 
                                              min_value=0.0, value=10.0, step=0.5)
    with col4:
        total_votes_gave_nb = st.number_input('Total votes on notebooks', min_value=0, value=8)
    with col5:
        total_votes_gave_ds = st.number_input('Total votes on datasets', min_value=0, value=3)
        total_votes_gave_dc = st.number_input('Total votes on discussions', min_value=0, value=2)

    # Submit button styled with primary color
    submit_button = st.form_submit_button(label='Predict', use_container_width=True)

# Handle form submission
if submit_button:
    with st.spinner('Processing prediction...'):
        # Convert gender to one-hot encoding
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0

        # Prepare data for API request
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
            "model_choice": model_choice
        }

        try:
            # Send data to backend for prediction
            response = requests.post(f'{API_URL}/predict/', json=user_input)

            # Display prediction results
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction')
                model_used = result.get('model_used', model_choice)

                # Show results in an expander
                with st.expander("📊 Prediction Results", expanded=True):
                    st.markdown(f"**Model used**: {model_used.replace('_', ' ').title()}")
                    
                    # Show prediction with appropriate styling
                    if prediction == 1:
                        st.markdown("""
                        <div style="background-color: #FFEBEE; padding: 20px; border-radius: 5px; border-left: 5px solid #F44336;">
                            <h3 style="color: #D32F2F; margin: 0;">⚠️ Bot Detected</h3>
                            <p>This user's activity pattern resembles that of a bot.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 5px; border-left: 5px solid #4CAF50;">
                            <h3 style="color: #2E7D32; margin: 0;">✅ Human User</h3>
                            <p>This user's activity pattern appears to be that of a legitimate human user.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display probability if available
                    if 'probability' in result:
                        prob = result['probability']
                        st.markdown(f"**Bot Probability**: {prob:.2%}")
                        
                        # Visualization of probability
                        st.progress(prob)
                        
                    # Display feature importance if available
                    if 'feature_importance' in result:
                        st.subheader("Feature Importance")
                        feature_imp = result['feature_importance']
                        
                        # Convert feature importance to DataFrame for display
                        feat_df = pd.DataFrame({
                            'Feature': list(feature_imp.keys()),
                            'Importance': list(feature_imp.values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Display as bar chart
                        st.bar_chart(feat_df.set_index('Feature'))
                        
                        # And as a table
                        st.dataframe(
                            feat_df.style.background_gradient(cmap='Blues', subset=['Importance'])
                        )
                    
                    # Show input data summary
                    with st.expander("Input Data Summary"):
                        input_df = pd.DataFrame([user_input])
                        # Remove technical fields
                        display_df = input_df.drop(['model_choice', 'gender_female', 'gender_male'], axis=1)
                        st.dataframe(display_df)
                
            else:
                # Handle API error
                error_message = "Unknown error"
                try:
                    error_detail = response.json()
                    error_message = error_detail.get('detail', error_message)
                except:
                    pass
                st.error(f'Server error: {error_message}')
        
        except requests.exceptions.ConnectionError:
            st.error('❌ Cannot connect to backend server. Please ensure it is running.')

# Add some explanation about the system at the bottom
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    ### How it works
    
    This tool uses machine learning models trained on user activity data to identify potential bot accounts.
    
    The models analyze various metrics about user behavior, such as:
    
    - Social interactions (followers, following)
    - Content creation (datasets, code notebooks, discussions)
    - Engagement patterns (voting behavior, read time)
    
    ### Why detect bots?
    
    Detecting bots is important for maintaining the integrity of platforms and ensuring fair user experiences.
    Bots can manipulate ratings, spread misinformation, or artificially inflate metrics.
    
    ### Available Models
    
    - **Random Forest**: Good for complex patterns and handles diverse features well
    - **Bagging**: Focuses on reducing variance and preventing overfitting
    """)