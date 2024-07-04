import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
log_reg = joblib.load('log_reg_model.pkl')
log_reg_weighted = joblib.load('log_reg_weighted_model.pkl')
nb_model_weighted = joblib.load('nb_model_weighted.pkl')
deep_learning_model = load_model('deep_learning_model.h5')

# Load the TF-IDF vectorizer used during training
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define target labels and their names
target_labels = ['EG', 'LB', 'LY', 'MA', 'SD']
target_names = ['Egypt', 'Lebanon', 'Libya', 'Morocco', 'Sudan']
label_name_mapping = dict(zip(target_labels, target_names))

# Set up Streamlit app
st.set_page_config(layout="centered")

# Adding background image using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://t4.ftcdn.net/jpg/03/49/38/23/360_F_349382308_jqIgbOIzRi034AjFSYpinhD5fOlkX4Y6.jpg") no-repeat center center fixed;
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.8); /* Adjust the RGBA value to control transparency */
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set larger header
st.markdown("<h1 style='font-size: 55px; text-align: center;'>Arabic Dialects Prediction</h1>", unsafe_allow_html=True)

# Description text
st.markdown("<p style='font-size: 28px; text-align: center;'>Choose a model and enter text to predict its dialect.</p>", unsafe_allow_html=True)

# Dropdown for model selection with custom styling
st.markdown("<p style='font-size: 24px; text-align: center;'>Choose a Model</p>", unsafe_allow_html=True)
model_choice = st.selectbox(
    "",
    ("Logistic Regression", "Logistic Regression with Weighted Classes", "Naive Bayes with Weighted Classes", "Deep Learning")
)

# Input text from user with custom styling
st.markdown("<p style='font-size: 24px; text-align: center;'>Enter text to predict its dialect:</p>", unsafe_allow_html=True)
user_input = st.text_area("", "")

# Button to make predictions
if st.button("Predict"):
    if user_input:
        # Preprocess the input text
        X_input = tfidf_vectorizer.transform([user_input])

        if model_choice in ["Logistic Regression", "Logistic Regression with Weighted Classes", "Naive Bayes with Weighted Classes"]:
            # Predict using the chosen model
            if model_choice == "Logistic Regression":
                model = log_reg
            elif model_choice == "Logistic Regression with Weighted Classes":
                model = log_reg_weighted
            elif model_choice == "Naive Bayes with Weighted Classes":
                model = nb_model_weighted

            y_pred_proba = model.predict_proba(X_input)[0]

        elif model_choice == "Deep Learning":
            # Ensure the input shape matches the deep learning model's expected input shape
            X_input_dl = X_input.toarray()
            if X_input_dl.shape[1] != deep_learning_model.input_shape[1]:
                # Pad or truncate to the required shape
                required_shape = deep_learning_model.input_shape[1]
                if X_input_dl.shape[1] > required_shape:
                    X_input_dl = X_input_dl[:, :required_shape]
                else:
                    padding = required_shape - X_input_dl.shape[1]
                    X_input_dl = np.pad(X_input_dl, ((0, 0), (0, padding)), 'constant')

            y_pred_proba = deep_learning_model.predict(X_input_dl)[0]
        
        # Get the predicted dialect
        predicted_dialect = label_name_mapping[target_labels[np.argmax(y_pred_proba)]]
        
        # Display the result
        st.write(f"Predicted Dialect: {predicted_dialect}")
        
        # Plot the probabilities
        fig, ax = plt.subplots()
        sns.barplot(x=target_names, y=y_pred_proba, ax=ax)
        ax.set_xlabel("Dialect")
        ax.set_ylabel("Probability")
        ax.set_title("Dialect Prediction Probabilities")
        
        st.pyplot(fig)
    else:
        st.write("Please enter text to predict its dialect.")







