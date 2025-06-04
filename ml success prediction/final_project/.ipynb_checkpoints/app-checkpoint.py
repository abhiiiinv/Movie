import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_filename = "trained_movie_model.sav"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Load dataset to get feature names
data = pd.read_csv("movie_encoded.csv")

# Remove unwanted columns and fix indexing issue
feature_columns = [col for col in data.columns if col not in ["Unnamed: 0", "success", "profit", "revenue"]]

# Extract categorical options
original_language_options = sorted(data["original_language"].unique().tolist())

# Mapping binary fields
binary_mapping = {0: "No", 1: "Yes"}

# Streamlit UI
st.title("Movie Success Prediction App")
st.write("Enter movie details to predict if it will be successful.")

# User inputs with appropriate widgets
adult = st.radio("Is this an adult movie?", options=[0, 1], format_func=lambda x: binary_mapping[x])
video = st.radio("Does this movie have a video?", options=[0, 1], format_func=lambda x: binary_mapping[x])

budget = st.slider("Budget", min_value=0, max_value=500000000, step=1000000)
popularity = st.slider("Popularity", min_value=0.0, max_value=600.0, step=0.1)
runtime = st.slider("Runtime (minutes)", min_value=0, max_value=300, step=1)
vote_average = st.slider("Vote Average", min_value=0.0, max_value=10.0, step=0.1)
vote_count = st.slider("Vote Count", min_value=0, max_value=10000, step=1)

original_language = st.selectbox("Original Language", original_language_options)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[adult, budget, original_language, popularity, runtime, video, vote_average, vote_count]],
        columns=feature_columns)

    prediction = model.predict(input_data)
    st.write(f"*Predicted Success:* {'Yes' if prediction[0] ==1 else 'No'}")