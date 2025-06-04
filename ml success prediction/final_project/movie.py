import streamlit as st
import pandas as pd
import pickle



st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0D0D0D !important; /* Slightly darker black */
    }
    * {
        color: firebrick !important; /* Firebrick red text */
    }
    </style>
    """,
    unsafe_allow_html=True
)


model_filename = "trained_movie_model2.sav"
with open(model_filename, "rb") as file:
    model = pickle.load(file)


data = pd.read_csv("movie_encoded.csv")

# unwanted columns from the data
feature_columns = [col for col in data.columns if col not in ["Unnamed: 0", "success", "profit", "revenue"]]

# Mapping binary fields
binary_mapping = {0: "No", 1: "Yes"}

# Streamlit UI

st.title("Movie Success Prediction App")

st.write("Enter movie details to predict if it will be successful.")

# User inputs with appropriate widgets - now with unique keys

with st.form(key="movie_form"):
    col1, col2 = st.columns(2)

    with col1:
        adult = st.radio(
            "Is this an adult movie?",
            options=[0, 1],
            format_func=lambda x: binary_mapping[x],
            key="adult_radio"  # Added unique key
        )



        budget = st.slider("Budget", min_value=0, max_value=500000000, step=1000000)
        popularity = st.slider("Popularity", min_value=0.0, max_value=600.0, step=0.1)
        runtime = st.slider("Runtime (minutes)", min_value=0, max_value=300, step=1)

    with col2:
        video = st.radio(
            "Does this movie have a video?",
            options=[1, 0],
            format_func=lambda x: binary_mapping[x],
            key="video_radio"  # Added unique key
        )

        # runtime = st.slider("Runtime (minutes)", min_value=0, max_value=300, step=1)
        vote_average = st.slider("Vote Average", min_value=0.0, max_value=10.0, step=0.1)
        vote_count = st.slider("Vote Count", min_value=0, max_value=10000, step=1)
    submit_button = st.form_submit_button(label="Predict")

# Predict button
if submit_button:
    # Set original_language to 0
    original_language = 1

    input_data = pd.DataFrame(
        [[adult, budget, popularity, original_language, runtime, video, vote_average, vote_count]],
        columns=feature_columns)

    prediction = model.predict(input_data)
    st.write(f"*Predicted Success:* {'Yes' if prediction[0] == 1 else 'No'}")



