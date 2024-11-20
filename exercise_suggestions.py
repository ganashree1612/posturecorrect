import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from streamlit_option_menu import option_menu
import pyttsx3

def personalized_exercise_plan():
    st.title("📋 Personalized Exercise Plan")

    # Gather user information
    age = st.number_input("Enter your age", min_value=1, max_value=120)
    fitness_level = st.selectbox(
        "Select your current fitness level", ["Beginner", "Intermediate", "Advanced"]
    )
    goal = st.selectbox(
        "Select your goal",
        [
            "Increase Strength",
            "Improve Flexibility",
            "Rehabilitation",
            "Weight Loss",
            "Cardio Endurance",
            "Muscle Toning",
        ],
    )

    # Display recommendations based on user input
    if st.button("Generate Plan"):
        st.markdown(
            "<h2 class='section-title'>Recommended Exercises Based on Your Inputs:</h2>",
            unsafe_allow_html=True,
        )

        # Example recommendations based on input
        if goal == "Increase Strength":
            if fitness_level == "Beginner":
                st.markdown("- **Bodyweight Squats**: 3 sets of 10 reps")
                st.markdown("- **Push-ups**: 3 sets of 8 reps")
                st.markdown("- **Dumbbell Lunges**: 3 sets of 10 reps each leg")
                st.markdown("- **Dumbbell Rows**: 3 sets of 12 reps")
