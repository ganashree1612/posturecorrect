import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Define exercises in a dropdown
exercises = {
    "Eye Blinking": [33, 133],  # Landmarks for eyes
    "Smile": [61, 291],  # Landmarks for mouth corners
    "Eyebrow Raising": [70, 276],  # Landmarks for eyebrows
}
selected_exercise = st.sidebar.selectbox("Choose an Exercise", list(exercises.keys()))


def calculate_accuracy(face_landmarks, exercise):
    # Placeholder function for calculating accuracy
    # You should define how to calculate the accuracy based on exercise
    return np.random.rand()  # Returns a random float as a dummy accuracy


def annotate_landmarks(image, landmarks, exercise_landmarks):
    for landmark in exercise_landmarks:
        x = int(landmarks.landmark[landmark].x * image.shape[1])
        y = int(landmarks.landmark[landmark].y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image


# Setting up the main app
st.title("Face Exercise App with MediaPipe")
st.write("Selected Exercise: ", selected_exercise)

# Setup webcam capture
cap = cv2.VideoCapture(0)
frameST = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(frame)

    # Draw face landmarks and calculate exercise accuracy
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            frame = annotate_landmarks(
                frame, face_landmarks, exercises[selected_exercise]
            )
            accuracy = calculate_accuracy(face_landmarks, selected_exercise)
            frame = cv2.putText(
                frame,
                f"Accuracy: {accuracy:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Display the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frameST.image(frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
