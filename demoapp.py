import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Helper function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Function to validate exercises
def validate_exercise(landmarks, img_shape, exercise_type):
    """
    Validate different face exercises based on MediaPipe landmarks.
    """
    feedback = "Perform the exercise correctly."
    status = "incorrect"

    # Get image width and height for scaling
    img_width, img_height = img_shape[1], img_shape[0]

    # Extract relevant landmarks
    left_mouth_corner = (landmarks[61].x * img_width, landmarks[61].y * img_height)
    right_mouth_corner = (landmarks[291].x * img_width, landmarks[291].y * img_height)
    upper_lip = (landmarks[13].x * img_width, landmarks[13].y * img_height)
    lower_lip = (landmarks[14].x * img_width, landmarks[14].y * img_height)
    chin = (landmarks[152].x * img_width, landmarks[152].y * img_height)
    eyebrow_left_upper = (landmarks[70].x * img_width, landmarks[70].y * img_height)
    eyebrow_left_lower = (landmarks[107].x * img_width, landmarks[107].y * img_height)

    # Exercise-specific validation
    if exercise_type == "Jawline":
        jaw_distance = calculate_distance(upper_lip, lower_lip)
        if jaw_distance > 0.05 * img_height:
            status = "correct"
            feedback = "Great! Keep your jaw open."

    elif exercise_type == "Eyebrow Lift":
        eyebrow_distance = calculate_distance(eyebrow_left_upper, eyebrow_left_lower)
        if eyebrow_distance > 0.02 * img_height:
            status = "correct"
            feedback = "Eyebrows lifted! Hold the position."

    elif exercise_type == "Cheek Lift":
        mouth_distance = calculate_distance(left_mouth_corner, right_mouth_corner)
        if mouth_distance > 0.2 * img_width:
            status = "correct"
            feedback = "Good smile! Cheeks lifted."

    elif exercise_type == "Mouth Stretch":
        horizontal_mouth = calculate_distance(left_mouth_corner, right_mouth_corner)
        vertical_mouth = calculate_distance(upper_lip, lower_lip)
        if horizontal_mouth > 0.15 * img_width and vertical_mouth > 0.1 * img_height:
            status = "correct"
            feedback = "Mouth stretched wide. Hold it!"

    return status, feedback


# Streamlit App
def main():
    st.title("Facial Exercise Correction")
    st.write("Perform different facial exercises, and get real-time feedback.")

    # Select exercise
    exercise_type = st.sidebar.selectbox(
        "Choose Exercise", ["Jawline", "Eyebrow Lift", "Cheek Lift", "Mouth Stretch"]
    )

    stframe = st.empty()
    feedback_text = st.empty()
    timer_text = st.empty()

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        correct_start_time = None
        correct_duration = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not found.")
                break

            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    status, feedback = validate_exercise(
                        landmarks.landmark, frame.shape, exercise_type
                    )

                    if status == "correct":
                        if correct_start_time is None:
                            correct_start_time = time.time()
                        correct_duration = int(time.time() - correct_start_time)
                    else:
                        correct_start_time = None
                        correct_duration = 0

                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION
                    )

            feedback_text.write(f"**Feedback:** {feedback}")
            timer_text.write(f"**Time Held:** {correct_duration} seconds")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)

        cap.release()


if __name__ == "__main__":
    main()
