import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def physiotherapy_exercises(exercise, engine):
    st.title("🧑‍⚕️ Physiotherapy Exercise Tracker")

    if "run" not in st.session_state:
        st.session_state["run"] = False

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"Start {exercise}"):
            st.session_state["run"] = True
    with col3:
        if st.button(f"Stop {exercise}"):
            st.session_state["run"] = False

    stframe = st.empty()
    analysis_frame = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.write("Error: Unable to access the camera.")
        return

    counter = 0
    stage = None

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            if not st.session_state["run"]:
                break
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = result.pose_landmarks.landmark

                if exercise == "Shoulder Raise":
                    elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                    hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                    angle = calculate_angle(elbow, shoulder, hip)
                elif exercise == "Leg Raise":
                    hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    ]
                    ankle = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                    ]
                    angle = calculate_angle(hip, knee, ankle)
                elif exercise == "Arm Curl":
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                    elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                    wrist = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                    ]
                    angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1

                    engine.say(f"Rep number {counter}")
                    engine.runAndWait()

                analysis_frame.markdown(
                    f"""
                    <div class='analysis-box'>
                    <h3 class='big-font'>{exercise} Analysis</h3>
                    <p><strong>Angle:</strong> {angle:.2f}</p>
                    <p><strong>Stage:</strong> {stage}</p>
                    <p><strong>Reps:</strong> {counter}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.write("Error in exercise analysis:", e)

            mp_drawing.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            stframe.image(image, channels="BGR")

    cap.release()
