import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def check_sitting_posture(landmarks):
    left_shoulder = np.array(
        [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
    )
    right_shoulder = np.array(
        [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
    )
    left_hip = np.array(
        [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
    )
    right_hip = np.array(
        [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
    )

    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    hip_distance = np.linalg.norm(left_hip - right_hip)
    if shoulder_distance < 0.1 and hip_distance < 0.1:
        return "Correct Sitting Posture"
    else:
        return "Incorrect Sitting Posture"


def check_standing_posture(landmarks):
    left_shoulder = np.array(
        [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
    )
    right_shoulder = np.array(
        [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
    )
    left_hip = np.array(
        [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
    )
    right_hip = np.array(
        [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
    )
    left_knee = np.array(
        [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
    )
    right_knee = np.array(
        [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
    )

    shoulder_alignment = abs(left_shoulder[0] - right_shoulder[0])
    hip_alignment = abs(left_hip[0] - right_hip[0])
    knee_alignment = abs(left_knee[0] - right_knee[0])

    if shoulder_alignment < 0.1 and hip_alignment < 0.1 and knee_alignment < 0.1:
        return "Correct Standing Posture"
    else:
        return "Incorrect Standing Posture"


def posture_correction(engine):
    st.title("Posture Detection")
    st.write(
        "This module detects whether you are sitting or standing in the correct posture."
    )

    if "run" not in st.session_state:
        st.session_state["run"] = False

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Posture Detection"):
            st.session_state["run"] = True
    with col3:
        if st.button("Stop Posture Detection"):
            st.session_state["run"] = False

    stframe = st.empty()
    analysis_frame = st.empty()

    cap = cv2.VideoCapture(0)

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

                if result.pose_landmarks:
                    if (
                        result.pose_landmarks.landmark[
                            mp_pose.PoseLandmark.NOSE.value
                        ].y
                        > 0.5
                    ):  # Assuming front view if nose is below center
                        posture = check_sitting_posture(landmarks)
                    else:  # Side view logic
                        posture = check_standing_posture(landmarks)

                    engine.say(posture)
                    engine.runAndWait()

                    analysis_frame.markdown(
                        f"<div class='analysis-box'><h3 class='big-font'>Posture Analysis</h3><p>{posture}</p></div>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.write("Error in posture detection:", e)

            mp_drawing.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            stframe.image(image, channels="BGR")

    cap.release()
