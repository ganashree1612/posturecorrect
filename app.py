import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from streamlit_option_menu import option_menu
import threading
from streamlit_webrtc import webrtc_streamer
import demoapp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
st.set_page_config(page_title="PhysioTherapy Assistant", page_icon="🧑‍⚕️", layout="wide")

# Add custom CSS for improved UI
st.markdown(
    """
    <style>
    .big-font {
        font-size:22px !important;
        font-weight: bold;
        color: #2e4053;
    }
    .analysis-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border: 2px solid #c3cfd9;
        margin-top: 20px;
        color:black;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #1f618d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def streamlit_menu():
    with st.sidebar:
        selected = option_menu(
            menu_title="Physiotherapy💪🧑‍⚕️",
            options=[
                "Select Exercise",
                "Physio Exercises",
                "Posture Correction",
                "hand exercises",
            ],
            icons=["activity", "person-rolodex", "list-task"],
            menu_icon="cast",
            default_index=0,
        )
    return selected


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


def physiotherapy_exercises():
    st.title("🧑‍⚕️ Physiotherapy Exercise Tracker")

    exercise = st.selectbox(
        "Choose an exercise", ["Shoulder Raise", "Leg Raise", "Arm Curl"]
    )

    # Start/Stop Toggle
    if "run" not in st.session_state:
        st.session_state["run"] = False

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"{'Stop' if st.session_state['run'] else 'Start'} {exercise}"):
            st.session_state["run"] = not st.session_state["run"]

    stframe = st.empty()  # Placeholder for video feed
    analysis_frame = st.empty()  # Placeholder for exercise analysis output

    # Open Webcam
    cap = cv2.VideoCapture(0)
    right_counter, left_counter = 0, 0
    right_stage, left_stage = None, None

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            if not st.session_state["run"]:
                st.write("Exercise tracking stopped.")
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            # Convert BGR to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = result.pose_landmarks.landmark

                # Shoulder Raise
                if exercise == "Shoulder Raise":
                    # Right arm
                    right_shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                    right_elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                    right_hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                    right_angle = calculate_angle(
                        right_elbow, right_shoulder, right_hip
                    )

                    # Left arm
                    left_shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    left_elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                    ]
                    left_hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    ]
                    left_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

                    # Feedback
                    if right_angle < 90:
                        cv2.putText(
                            image,
                            "Raise Right Hand!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Right Hand OK",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                    if left_angle < 90:
                        cv2.putText(
                            image,
                            "Raise Left Hand!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Left Hand OK",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                # Arm Curl
                elif exercise == "Arm Curl":
                    # Right arm
                    right_wrist = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                    ]
                    right_elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                    right_shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                    right_angle = calculate_angle(
                        right_wrist, right_elbow, right_shoulder
                    )

                    # Left arm
                    left_wrist = [
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                    ]
                    left_elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                    ]
                    left_shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)

                    # Feedback
                    if right_angle < 45:
                        cv2.putText(
                            image,
                            "Right Curl OK",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Curl Right Arm!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                    if left_angle < 45:
                        cv2.putText(
                            image,
                            "Left Curl OK",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Curl Left Arm!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                # Leg Raise
                elif exercise == "Leg Raise":
                    # Right leg
                    right_hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                    right_knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    ]
                    right_ankle = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                    ]
                    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

                    # Left leg
                    left_hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    ]
                    left_knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    ]
                    left_ankle = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                    ]
                    left_angle = calculate_angle(left_hip, left_knee, left_ankle)

                    # Feedback
                    if right_angle < 90:
                        cv2.putText(
                            image,
                            "Raise Right Leg!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Right Leg OK",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                    if left_angle < 90:
                        cv2.putText(
                            image,
                            "Raise Left Leg!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Left Leg OK",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                # Display real-time video feed
                stframe.image(image, channels="BGR")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    cap.release()

    # cv2.destroyAllWindows()


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


def posture_correction():
    st.title("Posture Correction")
    st.write("This module detects your posture and provides corrective feedback.")
    st.write("*Ensure you are visible in front of the camera.*")

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
    analysis_frame = st.empty()  # Placeholder for posture feedback

    cap = cv2.VideoCapture(0)
    print(cv2.VideoCapture(0).getBackendName())  # Get the backend used (DirectShow, etc.)

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

                # Shoulder alignment
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
                shoulder_difference = abs(left_shoulder[1] - right_shoulder[1])

                # Neck alignment (using nose as a reference point)
                nose = np.array(
                    [
                        landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                    ]
                )
                shoulder_midpoint = (left_shoulder + right_shoulder) / 2
                neck_alignment = abs(
                    nose[0] - shoulder_midpoint[0]
                )  # Horizontal alignment check

                # Spine alignment
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
                spine_alignment = abs(
                    shoulder_midpoint[0] - ((left_hip[0] + right_hip[0]) / 2)
                )  # Vertical alignment check

                # Feedback logic
                feedback = []
                if shoulder_difference > 0.05:
                    feedback.append("⚠ Your shoulders are not level.")
                if neck_alignment > 0.05:
                    feedback.append("⚠ Your neck is not aligned with your shoulders.")
                if spine_alignment > 0.05:
                    feedback.append("⚠ Your spine appears bent. Try to straighten it.")

                if not feedback:
                    analysis_frame.success("✅ Good posture! Keep it up.")
                else:
                    analysis_frame.warning(" ".join(feedback))

            except Exception as e:
                analysis_frame.warning(
                    "Pose landmarks not detected. Please adjust your position."
                )

            # Draw pose landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            stframe.image(image, channels="BGR", use_container_width=True)

    cap.release()


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

    # Calculate distances
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    hip_distance = np.linalg.norm(left_hip - right_hip)

    # Thresholds for standing posture
    if shoulder_distance < 0.2 and hip_distance < 0.2:
        return "Correct Standing Posture"
    else:
        return "Incorrect Standing Posture"


# Start the posture analysis application
def start_posture_analysis():
    st.title("Posture Analysis: Real-Time Feedback")

    exercise = st.selectbox("Select Posture Type", ["Sitting", "Standing"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the camera.")
        return

    feedback_placeholder = st.empty()  # Placeholder for feedback
    stframe = st.empty()  # Streamlit placeholder for video frames

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Ensure your webcam is working.")
                break

            # Convert the frame to RGB for Mediapipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw landmarks and connections
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            posture_status = "No landmarks detected."
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Check posture based on the selected exercise
                if exercise == "Sitting":
                    posture_status = posture_correction()
                elif exercise == "Standing":
                    posture_status = check_standing_posture(landmarks)

                # Draw pose landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Overlay posture status on the video
                cv2.putText(
                    image,
                    posture_status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if "Correct" in posture_status else (0, 0, 255),
                    2,
                )

            # Display the processed video frame in Streamlit
            stframe.image(image, channels="BGR", use_container_width=True)

            # Update the feedback at the bottom
            feedback_placeholder.markdown(
                f"### Feedback: {posture_status}", unsafe_allow_html=True
            )

            # Optional: Limit frame rate to prevent high CPU usage
            cv2.waitKey(1)

    cap.release()


def main():
    selected = streamlit_menu()

    if selected == "Physio Exercises":
        physiotherapy_exercises()
    elif selected == "Posture Correction":
        start_posture_analysis()
    elif selected == "hand exercises":
        demoapp.main()


if __name__ == "__main__":
    main()
