import cv2
import mediapipe as mp
import time
import numpy as np
import streamlit as st

# Initialize MediaPipe Hand Solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Feedback messages
EXERCISES = {
    "FIST": "Make a tight fist.",
    "STRETCH": "Stretch your fingers wide.",
    "FLEX": "Bend only your fingers (not the palm).",
}


# Function to calculate Euclidean distance
def distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Function to analyze and validate hand exercises
def analyze_hand_landmarks(landmarks, exercise_type):
    if exercise_type == "FIST":
        # Fist exercise: All fingertips close to their respective bases
        distances = [
            distance(landmarks[4], landmarks[0]),  # Thumb
            distance(landmarks[8], landmarks[0]),  # Index
            distance(landmarks[12], landmarks[0]),  # Middle
            distance(landmarks[16], landmarks[0]),  # Ring
            distance(landmarks[20], landmarks[0]),  # Pinky
        ]
        return all(d < 0.1 for d in distances)
    elif exercise_type == "STRETCH":
        # Stretch exercise: Fingertips far from the palm center
        distances = [
            distance(landmarks[8], landmarks[0]),
            distance(landmarks[12], landmarks[0]),
            distance(landmarks[16], landmarks[0]),
            distance(landmarks[20], landmarks[0]),
        ]
        return all(d > 0.2 for d in distances)
    elif exercise_type == "FLEX":
        # Flex exercise: Finger tips bending while keeping base stable
        return landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y
    else:
        return False


# Streamlit Frontend
def main():
    st.title("Hand Exercise Correction for Arthritis")
    st.markdown("Real-time analysis of hand exercises using MediaPipe and OpenCV.")

    # Streamlit UI Elements
    exercise_type = st.sidebar.selectbox(
        "Choose Exercise",
        ["FIST", "STRETCH", "FLEX"],
    )
    st.sidebar.write(EXERCISES[exercise_type])

    run = st.button("Start Exercise")
    stop = st.button("Stop Exercise")

    # Video Capture and Timer
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        # Initialize Timer
        start_time = 0
        correct_time = 0
        is_correct = False

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    st.write("Ignoring empty camera frame.")
                    continue

                # Process the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Analyze landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                        )

                        # Check exercise
                        correct = analyze_hand_landmarks(
                            hand_landmarks.landmark, exercise_type
                        )

                        if correct:
                            if not is_correct:
                                start_time = time.time()
                                is_correct = True
                            correct_time = time.time() - start_time
                            feedback = f"Correct! Hold time: {int(correct_time)}s"
                            color = (0, 255, 0)
                        else:
                            is_correct = False
                            feedback = "Incorrect. Adjust your hand position."
                            color = (0, 0, 255)
                            correct_time = 0

                        # Display Feedback
                        cv2.putText(
                            image,
                            feedback,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                # Display the image in Streamlit
                stframe.image(image, channels="BGR", use_column_width=True)

                if stop:
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
