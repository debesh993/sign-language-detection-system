import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load trained model
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Labels A-Z
labels_dict = {i: chr(65 + i) for i in range(26)}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Streamlit UI
st.title("Sign Language Detection")
video_placeholder = st.empty()
word_placeholder = st.empty()
sentence_placeholder = st.empty()

# Session state
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

if st.button("Refresh"):
    st.session_state.current_word = ""
    st.session_state.sentence = ""

# Timing variables
FIRST_LETTER_DELAY = 3.0
LETTER_INTERVAL = 2.0
WORD_END_INTERVAL = 2.0

predicted_character = ""
first_letter_added = False
hand_first_detected_time = None
last_letter_time = 0
last_hand_time = time.time()

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Failed to access camera")
    st.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from camera")
        break

    hand_present = False
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_present = True
        if hand_first_detected_time is None:
            hand_first_detected_time = time.time()

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            if not x_ or not y_ or (max(x_) - min(x_) < 0.01) or (max(y_) - min(y_) < 0.01):
                continue

            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min(x_)) / (max(x_) - min(x_))
                norm_y = (lm.y - min(y_)) / (max(y_) - min(y_))
                data_aux.extend([norm_x, norm_y])

            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            # --- Basic Prediction logic ---
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_class = int(prediction[0])
                predicted_character = labels_dict.get(predicted_class, '?')

                # Draw bounding box and character
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception as e:
                predicted_character = '?'

    else:
        hand_first_detected_time = None

    current_time = time.time()

    # --- Letter timing ---
    if hand_present:
        if not first_letter_added:
            if hand_first_detected_time and (current_time - hand_first_detected_time >= FIRST_LETTER_DELAY):
                st.session_state.current_word += predicted_character
                first_letter_added = True
                last_letter_time = current_time
        else:
            if current_time - last_letter_time >= LETTER_INTERVAL:
                st.session_state.current_word += predicted_character
                last_letter_time = current_time
        last_hand_time = current_time
    else:
        if st.session_state.current_word and (current_time - last_hand_time >= WORD_END_INTERVAL):
            st.session_state.sentence += st.session_state.current_word + " "
            st.session_state.current_word = ""
            first_letter_added = False
            hand_first_detected_time = None

    # Show frame and text
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB")
    word_placeholder.text(f"Word: {st.session_state.current_word}")
    sentence_placeholder.text(f"Sentence: {st.session_state.sentence}")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
