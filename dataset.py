import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'asl_dataset'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path) or dir_ == ".ipynb_checkpoints":
        continue

    print(f"\n Collecting data for class: {dir_}")

    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)

        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f" Skipping non-image file: {img_file}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f" Failed to read image {img_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                if (max(x_) - min(x_) < 0.05) or (max(y_) - min(y_) < 0.05):
                    print(f" Hand too small in image {img_path}, skipping.")
                    continue

                for landmark in hand_landmarks.landmark:
                    norm_x = (landmark.x - min(x_)) / (max(x_) - min(x_))
                    norm_y = (landmark.y - min(y_)) / (max(y_) - min(y_))
                    data_aux.extend([norm_x, norm_y])

                data.append(data_aux)
                labels.append(int(dir_)) 

        else:
            print(f" No hands detected in image {img_path}, skipping.")

print("\n Data Collection Complete!")
print(f" Total Samples Collected: {len(data)}")
print(f" Unique Classes: {set(labels)}")

try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(" Data successfully saved to 'data.pickle'!")
except Exception as e:
    print(f" Error saving data.pickle: {e}")
