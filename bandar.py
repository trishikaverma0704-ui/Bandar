# Bandar
monkey changes face as per expression
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load monkey images using absolute paths
def load_monkey(name):
    img_path = os.path.join(script_dir, "monkeys", f"{name}.jpg")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    return cv2.resize(img, (400, 400))

monkey_neutral = load_monkey("neutral")
monkey_happy = load_monkey("happy")
monkey_thinking = load_monkey("thinking")
monkey_shocked = load_monkey("shocked")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    monkey = monkey_neutral

    # Face detection
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]

        top_lip = face.landmark[13].y
        bottom_lip = face.landmark[14].y

        if abs(top_lip - bottom_lip) > 0.03:
            monkey = monkey_shocked

        left_mouth = face.landmark[61].x
        right_mouth = face.landmark[291].x

        if abs(left_mouth - right_mouth) > 0.09:
            monkey = monkey_happy

    # Hand detected
    if hand_result.multi_hand_landmarks:
        monkey = monkey_thinking

    frame = cv2.resize(frame, (400, 400))
    combined = np.hstack((frame, monkey))

    cv2.imshow("Thinking Monkey Detector", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
