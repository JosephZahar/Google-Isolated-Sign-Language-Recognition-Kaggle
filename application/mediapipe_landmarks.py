import cv2
import numpy as np
import json
import pandas as pd
import mediapipe as mp
import tflite_runtime.interpreter as tflite

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image.flags.writeable = False # img no longer writeable
    pred = model.process(image) # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color reconversion
    return image, pred

class CFG:
    data_dir = "/kaggle/input/asl-signs/"
    rows_per_frame = 543

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

sign_map = load_json_file('/Users/macbookpro/Desktop/Kaggle/asl_signs/data/sign_to_prediction_index_map.json')
train_data = pd.read_csv('/Users/macbookpro/Desktop/Kaggle/asl_signs/data/train.csv')

s2p_map = {k.lower(): v for k, v in load_json_file('/Users/macbookpro/Desktop/Kaggle/asl_signs/data/sign_to_prediction_index_map.json').items()}
p2s_map = {v: k for k, v in load_json_file('/Users/macbookpro/Desktop/Kaggle/asl_signs/data/sign_to_prediction_index_map.json').items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)


def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))


# Extract Landmarks from video
def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks.landmark else np.zeros(468, 3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks.landmark else np.zeros(33, 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks.landmark else np.zeros(21, 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks.landmark else np.zeros(21, 3)
    return np.concatenate([face, lh, pose, rh])


def real_time_asl():
    interpreter = tflite.Interpreter("/Users/macbookpro/Desktop/Kaggle/asl_signs/results/asl_model/model.tflite")
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")

    sequence_data = []
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)

            landmarks = extract_coordinates(results)
            sequence_data.append(landmarks)
            if len(sequence_data) == 30:
                prediction = prediction_fn(inputs=load_relevant_data_subset(sequence_data))
                sign = np.argmax(prediction["outputs"])
                print(prediction)
                cv2.putText(image, f"Prediction:    {decoder(sign)}", (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

real_time_asl()