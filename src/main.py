import json
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from torchvision import models, transforms
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_MODEL_PATH = "/Users/artemkuzmin/VSCProjects/python/user_sat/face_landmarker.task"
EMOTION_MODEL_PATH = "/Users/artemkuzmin/VSCProjects/python/user_sat/models/emotion_model.pth"
CLASSES_PATH = "/Users/artemkuzmin/VSCProjects/python/user_sat/models/classes.json"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}

EMOTION_VALENCE = {
    "happy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "sad": -1.0,
    "angry": -1.0,
    "fear": -0.7,
    "disgust": -0.8,
}


def compute_frame_analysis(probs):
    positive = sum(probs.get(emotion, 0.0) for emotion in POSITIVE_EMOTIONS)
    negative = sum(probs.get(emotion, 0.0) for emotion in NEGATIVE_EMOTIONS)
    valence = sum(
        EMOTION_VALENCE.get(emotion, 0.0) * (confidence / 100.0)
        for emotion, confidence in probs.items()
    )
    frame_satisfaction = max(0.0, min(100.0, 50.0 + 50.0 * valence))
    return frame_satisfaction, positive, negative

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

TESSELATION = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS


def draw_mesh(frame, landmarks):
    h, w, _ = frame.shape
    for connection in TESSELATION:
        p1 = landmarks[connection.start]
        p2 = landmarks[connection.end]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (80, 200, 255),
            1,
        )
    for connection in CONTOURS:
        p1 = landmarks[connection.start]
        p2 = landmarks[connection.end]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (0, 255, 0),
            2,
        )


def crop_face(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x1 = max(min(xs) - 40, 0)
    y1 = max(min(ys) - 40, 0)
    x2 = min(max(xs) + 40, w)
    y2 = min(max(ys) + 40, h)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face


def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    image = transform(Image.fromarray(face)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]

    probs = {
        classes[i]: probabilities[i].item() * 100.0
        for i in range(len(classes))
    }
    top_emotion = max(probs, key=probs.get)
    return top_emotion, probs[top_emotion], probs


def run():
    import sys
    from PyQt6.QtWidgets import QApplication
    from interface import DexPilotUI

    print("Device:", device)
    print("Classes:", classes)

    app = QApplication(sys.argv)
    window = DexPilotUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
