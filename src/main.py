import time
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from torchvision import models, transforms
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_MODEL_PATH = "../face_landmarker.task"
EMOTION_MODEL_PATH = "../models/emotion_model.pth"
CLASSES_PATH = "../models/classes.json"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)
print("Classes:", classes)

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        EMOTION_MODEL_PATH,
        map_location=device
    )
)


model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)

TESSELATION = (
    vision.FaceLandmarksConnections
    .FACE_LANDMARKS_TESSELATION
)
CONTOURS = (
    vision.FaceLandmarksConnections
    .FACE_LANDMARKS_CONTOURS
)

def draw_mesh(frame, landmarks):
    h, w, _ = frame.shape
    for connection in TESSELATION:
        p1 = landmarks[connection.start]
        p2 = landmarks[connection.end]
        cv2.line(
            frame,
            (
                int(p1.x*w),
                int(p1.y*h)
            ),
            (
                int(p2.x*w),
                int(p2.y*h)
            ),
            (80,200,255),
            1
        )
    for connection in CONTOURS:
        p1 = landmarks[connection.start]
        p2 = landmarks[connection.end]
        cv2.line(
            frame,
            (
                int(p1.x*w),
                int(p1.y*h)
            ),
            (
                int(p2.x*w),
                int(p2.y*h)
            ),
            (0,255,0),
            2
        )

def crop_face(frame, landmarks):
    h, w, _ = frame.shape
    xs = [
        int(p.x*w)
        for p in landmarks
    ]
    ys = [
        int(p.y*h)
        for p in landmarks
    ]
    x1 = max(min(xs)-40,0)
    y1 = max(min(ys)-40,0)

    x2 = min(max(xs)+40,w)
    y2 = min(max(ys)+40,h)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face

def predict_emotion(face):
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        face
    )

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    )

    image = image.to(device)
    with torch.no_grad():
        output = model(
            image
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            1
        )

    emotion = classes[
        index.item()
    ]

    percent = (
        confidence.item()*100
    )
    return emotion, percent

cap = cv2.VideoCapture(0)

print("Running. Press Q to exit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(
        frame,
        1
    )
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    mp_image = mp.Image(
        image_format=
        mp.ImageFormat.SRGB,
        data=rgb
    )
    timestamp = int(
        time.time()*1000
    )
    result = face_landmarker.detect_for_video(
        mp_image,
        timestamp
    )
    if result.face_landmarks:
        for landmarks in result.face_landmarks:
            draw_mesh(
                frame,
                landmarks
            )
            face = crop_face(
                frame,
                landmarks
            )
            if face is not None:
                emotion, confidence = predict_emotion(
                    face
                )
                text = (
                    f"{emotion}: "
                    f"{confidence:.1f}%"
                )
                cv2.putText(
                    frame,
                    text,
                    (
                        20,
                        50
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0,255,0),
                    3
                )

    cv2.imshow(
        "Emotion Recognition",
        frame
    )
    if cv2.waitKey(1)&0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
