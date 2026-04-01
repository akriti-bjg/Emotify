import cv2
import numpy as np
import torch
import timm
import time

MODEL_PATH = "swin_emotify_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]
CONF_THRESHOLD = 0.70    
STABLE_TIME = 2.5        

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Emotion model loaded on:", DEVICE)

prev_box = None
stable_emotion = None
emotion_start_time = None
recommendation_done = False

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    face = (face - mean) / std
    face = np.transpose(face, (2, 0, 1)) 
    face = np.expand_dims(face, axis=0)

    return face

def detect_emotion_from_frame(frame):
    """
    Input:
        frame (numpy array, BGR format from OpenCV)

    Returns:
        emotion (str or None)
        confidence (float 0-1)
        stable_triggered (bool)
        face_box (tuple or None)
    """

    global prev_box, stable_emotion, emotion_start_time, recommendation_done

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, 0.0, False, None

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]

    if prev_box is None:
        prev_box = (x, y, w, h)
    else:
        px, py, pw, ph = prev_box
        prev_box = (
            int(px * 0.7 + x * 0.3),
            int(py * 0.7 + y * 0.3),
            int(pw * 0.7 + w * 0.3),
            int(ph * 0.7 + h * 0.3)
        )

    (x, y, w, h) = prev_box
    face_region = frame[y:y+h, x:x+w]

    if face_region.size == 0:
        return None, 0.0, False, None

    processed_face = preprocess_face(face_region)
    input_tensor = torch.from_numpy(processed_face).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    emotion = CLASS_NAMES[pred.item()]
    conf = confidence.item()

    current_time = time.time()
    stable_triggered = False

    if conf >= CONF_THRESHOLD:
        if emotion != stable_emotion:
            stable_emotion = emotion
            emotion_start_time = current_time
            recommendation_done = False
        else:
            elapsed = current_time - emotion_start_time
            if elapsed >= STABLE_TIME and not recommendation_done:
                recommendation_done = True
                stable_triggered = True

    return emotion, conf, stable_triggered, (x, y, w, h)
