# File: main.py
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Face Recognition API")

# Add these lines to instrument your app and expose the /metrics endpoint
Instrumentator().instrument(app).expose(app)

# Load models once when the server starts
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

try:
    model = pickle.load(open("face_model.pkl", "rb"))
except FileNotFoundError:
    model = None

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    if model is None:
        return {"error": "SVM model not trained yet."}

    # Read the uploaded image into OpenCV format
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(rgb)
    results = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Boundary checks
            h, w, _ = rgb.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess for FaceNet
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255
            face_tensor = face_tensor.unsqueeze(0)

            # Extract Embedding
            with torch.no_grad():
                embedding = facenet(face_tensor).numpy()

            # Predict Name
            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)

            name = "Unknown"
            if max_prob >= 0.7:
                name = model.predict(embedding)[0]

            results.append({
                "name": name,
                "confidence": float(max_prob),
                "bounding_box": [x1, y1, x2, y2]
            })

    return {"faces_detected": len(results), "results": results}
