# 1. MUST IMPORT TORCH FIRST
import torch
torch.set_num_threads(1) # Force PyTorch to use 1 safe thread

# 2. MUST IMPORT OPENCV SECOND
import cv2
cv2.setNumThreads(0) # Force OpenCV to stop fighting PyTorch for threads

# 3. Import everything else
from fastapi import FastAPI, UploadFile, File
import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from facenet_pytorch import InceptionResnetV1, MTCNN
from prometheus_fastapi_instrumentator import Instrumentator
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from prometheus_client import Gauge

# 1. Define the Prometheus Gauge
DRIFT_SCORE_GAUGE = Gauge(
    "model_data_drift_score", 
    "Statistical drift score between training and production embeddings"
)

app = FastAPI(title="Face Recognition API")
Instrumentator().instrument(app).expose(app)

# Global list to store live embeddings for drift calculation
live_embeddings_cache = []
REFERENCE_DATA = np.load("embeddings.npy") # Your 80-celeb baseline

# Load models
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
try:
    # Use Exception here just in case the fallback file causes a different error
    model = pickle.load(open("face_model.pkl", "rb"))
except Exception:
    model = None

# 2. Background function to calculate drift
def calculate_drift():
    global live_embeddings_cache
    if len(live_embeddings_cache) < 10: # Wait for enough data
        return
    
    ref_df = pd.DataFrame(REFERENCE_DATA)
    curr_df = pd.DataFrame(live_embeddings_cache)
    
    drift_report = Report(metrics=[EmbeddingsDriftMetric('face_embeddings')])
    # Map the 128 dimensions of the FaceNet embedding
    column_mapping = {"embeddings": {"face_embeddings": list(range(128))}}
    
    drift_report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
    
    # Update Prometheus
    score = drift_report.as_dict()['metrics'][0]['result']['drift_score']
    DRIFT_SCORE_GAUGE.set(score)
    print(f"Prometheus Gauge Updated: {score}")

@app.post("/predict")
async def predict_face(file: UploadFile = File(...),background_tasks: BackgroundTasks = BackgroundTasks()):
# ... (KEEP THE REST OF YOUR PREDICT FUNCTION EXACTLY THE SAME) ...
    if model is None:
        return {"error": "SVM model not trained yet."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb)
    results = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # ... (your existing boundary check code) ...
            face = rgb[y1:y2, x1:x2]
            if face.size == 0: continue

            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                embedding = facenet(face_tensor).numpy()
                # 3. Add to cache for drift detection
                live_embeddings_cache.append(embedding.flatten())

            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)
            name = model.predict(embedding)[0] if max_prob >= 0.7 else "Unknown"

            results.append({"name": name, "confidence": float(max_prob)})

    # Trigger drift update in the background
    background_tasks.add_task(calculate_drift)
    
    return {"results": results}