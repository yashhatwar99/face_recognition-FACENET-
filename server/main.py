# 1. MUST IMPORT TORCH FIRST
import torch
torch.set_num_threads(1) # Force PyTorch to use 1 safe thread

# 2. MUST IMPORT OPENCV SECOND
import cv2
cv2.setNumThreads(0) # Force OpenCV to stop fighting PyTorch for threads

# 3. Import everything else
import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from facenet_pytorch import InceptionResnetV1, MTCNN
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric

# --- 1. PROMETHEUS METRICS DEFINITIONS ---
api_requests_total = Counter("api_requests_total", "Number of API calls")
prediction_requests_total = Counter("prediction_requests_total", "Number of predictions")
DRIFT_SCORE_GAUGE = Gauge("model_data_drift_score", "Statistical drift score between training and production embeddings")
model_accuracy = Gauge("model_accuracy", "Model performance monitoring")

app = FastAPI(title="Face Recognition API")

# Instrumentator automatically exposes the /metrics endpoint, 
# so we don't need make_asgi_app() anymore.
Instrumentator().instrument(app).expose(app)

# --- 2. GLOBAL VARIABLES FOR TRACKING ---
live_embeddings_cache = []
correct_predictions = 0
total_evaluated = 0

try:
    REFERENCE_DATA = np.load("embeddings.npy") # Your 80-celeb baseline
except FileNotFoundError:
    REFERENCE_DATA = np.array([])

# Load models
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
try:
    model = pickle.load(open("face_model.pkl", "rb"))
except Exception:
    model = None

# --- 3. BACKGROUND DRIFT CALCULATION ---
def calculate_drift():
    global live_embeddings_cache
    if len(live_embeddings_cache) < 10 or len(REFERENCE_DATA) == 0: 
        return # Wait for enough data
    
    ref_df = pd.DataFrame(REFERENCE_DATA)
    curr_df = pd.DataFrame(live_embeddings_cache)
    
    drift_report = Report(metrics=[EmbeddingsDriftMetric('face_embeddings')])
    column_mapping = {"embeddings": {"face_embeddings": list(range(128))}}
    
    drift_report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
    
    score = drift_report.as_dict()['metrics'][0]['result']['drift_score']
    DRIFT_SCORE_GAUGE.set(score)
    print(f"Prometheus Gauge Updated: {score}")

# --- 4. PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_face(
    background_tasks: BackgroundTasks, # FIXED: Injected background tasks
    file: UploadFile = File(...),
    actual_name: str = Form(None) # Optional ground truth for accuracy tracking
):
    global correct_predictions, total_evaluated
    
    # Increment custom Prometheus counters
    api_requests_total.inc()
    prediction_requests_total.inc()

    if model is None:
        return {"error": "SVM model not trained yet."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb)
    results = []
    primary_prediction = "Unknown"

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Boundary checks
            h, w, _ = rgb.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = rgb[y1:y2, x1:x2]
            if face.size == 0: continue

            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                embedding = facenet(face_tensor).numpy()
                live_embeddings_cache.append(embedding.flatten())

            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)
            name = model.predict(embedding)[0] if max_prob >= 0.7 else "Unknown"
            
            # Assuming the first detected face is the primary one for accuracy checks
            if not results: 
                primary_prediction = name

            results.append({"name": name, "confidence": float(max_prob), "bounding_box": [x1, y1, x2, y2]})

    # --- 5. TRACK MODEL ACCURACY ---
    if actual_name is not None and results:
        total_evaluated += 1
        if primary_prediction.lower() == actual_name.lower():
            correct_predictions += 1
        
        current_accuracy = correct_predictions / total_evaluated
        model_accuracy.set(current_accuracy)

    # Trigger drift update in the background
    background_tasks.add_task(calculate_drift)
    
    return {"faces_detected": len(results), "results": results}