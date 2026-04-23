# 1. MUST IMPORT TORCH FIRST
import torch
torch.set_num_threads(1) # Force PyTorch to use 1 safe thread

# 2. MUST IMPORT OPENCV SECOND
import cv2
cv2.setNumThreads(0) # Force OpenCV to stop fighting PyTorch for threads

# 3. Import everything else
import numpy as np
import pickle
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from facenet_pytorch import InceptionResnetV1, MTCNN
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram

# --- 1. PROMETHEUS METRICS DEFINITIONS ---
DRIFT_SCORE_GAUGE = Gauge(
    "model_data_drift_score", 
    "Cosine distance drift score between training and production embeddings"
)
API_REQUESTS_TOTAL = Counter(
    "num_api_requests", 
    "Total number of prediction API requests received"
)
MODEL_ACCURACY_GAUGE = Gauge(
    "model_accuracy", 
    "Rolling model prediction accuracy based on provided ground truth"
)
ACCURACY_EVALUATED_TOTAL = Gauge(
    "model_accuracy_evaluated_total",
    "Number of predictions evaluated for model_accuracy (ground truth provided)"
)
ACCURACY_CORRECT_TOTAL = Gauge(
    "model_accuracy_correct_total",
    "Number of correct predictions counted in model_accuracy"
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Time spent in /predict endpoint",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2.5, 5, 10)
)
UNKNOWN_PREDICTIONS_TOTAL = Counter(
    "unknown_predictions_total",
    "Total predictions classified as Unknown (primary face)"
)
AVG_PRIMARY_CONFIDENCE = Gauge(
    "avg_primary_confidence",
    "Rolling average confidence for the primary detected face"
)
LIVE_EMBEDDINGS_CACHE_SIZE = Gauge(
    "live_embeddings_cache_size",
    "Number of embeddings currently cached for drift calculation"
)

app = FastAPI(title="Face Recognition API")
Instrumentator().instrument(app).expose(app)

# --- 2. GLOBAL VARIABLES ---
live_embeddings_cache = []
correct_predictions = 0
total_evaluated = 0
primary_conf_sum = 0.0
primary_conf_count = 0

try:
    REFERENCE_DATA = np.load("embeddings.npy") # Your 80-celeb baseline
    # Pre-calculate the reference centroid (mean vector) to save CPU later
    REF_CENTROID = np.mean(REFERENCE_DATA, axis=0) if len(REFERENCE_DATA) > 0 else None
except Exception:
    REFERENCE_DATA = np.array([])
    REF_CENTROID = None

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
    
    # Wait for enough data and ensure we have reference data to compare against
    if len(live_embeddings_cache) < 10 or REF_CENTROID is None: 
        return
    
    try:
        curr_arr = np.array(live_embeddings_cache)
        curr_centroid = np.mean(curr_arr, axis=0)
        
        # Calculate Cosine Similarity: dot(A, B) / (norm(A) * norm(B))
        cos_sim = np.dot(REF_CENTROID, curr_centroid) / (np.linalg.norm(REF_CENTROID) * np.linalg.norm(curr_centroid))
        
        # Convert similarity to distance (0 means identical distributions, higher means drift)
        drift_score = 1.0 - cos_sim
        
        DRIFT_SCORE_GAUGE.set(float(drift_score))
        LIVE_EMBEDDINGS_CACHE_SIZE.set(len(live_embeddings_cache))
        print(f"Prometheus Gauge Updated (Cosine Distance): {drift_score:.4f}")

        if len(live_embeddings_cache) > 1000:
            live_embeddings_cache = live_embeddings_cache[-500:]
            LIVE_EMBEDDINGS_CACHE_SIZE.set(len(live_embeddings_cache))

    except Exception as e:
        print(f"Error calculating drift: {e}")

# --- 4. PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_face(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    actual_name: str = Form(None) # Optional ground truth to track model accuracy
):
    global correct_predictions, total_evaluated, primary_conf_sum, primary_conf_count
    
    # Increment total API requests
    API_REQUESTS_TOTAL.inc()
    start_t = time.perf_counter()

    if model is None:
        return {"error": "SVM model not trained yet."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb)
    results = []
    primary_prediction = "Unknown"
    primary_confidence = None

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
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
                LIVE_EMBEDDINGS_CACHE_SIZE.set(len(live_embeddings_cache))

            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)
            
            # Using your original 0.15 threshold
            name = model.predict(embedding)[0] if max_prob >= 0.15 else "Unknown"

            if not results: 
                primary_prediction = name
                primary_confidence = float(max_prob)

            results.append({"name": name, "confidence": float(max_prob), "bounding_box": [x1, y1, x2, y2]})

    # --- 5. TRACK MODEL ACCURACY ---
    if actual_name is not None and results:
        total_evaluated += 1
        if primary_prediction.lower() == actual_name.lower():
            correct_predictions += 1
        
        current_accuracy = correct_predictions / total_evaluated
        MODEL_ACCURACY_GAUGE.set(current_accuracy)
        ACCURACY_EVALUATED_TOTAL.set(total_evaluated)
        ACCURACY_CORRECT_TOTAL.set(correct_predictions)

    background_tasks.add_task(calculate_drift)
    PREDICTION_LATENCY_SECONDS.observe(time.perf_counter() - start_t)

    if primary_prediction == "Unknown":
        UNKNOWN_PREDICTIONS_TOTAL.inc()

    if primary_confidence is not None:
        primary_conf_sum += primary_confidence
        primary_conf_count += 1
        AVG_PRIMARY_CONFIDENCE.set(primary_conf_sum / primary_conf_count)
    
    # Returning faces_detected alongside results so the frontend won't crash
    return {"faces_detected": len(results), "results": results}