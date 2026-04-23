# 1. MUST IMPORT TORCH FIRST
import torch
torch.set_num_threads(1) # Force PyTorch to use 1 safe thread

# 2. MUST IMPORT OPENCV SECOND
import cv2
cv2.setNumThreads(0) # Force OpenCV to stop fighting PyTorch for threads

# 3. Import everything else
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from facenet_pytorch import InceptionResnetV1, MTCNN
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

# 1. Define the Prometheus Gauge
DRIFT_SCORE_GAUGE = Gauge(
    "model_data_drift_score", 
    "Cosine distance drift score between training and production embeddings"
)

app = FastAPI(title="Face Recognition API")
Instrumentator().instrument(app).expose(app)

# Global list to store live embeddings for drift calculation
live_embeddings_cache = []

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

# 2. Background function to calculate drift using Numpy (Cosine Distance)
def calculate_drift():
    global live_embeddings_cache
    
    # Wait for enough data and ensure we have reference data to compare against
    if len(live_embeddings_cache) < 10 or REF_CENTROID is None: 
        return
    
    try:
        # Convert live cache to numpy array
        curr_arr = np.array(live_embeddings_cache)
        
        # Calculate the centroid (mean vector) of the live production faces
        curr_centroid = np.mean(curr_arr, axis=0)
        
        # Calculate Cosine Similarity: dot(A, B) / (norm(A) * norm(B))
        cos_sim = np.dot(REF_CENTROID, curr_centroid) / (np.linalg.norm(REF_CENTROID) * np.linalg.norm(curr_centroid))
        
        # Convert similarity to distance (0 means identical distributions, higher means drift)
        drift_score = 1.0 - cos_sim
        
        # Update Prometheus
        DRIFT_SCORE_GAUGE.set(float(drift_score))
        print(f"Prometheus Gauge Updated (Cosine Distance): {drift_score:.4f}")

        # Optional: Prevent the cache from eating all your server RAM over time
        if len(live_embeddings_cache) > 1000:
            # Keep only the 500 most recent faces
            live_embeddings_cache = live_embeddings_cache[-500:]

    except Exception as e:
        print(f"Error calculating drift: {e}")

@app.post("/predict")
async def predict_face(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
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
                # 3. Add to cache for drift detection
                live_embeddings_cache.append(embedding.flatten())

            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)
            name = model.predict(embedding)[0] if max_prob >= 0.7 else "Unknown"

            results.append({"name": name, "confidence": float(max_prob)})

    # Trigger drift update in the background
    background_tasks.add_task(calculate_drift)
    
    return {"results": results}