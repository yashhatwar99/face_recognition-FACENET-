import mlflow
import mlflow.sklearn
import numpy as np
import pickle
import time  # For tracking latency
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_experiment("Face_Recognition_Model")

X = np.load("embeddings.npy")
y = np.load("labels.npy")

with mlflow.start_run(run_name="lets see data drit just for aamir khan"):
    # 1. Log Parameters
    mlflow.log_param("dataset_size", len(y))
    mlflow.log_param("num_classes", len(np.unique(y)))

    # 2. Track Training Time
    start_train = time.time()
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    train_duration = time.time() - start_train
    mlflow.log_metric("training_duration_sec", train_duration)

    # 3. Calculate and Log Core Metrics
    start_pred = time.time()
    y_pred = model.predict(X)
    inference_duration = (time.time() - start_pred) / len(X) # Avg time per face
    
    acc = accuracy_score(y, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("avg_inference_latency", inference_duration)

    # 4. Generate a detailed Classification Report
    report = classification_report(y, y_pred, output_dict=True)
    # Log global averages
    mlflow.log_metric("macro_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("weighted_recall", report['weighted avg']['recall'])

    # 5. Local Save and MLflow Registration
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="face_recognition_model",
        registered_model_name="Production_Face_Classifier"
    )
    
    print(f"Logged Accuracy: {acc:.4f} | Training Time: {train_duration:.2f}s")