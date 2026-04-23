import mlflow
import mlflow.sklearn
import numpy as np
import pickle 
import time
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
    
    # Calculate accuracy on the dataset
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # 4. LOG THE METRIC
    mlflow.log_metric("accuracy", acc)

    # 5. Log the trained model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="face_recognition_model",
        registered_model_name="Production_Face_Classifier"
    )
    
    # --- NEW: SAVE THE MODEL LOCALLY FOR GITHUB ACTIONS ---
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("pickle save done")
    # ------------------------------------------------------
    
    print(f"Model trained with an accuracy of {acc:.4f} and logged to MLflow Registry!")
