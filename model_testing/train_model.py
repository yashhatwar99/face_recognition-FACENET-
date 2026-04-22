import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Set the experiment
mlflow.set_experiment("Face_Recognition_Model")

# 2. Load the data
X = np.load("embeddings.npy")
y = np.load("labels.npy")

# 3. Start the MLflow run
with mlflow.start_run(run_name="Dataset2_training"):
    
    # Log the parameters
    mlflow.log_param("dataset", "dataset2")
    mlflow.log_param("kernel", "linear")
    mlflow.log_param("probability", True)
    
    # Train the model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    # Calculate accuracy on the dataset
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # 4. LOG THE METRIC (This fixes the blank UI!)
    mlflow.log_metric("accuracy", acc)

    # 5. Log the trained model and register it as the latest version
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="face_recognition_model",
        registered_model_name="Production_Face_Classifier"
    )
    
    print(f"Model trained with an accuracy of {acc:.4f} and logged to MLflow Registry!")