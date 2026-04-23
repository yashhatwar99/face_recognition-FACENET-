import pandas as pd
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric

# 1. Load your Reference Data (80 Celebs)
# Replace with your 80-celeb embeddings file
reference_embeddings = np.load("embeddings_80_celebs.npy") 
ref_df = pd.DataFrame(reference_embeddings)

# 2. Load your Current Data (100 Celebs)
# This includes the 20 new people
current_embeddings = np.load("embeddings_100_celebs.npy") 
curr_df = pd.DataFrame(current_embeddings)

# 3. Setup the Evidently Report for Embeddings
# This specifically looks at high-dimensional vector drift
drift_report = Report(metrics=[
    EmbeddingsDriftMetric('face_embeddings')
])

# Define the columns (FaceNet produces 128 dimensions)
column_mapping = {
    "embeddings": {"face_embeddings": list(range(128))}
}

drift_report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

# 4. Log the Score to your existing MLflow Experiment
mlflow.set_experiment("Face_Recognition_Model")

with mlflow.start_run(run_name="Drift_Analysis_80_vs_100"):
    # Extract the numerical drift score
    result = drift_report.as_dict()
    drift_score = result['metrics'][0]['result']['drift_score']
    
    # Log the metric for your MLflow graph
    mlflow.log_metric("data_drift_score", drift_score)
    
    # Save the full HTML report so you can show the visualization
    drift_report.save_html("drift_report.html")
    mlflow.log_artifact("drift_report.html")

    print(f"Drift Analysis Complete. Score: {drift_score:.4f}")