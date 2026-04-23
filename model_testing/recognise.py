import cv2
import torch
import pickle
import numpy as np
import os
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC

# Initialize models
mtcnn = MTCNN(keep_all=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def load_data():
    if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
        X = np.load("embeddings.npy").tolist()
        y = np.load("labels.npy").tolist()
        return X, y
    return [], []

X, y = load_data()
model = pickle.load(open("face_model.pkl", "rb")) if os.path.exists("face_model.pkl") else None

cap = cv2.VideoCapture(0)

# Capture State Variables
target_name = ""
photos_to_take = 0
photos_taken = 0
is_capturing = False

print("Controls: 'k' to setup capture, 'c' to capture photo, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)
    current_embedding = None

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            face_crop = rgb[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255
                face_tensor = face_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    current_embedding = facenet(face_tensor).numpy()[0]

                name = "Unknown"
                if model is not None and not is_capturing:
                    probs = model.predict_proba([current_embedding])
                    if np.max(probs) > 0.2:
                        name = model.predict([current_embedding])[0]

                # Visual Feedback
                color = (0, 255, 0) if not is_capturing else (0, 165, 255) # Green vs Orange
                label = f"CAPTURING: {target_name} ({photos_taken}/{photos_to_take})" if is_capturing else name
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-time Dataset Builder", frame)
    key = cv2.waitKey(1) & 0xFF

    # 1. Setup Capture Session
    if key == ord('k'):
        target_name = input("Enter name: ").strip()
        try:
            photos_to_take = int(input("How many photos to take? "))
            photos_taken = 0
            is_capturing = True
            print(f"Ready to capture {target_name}. Press 'c' {photos_to_take} times.")
        except ValueError:
            print("Invalid number.")

    # 2. Capture Single Photo in Session
    elif key == ord('c') and is_capturing and current_embedding is not None:
        # Update Data Arrays
        X.append(current_embedding)
        y.append(target_name)
        
        # Update Physical Dataset Folder
        person_dir = os.path.join("dataset", target_name)
        os.makedirs(person_dir, exist_ok=True)
        img_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
        
        photos_taken += 1
        print(f"Captured {photos_taken}/{photos_to_take}")

        # Auto-finish and Train
        if photos_taken >= photos_to_take:
            print("Training model with new data...")
            model = SVC(kernel='linear', probability=True)
            model.fit(X, y)
            
            # Save everything
            np.save("embeddings.npy", np.array(X))
            np.save("labels.npy", np.array(y))
            with open("face_model.pkl", "wb") as f:
                pickle.dump(model, f)
            
            is_capturing = False
            print(f"Done! {target_name} is now recognized.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()