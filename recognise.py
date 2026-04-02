import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

model = pickle.load(open("face_model.pkl", "rb"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = rgb.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (160,160))
            
            face = torch.tensor(face).permute(2,0,1).float()/255
            face = face.unsqueeze(0)

            # embedding = facenet(face).detach().numpy()
            with torch.no_grad():
                embedding = facenet(face).numpy()

            # name = model.predict(embedding)[0]
            probs = model.predict_proba(embedding)
            max_prob = np.max(probs)

            if max_prob < 0.7:
                name = "Unknown"
            else:
                name = model.predict(embedding)[0]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()