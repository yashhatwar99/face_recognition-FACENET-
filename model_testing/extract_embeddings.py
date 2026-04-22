from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
import os
from PIL import Image

mtcnn = MTCNN(image_size=160)
model = InceptionResnetV1(pretrained='vggface2').eval()

dataset_path = "dataset" # Update this to your dataset path

embeddings = []
labels = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        img = Image.open(img_path).convert('RGB')
        
        face = mtcnn(img)
        if face is None:
            continue
        
        face = face.unsqueeze(0)
        embedding = model(face).detach().numpy()[0]

        embeddings.append(embedding)
        labels.append(person)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)