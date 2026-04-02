from sklearn.svm import SVC
import numpy as np
import pickle

X = np.load("embeddings.npy")
y = np.load("labels.npy")

model = SVC(kernel='linear', probability=True)
model.fit(X, y)

pickle.dump(model, open("face_model.pkl", "wb"))

