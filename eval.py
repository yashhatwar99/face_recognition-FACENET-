from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

X = np.load("embeddings.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model2 = SVC(kernel='linear', probability=True)
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)