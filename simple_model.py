# simple_model.py
from sklearn.dummy import DummyClassifier
import pickle

# Create a dummy classifier (just as a placeholder model)
model = DummyClassifier(strategy="most_frequent")
X = [[0], [1], [2], [3]]  # Some dummy data
y = [0, 1, 0, 1]           # Dummy labels
model.fit(X, y)

# Save the model to a .pkl file
with open("autism_model.pkl", "wb") as f:
    pickle.dump(model, f)
