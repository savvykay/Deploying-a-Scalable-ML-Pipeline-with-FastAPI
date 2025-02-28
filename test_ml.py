import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

# Sample test data for validation
sample_data = pd.DataFrame({
    "age": [25, 45],
    "workclass": ["Private", "Self-emp-not-inc"],
    "fnlgt": [226802, 89814],
    "education": ["11th", "HS-grad"],
    "education-num": [7, 9],
    "marital-status": ["Never-married", "Married-civ-spouse"],
    "occupation": ["Machine-op-inspct", "Farming-fishing"],
    "relationship": ["Own-child", "Husband"],
    "race": ["Black", "White"],
    "sex": ["Male", "Male"],
    "capital-gain": [0, 0],
    "capital-loss": [0, 0],
    "hours-per-week": [40, 50],
    "native-country": ["United-States", "United-States"],
    "salary": [">50K", "<=50K"]
})

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process data to get train-ready inputs
X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)

# Test 1: Ensure train_model returns a trained model
def test_train_model():
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), "train_model should return a LogisticRegression model"

# Test 2: confirm returns a NumPy array of predictions
def test_inference():
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Inference should return a NumPy array"
    assert preds.shape[0] == X.shape[0], "Number of predictions should match input data"

# Test 3: Ensure compute_model_metrics returns expected precision, recall, and F1-score
def test_compute_model_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(f1, float), "F1-score should be a float"
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1-score should be between 0 and 1"

# Test 4: Ensure process_data returns correctly formatted NumPy arrays
def test_process_data():
    assert isinstance(X, np.ndarray), "X should be a NumPy array"
    assert isinstance(y, np.ndarray), "y should be a NumPy array"
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"

if __name__ == "__main__":
    pytest.main()