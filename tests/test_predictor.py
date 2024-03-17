import pytest
from src.predictor import Predictor
from joblib import load

from collections import namedtuple

# Define a namedtuple to represent the input data
PredictionData = namedtuple('PredictionData', ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

def test_predictor():
    model_path = "./src/models/knn_model_1.joblib"
    model = load(model_path)
    # Create a Predictor object with the loaded model
    predictor = Predictor(model)
    # Sample input data for prediction
    sample_data = [
        PredictionData(SepalLengthCm=5.1, SepalWidthCm=3.5, PetalLengthCm=1.4, PetalWidthCm=0.2),
        PredictionData(SepalLengthCm=7.0, SepalWidthCm=3.2, PetalLengthCm=4.7, PetalWidthCm=1.4),
        PredictionData(SepalLengthCm=6.5, SepalWidthCm=3.2, PetalLengthCm=5.1, PetalWidthCm=2.0)
    ]
    # Perform prediction with confidence
    predictions = predictor.predict_with_confidence(sample_data)
    # Define expected predictions and confidence scores
    expected_predictions = [
        {"predicted_class": "Iris-setosa", "confidence_scores": 1},
        {"predicted_class": "Iris-versicolor", "confidence_scores": 1},
        {"predicted_class": "Iris-virginica", "confidence_scores": 1}
    ]
    
    # Assert the predictions
    assert predictions == expected_predictions
