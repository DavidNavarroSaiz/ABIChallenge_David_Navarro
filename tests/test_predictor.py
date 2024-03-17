import pytest
from src.predictor import Predictor
from src.model_training import KNNModel_Trainning
from joblib import load

from collections import namedtuple

# Define a namedtuple to represent the input data
PredictionData = namedtuple('PredictionData', ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

def test_predictor():
    knn_model = KNNModel_Trainning(data_file='./src/Iris.csv', model_directory='./models')
    
    # Train and save the model
    model = knn_model.train_model()
    # Create a Predictor object with the loaded model
    predictor = Predictor(model)
    # Sample input data for prediction
    sample_data = [
        PredictionData(SepalLengthCm=5.1, SepalWidthCm=3.5, PetalLengthCm=1.4, PetalWidthCm=0.2),
        PredictionData(SepalLengthCm=7.0, SepalWidthCm=3.2, PetalLengthCm=4.7, PetalWidthCm=1.4),
        PredictionData(SepalLengthCm=6.5, SepalWidthCm=3.2, PetalLengthCm=5.1, PetalWidthCm=2.0)
    ]
    # Perform prediction with confidence
    predictions = predictor.predict_with_confidence_no_store(sample_data)
    # Define expected predictions and confidence scores
    expected_predictions = [
        {"predicted_class": "Iris-setosa", "confidence_scores": 1},
        {"predicted_class": "Iris-versicolor", "confidence_scores": 1},
        {"predicted_class": "Iris-virginica", "confidence_scores": 1}
    ]
    
    # Assert the predictions
    assert predictions == expected_predictions
