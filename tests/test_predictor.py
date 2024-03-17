import pytest
from src.predictor import Predictor
from src.model_training import KNNModelTraining
from joblib import load

from collections import namedtuple

from collections import namedtuple
from src.models import KNNModelTraining, Predictor

# Define a namedtuple to represent the input data
PredictionData = namedtuple('PredictionData', ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm'])

def test_predictor():
    knn_model = KNNModelTraining(data_file='./src/Iris.csv', model_directory='./models')
    
    # Train and save the model
    model = knn_model.train_model()
    # Create a Predictor object with the loaded model
    predictor = Predictor(model)
    # Sample input data for prediction
    sample_data = [
        PredictionData(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2),
        PredictionData(sepal_length_cm=7.0, sepal_width_cm=3.2, petal_length_cm=4.7, petal_width_cm=1.4),
        PredictionData(sepal_length_cm=6.5, sepal_width_cm=3.2, petal_length_cm=5.1, petal_width_cm=2.0)
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
