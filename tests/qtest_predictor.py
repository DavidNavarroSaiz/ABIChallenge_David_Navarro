import pytest
from src.predictor import Predictor
from unittest.mock import Mock

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def mock_db_manager():
    return Mock()

@pytest.fixture
def predictor(mock_model, mock_db_manager):
    return Predictor(mock_model)

def test_predict_with_confidence(predictor, mock_model, mock_db_manager):
    # Mock the model's predict_proba and predict methods
    mock_model.predict_proba.return_value = [[0.2, 0.8]]  # Example probabilities
    mock_model.predict.return_value = ["Iris-versicolor"]  # Example predicted class

    # Sample input data for prediction
    sample_data = [
        {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
        {"SepalLengthCm": 7.0, "SepalWidthCm": 3.2, "PetalLengthCm": 4.7, "PetalWidthCm": 1.4},
        {"SepalLengthCm": 6.5, "SepalWidthCm": 3.2, "PetalLengthCm": 5.1, "PetalWidthCm": 2.0}
    ]

    # Perform prediction with confidence
    predictions = predictor.predict_with_confidence(sample_data)

    # Assert that DbManager method was called correctly
    assert mock_db_manager.add_new_prediction.call_count == len(sample_data)

    # Assert the returned predictions
    assert predictions == [{"predicted_class": "Iris-versicolor", "confidence_scores": 0.8}]
