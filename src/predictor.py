from typing import List
from src.db_manager import DbManager
from datetime import datetime
from joblib import load
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Predictor:
    def __init__(self, model):
        """
        Initializes the Predictor class with a trained model.

        Args:
            model: The trained machine learning model.
        """
        self.model = model
        DATABASE_URL = os.getenv("DATABASE_URL")
        # Initialize the database manager
        self.manager = DbManager(database_url=DATABASE_URL)

    def predict_with_confidence(self, p_data_list) -> List[dict]:
        """
        Performs predictions on the input data list and adds the results to the database.

        Args:
            p_data_list (List[PredictionData]): List of input data for prediction.

        Returns:
            List[dict]: List of dictionaries containing predicted classes and confidence scores.
        """
        predictions_batch = []
        for p_data in p_data_list:
            input_data = [[p_data.SepalLengthCm, p_data.SepalWidthCm, p_data.PetalLengthCm, p_data.PetalWidthCm]]
            # Perform prediction
            probabilities = self.model.predict_proba(input_data)
            confidence_scores = max(probabilities[0])
            predicted_class = self.model.predict(input_data)[0]
            predictions_batch.append({"predicted_class": predicted_class, "confidence_scores": confidence_scores})
            # Get the current date and time
            date = datetime.today()
            # Add the prediction to the database
            self.manager.add_new_prediction(p_data.SepalLengthCm, p_data.SepalWidthCm, p_data.PetalLengthCm, p_data.PetalWidthCm, predicted_class, confidence_scores, date)
            
        return predictions_batch
# Usage Example
if __name__ == "__main__":
    # Load the trained model
    model_path = "./src/models/knn_model.joblib"
    model = load(model_path)
    # Create a Predictor object with the loaded model
    predictor = Predictor(model)
    # Sample input data for prediction
    sample_data = [
        {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
        {"SepalLengthCm": 7.0, "SepalWidthCm": 3.2, "PetalLengthCm": 4.7, "PetalWidthCm": 1.4},
        {"SepalLengthCm": 6.5, "SepalWidthCm": 3.2, "PetalLengthCm": 5.1, "PetalWidthCm": 2.0}
    ]
    # Perform prediction with confidence and print the results
    predictions = predictor.predict_with_confidence(sample_data)
    print(predictions)
