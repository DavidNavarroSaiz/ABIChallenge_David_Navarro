from sqlalchemy import create_engine, inspect
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
from joblib import load
from src.db import Base
from src.db_manager import DbManager
import uvicorn
import os
from src.model_training import KNNModel_Trainning
from src.predictor import Predictor

# FastAPI app instance
app = FastAPI()

# Set up database connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

inspector = inspect(engine)
if not inspector.has_table("Abi_InferenceTable"):
    print("creating table")
    Base.metadata.create_all(bind=engine)

# Load initial model on application startup
def load_model():
    """
    Load the latest trained model.
    If no model is found, train a new one.
    """
    model_files = [f for f in os.listdir('./src/models') if f.startswith('knn_model') and f.endswith('.joblib')]
    if not model_files:
        print("No trained model found. Training a new one...")
        train_model()
        model_files = [f for f in os.listdir('./src/models') if f.startswith('knn_model') and f.endswith('.joblib')]
    latest_model = max(model_files)
    name_last_model = os.path.join('./src/models', latest_model)
    print("predicting with model: ", name_last_model)
    return load(name_last_model)

def train_model():
    """
    Train the KNN model over the Iris.csv Dataset.
    """
    try:
        knn_model = KNNModel_Trainning('./src/Iris.csv', './src/models')
        message = knn_model.train_and_save_model()
        print(f"Model trained and saved successfully: {message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

predict_model = load_model()

# Initialize database manager
db_manager = DbManager(database_url=DATABASE_URL)

# Pydantic models
class TrainData(BaseModel):
    data_file: str
    model_dir: str

class PredictionData(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

# Endpoints

@app.post("/train/")
async def train_model(data_file: str = './src/Iris.csv', model_directory: str = './src/models'):
    """
    Endpoint to train the KNN model over the Iris.csv Dataset.

    Receives training data file path and model Folder path, trains the model, and saves it.

    Returns:
        dict: Success message.
    """
    try:
        knn_model = KNNModel_Trainning(data_file, model_directory)
        message = knn_model.train_and_save_model()
        # Load the latest trained model after training
        global predict_model
        predict_model = load_model()
        return {"message": f"Model trained and saved successfully, {message}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/predict/")
async def predict(p_data_list: List[PredictionData] = Body(...)):
    """
    Endpoint to make batch predictions with the latest trained model.

    Receives a list of PredictionData instances, makes predictions for each, and returns the predictions.

    Args:
        p_data_list (List[PredictionData]): List of input data for batch prediction. the list can only have one element

    Returns:
        dict: Batch predictions.
    Example input Data
        [
            {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2
            },
            {
                "SepalLengthCm": 7.0,
                "SepalWidthCm": 3.2,
                "PetalLengthCm": 4.7,
                "PetalWidthCm": 1.4
            },
            {
                "SepalLengthCm": 6.5,
                "SepalWidthCm": 3.2,
                "PetalLengthCm": 5.1,
                "PetalWidthCm": 2.0
            }
        ]
    """
    # try:
    predictor = Predictor(predict_model)
    predictions_result = predictor.predict_with_confidence(p_data_list)

    return {"predictions": predictions_result}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")

@app.get("/predictions/")
async def read_all_predictions():
    """
    Endpoint to read all prediction records from the database.

    Returns:
        list: A list of dictionaries representing all prediction records.
    """
    try:
        predictions = db_manager.read_all_predictions()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while reading predictions from database: {str(e)}")

@app.get("/prediction/{prediction_id}")
async def read_prediction(prediction_id: int):
    """
    Endpoint to read a prediction record from the database by its ID.

    Args:
        prediction_id (int): The ID of the prediction record to read.

    Returns:
        dict: A dictionary representing the prediction record.
    """
    try:
        prediction = db_manager.read_prediction_by_id(prediction_id)
        if prediction:
            return prediction
        else:
            raise HTTPException(status_code=404, detail="Prediction not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while reading prediction from database: {str(e)}")

@app.delete("/prediction/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """
    Endpoint to delete a prediction record from the database by its ID.

    Args:
        prediction_id (int): The ID of the prediction record to delete.

    Returns:
        dict: Success message.
    """
    try:
        result = db_manager.delete_prediction(prediction_id)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while deleting prediction from database: {str(e)}")

@app.delete("/predictions/")
async def delete_all_predictions():
    """
    Endpoint to delete all prediction records from the database.

    Returns:
        dict: Success message.
    """
    try:
        db_manager.delete_all_predictions()
        return {"message": "All predictions deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while deleting predictions from database: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
