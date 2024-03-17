from sqlalchemy import create_engine
from datetime import datetime
from src.models import AbiInferenceTable
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc
from dotenv import load_dotenv
import os

class DbManager:
    def __init__(self, database_url):
        """Initializes the DbManager.

        Args:
            database_url (str): The URL of the database.
        """
        self.engine = create_engine(database_url)
        self.Session_maker = sessionmaker(bind=self.engine)
        self.session = self.Session_maker() 

    def add_new_prediction(self, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, prediction, confidence, date_str):
        """Creates a new prediction record and adds it to the database.

        Args:
            sepal_length_cm (float): Sepal length in cm.
            sepal_width_cm (float): Sepal width in cm.
            petal_length_cm (float): Petal length in cm.
            petal_width_cm (float): Petal width in cm.
            prediction (str): The prediction result.
            confidence (float): The confidence level of the prediction.
            date_str (datetime.datetime): The date and time of the prediction.

        Returns:
            AbiInferenceTable: The newly created AbiInferenceTable object, or None if creation fails.
        """
        new_state = AbiInferenceTable(
            SepalLengthCm=sepal_length_cm,
            SepalWidthCm=sepal_width_cm,
            PetalLengthCm=petal_length_cm,
            PetalWidthCm=petal_width_cm,
            prediction=prediction,
            confidence=confidence,
            call_datetime=date_str,
        )
        self.session.add(new_state)
        self.session.commit()
        return new_state

    def delete_prediction(self, prediction_id):
        """Deletes a prediction record from the database.

        Args:
            prediction_id (int): The ID of the prediction record to delete.

        Raises:
            PredictionNotFoundError: If the prediction with the given ID is not found.
        """
        prediction = self.session.query(AbiInferenceTable).filter_by(id=prediction_id).first()
        if prediction:
            self.session.delete(prediction)
            self.session.commit()
            return ("Prediction deleted successfully")
        else:
            return  (f"Prediction with ID {prediction_id} not found")
        
    def delete_all_predictions(self):
        """Deletes all prediction records from the database."""
        self.session.query(AbiInferenceTable).delete()
        self.session.commit()

    def read_prediction_by_id(self, prediction_id):
        """Reads a prediction record from the database by its ID.

        Args:
            prediction_id (int): The ID of the prediction record to read.

        Returns:
            AbiInferenceTable: The AbiInferenceTable object with the specified ID, or None if not found.
        """
        return self.session.query(AbiInferenceTable).filter_by(id=prediction_id).first()

    def read_all_predictions(self):
        """Reads all prediction records from the database.

        Returns:
            list: A list of AbiInferenceTable objects representing all prediction records.
        """
        return self.session.query(AbiInferenceTable).all()


if __name__ == "__main__":
    from db import Base
    from sqlalchemy import create_engine, inspect

    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")

    manager = DbManager(database_url=DATABASE_URL)
    # date = datetime.today()
    # manager.add_new_prediction(0.1, 0.2,0.4,0.3, "petal",0.34,date )
    # manager.add_new_prediction(0.2, 0.3,0.4,0.5, "petalic 2",0.34,date )
