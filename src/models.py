# models.py

from sqlalchemy import Column, Integer, DateTime, Float, String
from src.db import Base

class DB_State(Base):
    """
    Database model representing the state of an inference made by the ABI model.

    Attributes:
        id (int): The primary key identifier for the inference record.
        SepalLengthCm (float): The sepal length in centimeters.
        SepalWidthCm (float): The sepal width in centimeters.
        PetalLengthCm (float): The petal length in centimeters.
        PetalWidthCm (float): The petal width in centimeters.
        prediction (str): The predicted class for the input features.
        confidence (float): The confidence score associated with the prediction.
        call_datetime (datetime): The date and time when the inference was made.
    """

    __tablename__ = "Abi_InferenceTable"

    id = Column(Integer, primary_key=True, index=True)
    SepalLengthCm = Column(Float)
    SepalWidthCm = Column(Float)
    PetalLengthCm = Column(Float)
    PetalWidthCm = Column(Float)
    prediction = Column(String)
    confidence = Column(Float)
    call_datetime = Column(DateTime)

    def __repr__(self):
        """
        Returns a string representation of the DB_State object.

        Returns:
            str: A formatted string containing the values of the object's attributes.
        """
        return f"<DB_State(id={self.id}, SepalLengthCm={self.SepalLengthCm}, SepalWidthCm={self.SepalWidthCm}, PetalLengthCm={self.PetalLengthCm}, PetalWidthCm={self.PetalWidthCm}, prediction={self.prediction}, confidence={self.confidence}, call_datetime={self.call_datetime})>"
