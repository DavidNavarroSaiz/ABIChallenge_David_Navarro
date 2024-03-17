from sqlalchemy import Column, Integer, DateTime, Float, String
from src.db import Base

class AbiInferenceTable(Base):
    """
    Database model representing the state of an inference made by the ABI model.

    Attributes:
        id (int): The primary key identifier for the inference record.
        sepal_length_cm (float): The sepal length in centimeters.
        sepal_width_cm (float): The sepal width in centimeters.
        petal_length_cm (float): The petal length in centimeters.
        petal_width_cm (float): The petal width in centimeters.
        prediction (str): The predicted class for the input features.
        confidence (float): The confidence score associated with the prediction.
        call_datetime (datetime): The date and time when the inference was made.
    """

    __tablename__ = "Abi_InferenceTable"

    id = Column(Integer, primary_key=True, index=True)
    sepal_length_cm = Column(Float)
    sepal_width_cm = Column(Float)
    petal_length_cm = Column(Float)
    petal_width_cm = Column(Float)
    prediction = Column(String)
    confidence = Column(Float)
    call_datetime = Column(DateTime)

    def __repr__(self):
        """
        Returns a string representation of the AbiInferenceTable object.

        Returns:
            str: A formatted string containing the values of the object's attributes.
        """
        return f"<AbiInferenceTable(id={self.id}, sepal_length_cm={self.sepal_length_cm}, sepal_width_cm={self.sepal_width_cm}, petal_length_cm={self.petal_length_cm}, petal_width_cm={self.petal_width_cm}, prediction={self.prediction}, confidence={self.confidence}, call_datetime={self.call_datetime})>"
