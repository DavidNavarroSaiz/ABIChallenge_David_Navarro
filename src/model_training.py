import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

class KNNModel_Trainning:
    """
    A class to train and save a K-Nearest Neighbors classifier model.
    
    Attributes:
        data_file (str): The path to the CSV file containing the dataset.
        model_dir (str): The directory to save the trained model.
    """
    
    def __init__(self, data_file, model_directory):
        """
        Initializes the KNNModel with data file and model directory.
        
        Args:
            data_file (str): The path to the CSV file containing the dataset.
            model_dir (str): The directory to save the trained model.
        """
        self.data_file = data_file
        self.model_dir = model_directory
    
    def train_and_save_model(self):
        """
        Trains a K-Nearest Neighbors classifier model using the provided dataset 
        and saves the trained model to a file.
        """
        print("training new model")
        # Load dataset
        df = pd.read_csv(self.data_file)
        print(df.head())

        # Preprocess data
        iris = df.drop("Id", axis= 1)
        x = iris.drop("Species", axis=1)
        y = iris["Species"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train, y_train)

        # Find the next available model number
        model_number = 1
        while os.path.exists(os.path.join(self.model_dir, f'knn_model_{model_number}.joblib')):
            model_number += 1

        print("model_number",model_number)
        # Save the trained model to a file
        model_file = os.path.join(self.model_dir, f'knn_model_{model_number}.joblib')
        dump(knn, model_file)

        # Evaluate model
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)*100
        formatted_accuracy = "{:.2f}".format(accuracy)
        message = f" Trained knn_model_{model_number} Test Accuracy: {formatted_accuracy}%"

        return message
# Example usage
if __name__ == "__main__":
    # Instantiate KNNModel
    knn_model = KNNModel_Trainning(data_file='./Iris.csv', model_dir='./models')
    
    # Train and save the model
    knn_model.train_and_save_model()
