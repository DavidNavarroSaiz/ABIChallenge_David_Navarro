<a name="readme-top"></a>
[![Build](https://github.com/DavidNavarroSaiz/ABIChallenge_David_Navarro/actions/workflows/build.yml/badge.svg)](https://github.com/DavidNavarroSaiz/ABIChallenge_David_Navarro/actions/workflows/build.yml)



<h3 align="center">Abi challenge</h3>

  <p >
    This project is a solution for the AB InBev MLOps Challenge v7 wher eit is implemented a RESTAPI Service ready for production
  </p>


</div>

# Architecture of the solution

<p align="center">
  <img src="./architecture.png" alt="Architecture">
</p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

ML Model Deployment
Expose any already trained machine learning model as an API, please keep working on the
same private repo.***

Your API might meet the following requirements:
• Should be written in Python following OOP standards.
• Should contain inline documentation.
• You should be running some tests using Sonar Cloud.
• The API might be able to make an estimation one by one or in a batch.
• The code might contain a Docker compose file to run the project locally.
• You might be able to store each estimation in any database.





<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* python
* docker
* PostgreSQL
* FastAPI

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

Make sure you have the following prerequisites installed:

- Python (3.7 or higher)
- Git (for version control)
- PostgreSQL
- Docker

* It is recommended to create your own environment for this project:


### Installation
1. Clone the repository to your local machine:

    ```
    git clone https://github.com/DavidNavarroSaiz/ABIChallenge_David_Navarro
    ```

2. Navigate to the project directory:

    ``` 
    cd your-project
    ```

3. Install project dependencies using `pip`:

    ```
    pip install -r requirements.txt
    ```

4. setup the environment variables:
    Follow these steps to set up the necessary environment variables for your project:
    create a new database in PostgreSQL 

    Create a new file named .env in the root directory of your project.

    Open the .env file and add the following line with the URL with the connection with the database:


    ```
    DATABASE_URL="postgresql://postgres:david.123@localhost:5432/<DATABASE_NAME>"
    ```
    replace the <DATABASE_NAME> with your own database name
    Save the .env file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


### Using the FastAPI Service

To use the FastAPI service for interacting with the service, follow these steps:

1. Run the FastAPI service:

    ```
    python main.py
    ```
    or using the console:

    ```
    uvicorn main:app --reload --port 8000
    ```

2. The service will start on a specified port (typically 8000). You can access the API endpoints using tools like `curl` or API testing platforms.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Using the docker:

a docker-compose.yml is used to create the database called "my_dev_database" but it can be changed you can also run the following command:
```
 docker-compose up --build
```

it will create a docker image with the database and the fastapi app.


In both cases you can acces to the fastAPI documentation and usage in the  specified port(typically 8000)
```
http://127.0.0.1:8000/docs#

```

# FastAPI Endpoints Documentation

This documentation provides details about the endpoints available in the FastAPI project.

## Train Model Endpoint

### Endpoint: `/train/`

**Description:**  
Trains the KNN model over the Iris.csv Dataset.

**Method:**  
POST

**Parameters:**  
- `data_file` (optional): Path to the training data file. Defaults to `./src/Iris.csv`.
- `model_directory` (optional): Path to the directory where the trained model will be saved. Defaults to `./src/models`.

**Response:**  
- Success: 200 OK
  - Content: `{"message": "Model trained and saved successfully"}`
- Error: 500 Internal Server Error
  - Content: Details of the error encountered during training.

## Predict Endpoint

### Endpoint: `/predict/`

**Description:**  
Makes batch predictions with the latest trained model.

**Method:**  
POST

**Parameters:**  
- Body:
  - `p_data_list` (required): List of PredictionData instances. Each instance should contain:
    - `sepal_length_cm`: Sepal length in centimeters (float).
    - `sepal_width_cm`: Sepal width in centimeters (float).
    - `petal_length_cm`: Petal length in centimeters (float).
    - `petal_width_cm`: Petal width in centimeters (float).

**Response:**  
- Success: 200 OK
  - Content: Predictions for each input data instance.
- Error: 500 Internal Server Error
  - Content: Details of the error encountered during batch prediction.

## Read All Predictions Endpoint

### Endpoint: `/predictions/`

**Description:**  
Reads all prediction records from the database.

**Method:**  
GET

**Response:**  
- Success: 200 OK
  - Content: A list of dictionaries representing all prediction records.
- Error: 500 Internal Server Error
  - Content: Details of the error encountered while reading predictions from the database.

## Read Prediction Endpoint

### Endpoint: `/prediction/{prediction_id}`

**Description:**  
Reads a prediction record from the database by its ID.

**Method:**  
GET

**Parameters:**  
- `prediction_id` (required): ID of the prediction record to read.

**Response:**  
- Success: 200 OK
  - Content: A dictionary representing the prediction record.
- Error: 
  - 404 Not Found: Prediction not found.
  - 500 Internal Server Error: Details of the error encountered while reading prediction from the database.

## Delete Prediction Endpoint

### Endpoint: `/prediction/{prediction_id}`

**Description:**  
Deletes a prediction record from the database by its ID.

**Method:**  
DELETE

**Parameters:**  
- `prediction_id` (required): ID of the prediction record to delete.

**Response:**  
- Success: 200 OK
  - Content: `{"message": "Prediction deleted successfully"}`
- Error: 500 Internal Server Error
  - Content: Details of the error encountered while deleting prediction from the database.

## Delete All Predictions Endpoint

### Endpoint: `/predictions/`

**Description:**  
Deletes all prediction records from the database.

**Method:**  
DELETE

**Response:**  
- Success: 200 OK
  - Content: `{"message": "All predictions deleted successfully"}`
- Error: 500 Internal Server Error
  - Content: Details of the error encountered while deleting predictions from the database.



