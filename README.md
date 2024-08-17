# House Prices Regression Using Fast API

## Project Overview
This project is a REST API developed using FastAPI that allows users to submit text data and receive a classification result based on a pre-trained machine learning model. The API handles text classification requests and returns the predicted class along with a confidence score.

## Project Structure

```plaintext
house-prices-regression/
│
├── api
|    └── text_classification.py            # Text classification module
├── data
|    └── sample_data.csv                   # Sample data for testing
├── schemas
|    └── text_classification.py            # Pydantic schemas for request and response
├── model_dev_script.py                    # Script for training and saving the model
├── main.py                                # Main FastAPI application
├── test_main.py                           # Unit tests for the FastAPI endpoints
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
├──text_classification_model.pkl           # Pre-trained text classification model
```

## Requirements
1. FastAPI: Used to create the REST API.
2. Uvicorn: ASGI server for serving the FastAPI application.
3. Scikit-learn or Hugging Face: Used for training or loading a pre-trained text classification model.
4. Pytest: Used for testing the API.
5. Pandas: Used for data manipulation and preprocessing.
6. Numpy: Used for numerical operations.
7. Pickle: Used for saving and loading the pre-trained model.
8. Scikit-learn: Used for training and evaluating the model.
9. Requests: Used for sending HTTP requests to the API.
10. Pydantic: Used for data validation and serialization.

## Installation
Clone the repository: 
```
git clone https://github.com/shadhin-int/House-Prices-Regression
cd House-Prices-Regression
```
Create and activate a virtual environment:
```
python3 -m venv env
source env/bin/activate
```
Install the dependencies:
```
pip install -r requirements.txt
```
Run the model development script to train and save the model:
```
python model_dev_script.py
```

Start the FastAPI server:

- Using Uvicorn:
    ```
    uvicorn main:app --reload
    The API will be available at http://127.0.0.1:8000
    ```
- Using python:
     ```
    python main.py
    The API will be available at http://0.0.0.0.0:8002
    ```


API Endpoint:
1. GET /generate-model: Generates a new model and saves it to the disk.
    Response:
    ```json
    {
      "message": "Model generated successfully"
    }
    ```
2. POST /classify: Accepts a JSON payload with a text field and returns the predicted class and confidence score.
   
   - Request 
       ```json
       {
         "text": "your_text_here"
       }
       ```
   - Response:
       ```json
       {
         "class": "your_predicted_class",
         "confidence": 0.95
       }
       ```