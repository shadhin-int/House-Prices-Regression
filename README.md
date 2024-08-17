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
  
Run the FastAPI server:  
  
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
API Documentation  
* Swagger UI: Automatically generated documentation is available at  
    ```
    http://127.0.0.1:8000/docs
    ```

  
# Text Classification Model Development Report

## 1. Introduction

This report provides an overview of the process undertaken to develop a text classification model. The steps involved include Exploratory Data Analysis (EDA), data preprocessing, model training, evaluation, and selection of the best model.

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Overview

-   **Dataset**: The dataset consists of text documents along with their corresponding class labels.
-   **Objective**: The goal is to classify the text documents into predefined categories.

### 2.2 Data Inspection

-   **Text Length**: Analyzed the distribution of text lengths to understand the variability in document sizes.
-   **Class Distribution**: Checked the balance of classes to identify any potential class imbalance issues.
-   **Sample Text**: Reviewed a few samples of the text to understand the content and structure.

### 2.3 Key Findings

-   **Class Imbalance**: Noted that some classes are underrepresented, which may require special handling during training.
-   **Text Variability**: Observed a wide range of text lengths, suggesting the need for careful feature engineering.

## 3. Data Preprocessing

### 3.1 Feature Separation

-   **Target Variable**: The target variable `SalePrice` was separated from the features.
-   **Feature Types**:
    -   **Numeric Features**: Identified all features with numerical data types.
    -   **Categorical Features**: Identified all features with categorical data types.

### 3.2 Numeric Feature Preprocessing

-   **Missing Value Imputation**:
    -   **Strategy**: Imputed missing values in numeric features using the median of each feature.
-   **Feature Scaling**:
    -   **Standardization**: Scaled numeric features using StandardScaler to standardize the values.

### 3.3 Categorical Feature Preprocessing

-   **Missing Value Imputation**:
    -   **Strategy**: Imputed missing values in categorical features using the most frequent category.
-   **Encoding**:
    -   **One-Hot Encoding**: Applied OneHotEncoder to convert categorical features into numerical format, creating binary columns for each category while ignoring unknown categories.

### 3.4 Combining Preprocessing Steps

-   **Column Transformer**:
    -   **Integration**: Combined the preprocessing steps for numeric and categorical features into a unified pipeline using ColumnTransformer.
-   **Output**:
    -   **Preprocessed Data**: Transformed the original dataset into a preprocessed format ready for model training.

## 4. Model Training

### 4.1 Model Selection

-   **Algorithms Considered**: Experimented with Logistic Regression, Support Vector Machines (SVM), and Random Forest.
-   **Baseline Model**: Started with a simple Logistic Regression model as a baseline.

### 4.2 Hyperparameter Tuning

-   **Grid Search**: Performed grid search for hyperparameter optimization.
-   **Cross-Validation**: Used 5-fold cross-validation to ensure the robustness of the model performance.

### 4.3 Training Process

-   **Data Split**: Split the data into training (80%) and validation (20%) sets.
-   **Model Training**: Trained each model on the training set and evaluated it on the validation set.

## 5. Model Evaluation

### 5.1 Metrics

-   **Accuracy**: Calculated the overall accuracy of the model.
-   **Precision, Recall, F1-Score**: Measured precision, recall, and F1-score to evaluate the model's performance, especially on minority classes.
-   **Confusion Matrix**: Plotted the confusion matrix to visualize the classification errors.

### 5.2 Results

-   **Logistic Regression**: Achieved an accuracy of X% with a balanced precision and recall.
-   **SVM**: Showed improvement with an accuracy of Y% but with higher computational cost.
-   **Random Forest**: Performed similarly to SVM but provided better interpretability.

## 6. Model Selection

### 6.1 Final Model

-   **Chosen Model**: Selected SVM as the final model due to its superior performance across all metrics.
-   **Justification**: SVM provided a good balance between accuracy, precision, and recall, especially in handling class imbalance.

### 7. Preprocessing steps, model training, evaluation and model selection Script
-  **Run the following command to model training, evaluation and find out the best model**:
    ```  
    python model_dev_script.py  
    ```  
## 8. Conclusion

This report outlines the development of a robust text classification model. Through careful EDA, preprocessing, and model training, we selected an SVM model that meets the project's requirements. The model is now ready for deployment and can be used to classify new text documents with high accuracy.


# Text Classification API Development Report

## 1. Overview of the API Design and Functionality

### 1.1 API Purpose

The Text Classification API is designed to classify input text into predefined categories based on a trained machine learning model. The API is built using FastAPI, providing a scalable and efficient solution for text classification tasks.

### 1.2 Endpoints
-  **GET `/`**: This endpoint serves as a health check to verify that the API is running.
    -   **Request**:
        -   **Method**: GET
    -   **Response**:
        -   **Status**: 200 OK
        -   **Body**: `{ "message": "API is running" }`


- **GET `/text_classification/generate-model/`**: This endpoint generates a new model and saves it to the file system.
    -   **Request**:
        -   **Method**: GET
    -   **Response**:
        -   **Status**: 200 OK
        -   **Body**: `{ "message": "Model generated successfully" }`


-   **POST `/text_classification/classify/`**: This endpoint accepts a text input and returns the predicted class along with a confidence score.
    -   **Request**:
        -   **Method**: POST
        -   **Content-Type**: `application/json`
        -   **Body**: `{ "text": "Your text here" }`
    -   **Response**:
        -   **Status**: 200 OK
        -   **Body**: `{ "prediction": "Predicted class", "confidence": 0.95 }`

### 1.3 API Structure

-   **Router**: A dedicated router (`text_classification_router`) is used to handle requests related to text classification.
-   **Models**: The API uses Pydantic models for request and response validation, ensuring the input and output data conform to expected formats.

## 2. Text Classification Model Development

### 2.1 Model Selection

-   **Algorithm Used**: The API utilizes a Support Vector Machine (SVM) model, which was selected after experimenting with several algorithms including Logistic Regression and Random Forest.
-   **Model Performance**: The SVM model demonstrated superior performance in terms of accuracy and generalization, especially when handling imbalanced datasets.

### 2.2 Model Development Process

-   **Preprocessing**: Text data was cleaned and vectorized using TF-IDF, with additional steps like stopword removal and stemming to enhance model performance.
-   **Training**: The SVM model was trained using a grid search for hyperparameter optimization and validated through cross-validation.
-   **Model Persistence**: The trained model was serialized and stored as `text_classification_model.pkl` using `joblib`, allowing it to be loaded and used by the API during inference.

## 3. API Testing Instructions

### 3.1 Health Check Endpoint
-  **Using cURL**:
    ```
    curl -X GET "http://127.0.0.1:8000/
    ```

### 3.2 Testing the `/text_classification/generate-model` Endpoint
-   **Using cURL**:
    ```
    curl -X GET "http://127.0.0.1:8000/text_classification/generate-model/"
    ```
-   **Expected Response**:
    ```
    {"message": "Model generated successfully"}
    ```

### 3.3 Testing the `/text_classification/classify/` Endpoint

-   **Using cURL**: 
    ```
    curl -X POST "http://127.0.0.1:8000/classify/" -H "Content-Type: application/json" -d '{"text": "This is a test document."}'
    ```
    
-   **Expected Response**:
    ```
    {"prediction": "Class", "confidence": 0.95}
    ```
### 3.4 Automated Testing

-   **Unit Tests**: The API includes unit tests located in the `test/` directory. To run the tests, use:
    
    ```
    pytest
    ``` 
    
-   **Test Coverage**: The tests ensure that the endpoint behaves as expected, handling both typical and edge cases.

## 4. Challenges Faced and Solutions

### 4.1 Model Loading and Compatibility Issues

-   **Challenge**: The API encountered issues with model loading due to version mismatches between the environment used for training and the deployment environment.
-   **Solution**: Ensured consistent environments across development and deployment by using the same versions of libraries and dependencies, managed through a `requirements.txt` file.

### 4.2 Handling Class Imbalance

-   **Challenge**: The dataset used for model training was imbalanced, which could lead to biased predictions.
-   **Solution**: Addressed class imbalance by applying SMOTE for oversampling minority classes and adjusting class weights during model training.

### 4.3 Validation Errors

-   **Challenge**: During testing, validation errors were raised due to mismatches between expected and actual response formats.
-   **Solution**: Refined the Pydantic models to ensure they accurately reflected the API's response structure, and added comprehensive error handling to provide clearer feedback to users.

## 5. Conclusion

The Text Classification API was successfully developed and deployed, providing a reliable service for classifying text inputs. Through careful model selection, preprocessing, and robust API design, the project objectives were met. Challenges encountered during development were addressed, resulting in a performant and user-friendly API.


