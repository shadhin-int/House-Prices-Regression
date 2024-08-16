import os

import joblib
from fastapi import APIRouter, HTTPException
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from schemas.text_classification import TextRequest, TextResponse

text_classification_router = APIRouter()


@text_classification_router.get("/generate-model")
def generate_model():
    model_path = "text_classification_model.pkl"

    # Delete the existing model if it exists
    if os.path.exists(model_path):
        os.remove(model_path)

    # Load the 20 newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    # Create a pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
    ])

    # Train the model
    text_clf.fit(newsgroups_train.data, newsgroups_train.target)

    # Evaluate the model
    predicted = text_clf.predict(newsgroups_test.data)
    report = classification_report(newsgroups_test.target, predicted, output_dict=True)

    # Save the new model
    joblib.dump(text_clf, model_path)

    return report


@text_classification_router.post("/classify/", response_model=TextResponse)
def classify_text(request_data: TextRequest):
    try:
        # Load the pre-trained model
        model = joblib.load("text_classification_model.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")

    # Predict the class and confidence
    prediction = model.predict([request_data.text])
    probabilities = model.predict_proba([request_data.text])[0]
    confidence = max(probabilities)

    return TextResponse(confidence=confidence, prediction=prediction[0])
