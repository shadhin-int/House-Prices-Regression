import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def sample_request_data():
    return {"text": "This is a test document."}


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data == {"status": "API is running smoothly"}


def test_generate_model():
    response = client.get("/text_classification/generate-model")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data == {"message": "Model generated successfully"}


def test_classify_text(sample_request_data):
    response = client.post("/text_classification/classify/", json=sample_request_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert "confidence" in response_data
    assert isinstance(response_data["prediction"], int)
    assert isinstance(response_data["confidence"], float)
