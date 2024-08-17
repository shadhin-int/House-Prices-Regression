import logging
import uvicorn
from fastapi import FastAPI

from api.text_classification import text_classification_router

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

app.include_router(text_classification_router, prefix="/text_classification", tags=["Text Classification"])


@app.get("/")
def health_check():
	return {"status": "API is running smoothly"}


if __name__ == '__main__':
	host = "0.0.0.0"
	port = 8002
	LOG.info(f"Starting server at {host}:{port}")
	uvicorn.run("main:app", host=host, port=port, reload=True)
