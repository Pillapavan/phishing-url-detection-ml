from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phishing_detection.components.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="Phishing URL Detection API",
    description="ML API for URL classification",
    version="1.0.0"
)


class URLRequest(BaseModel):
    url: str


MODEL_PATH = "artifacts/09_30_2026_21_39_39/model_trainer/trained_model/model.pkl"
PROCESSOR_PATH = "artifacts/09_30_2026_21_39_39/data_transformation/processor.pkl"

prediction_pipeline = PredictionPipeline(
    model_path=MODEL_PATH,
    processor_path=PROCESSOR_PATH
)


@app.get("/")
def root():
    return {
        "message": "Phishing URL Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_url(request: URLRequest):

    try:
        result = prediction_pipeline.initiate_prediction(request.url)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )