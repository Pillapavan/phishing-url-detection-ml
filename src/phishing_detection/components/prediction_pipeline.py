import pickle
import pandas as pd

from phishing_detection.components.url_feature_extractor import URLFeatureExtractor
from phishing_detection.exception.exception import CustomException
import sys


class PredictionPipeline:

    def __init__(self, model_path, processor_path):
        self.model_path = model_path
        self.processor_path = processor_path
        self.feature_extractor = URLFeatureExtractor()

    def load_model(self):
        try:
            with open(self.model_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise CustomException(e, sys)

    def load_processor(self):
        try:
            with open(self.processor_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, url: str):

        try:
            # 1. Extract features from URL
            features = self.feature_extractor.extract_features(url)

            # 2. Convert to DataFrame
            features_df = pd.DataFrame([features])

            # 3. Load preprocessing object
            processor = self.load_processor()

            # 4. Transform features
            transformed_features = processor.transform(features_df)

            # 5. Load trained model
            model = self.load_model()

            # 6. Prediction
            prediction = model.predict(transformed_features)[0]

            # 7. Probability
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_features)[0]
                probability = float(probabilities[1])

            # 8. Convert prediction to business result
            result = "Phishing" if prediction == 0 else "Legitimate"

            return {
                "url": url,
                "prediction": result,
                "probability": probability
            }

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_prediction(self, url: str):
        return self.predict(url)