import pickle
import pandas as pd
import sys

from phishing_detection.components.url_feature_extractor import URLFeatureExtractor
from phishing_detection.exception.exception import CustomException


class PredictionPipeline:

    def __init__(self, model_path, processor_path):
        self.model_path = model_path
        self.processor_path = processor_path
        self.feature_extractor = URLFeatureExtractor()

    def load_model(self):
        try:
            with open(self.model_path, "rb") as file:
                model = pickle.load(file)

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def load_processor(self):
        try:
            with open(self.processor_path, "rb") as file:
                processor = pickle.load(file)

            return processor

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, url: str):

        try:
            # 1. Extract 22 features from raw URL
            features = self.feature_extractor.extract_features(url)

            # 2. Convert dictionary to DataFrame
            features_df = pd.DataFrame([features])

            # 3. Load preprocessing object
            processor = self.load_processor()

            # 4. Apply same preprocessing used during training
            transformed_features = processor.transform(features_df)

            # 5. Load trained Random Forest
            model = self.load_model()

            # 6. Predict
            prediction = model.predict(transformed_features)[0]

            # 7. Probability
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    transformed_features
                )[0]

                probability = float(probabilities[1])

            # Dataset:
            # 0 = Legitimate
            # 1 = Phishing

            result = "Phishing" if prediction == 1 else "Legitimate"

            return {
                "url": url,
                "prediction": result,
                "probability": probability
            }

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_prediction(self, url: str):
        return self.predict(url)