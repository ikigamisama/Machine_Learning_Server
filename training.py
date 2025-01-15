import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from module.LoanApprovalPrediction import SkewnessHandler, BinningNumToYN, OneHotWithFeatNames, OrdinalFeatNames


class LoanApprovalPredictionProcess:
    def __init__(self, train_path, test_path, deploy_path):
        self.train_path = train_path
        self.test_path = test_path
        self.deploy_path = deploy_path
        self.pipeline = Pipeline([
            ('skewness_handler', SkewnessHandler()),
            ('binning_num_to_yn', BinningNumToYN()),
            ('one_hot_with_feat_names', OneHotWithFeatNames()),
            ('ordinal_feat_names', OrdinalFeatNames())
        ])
        self.model = XGBClassifier(random_state=42, enable_categorical=False)
        self.feature_names = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        self.X_train = self.train_data.drop(columns=['loan_status'])
        self.y_train = self.train_data['loan_status']
        self.X_test = self.test_data.drop(columns=['loan_status'])
        self.y_test = self.test_data['loan_status']

    def train_model(self):
        # Fit and transform the training data
        self.pipeline.fit(self.X_train, self.y_train)
        X_train_transformed = self.pipeline.transform(self.X_train)
        self.feature_names = list(X_train_transformed.columns)

        # Train the model
        self.model.fit(X_train_transformed, self.y_train)

        # Save the model and pipeline using joblib
        joblib.dump({
            "model": self.model,
            "pipeline": self.pipeline,
            "feature_names": self.feature_names
        }, self.deploy_path)

    def predict(self, new_data):
        # Load pipeline and model using joblib
        loaded_objects = joblib.load(self.deploy_path)
        model = loaded_objects["model"]
        pipeline = loaded_objects["pipeline"]
        feature_names = loaded_objects["feature_names"]

        # Transform and predict
        transformed_data = pipeline.transform(new_data)
        transformed_data = transformed_data.reindex(
            columns=feature_names, fill_value=0)
        predictions = model.predict(transformed_data)
        predictions_bool = ["True" if pred ==
                            1 else "False" for pred in predictions]
        return predictions_bool[0]
