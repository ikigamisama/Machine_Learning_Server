import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from module.LoanApprovalPrediction import SkewnessHandler, BinningNumToYN, OneHotWithFeatNames, OrdinalFeatNames

app = FastAPI()


@app.post('/predict/loan-application/')
def predict_loan_application(data: dict):
    try:
        loaded_objects = joblib.load(
            'deploy/LoanAPprovalPrediction/complete_model.pkl')
        model = loaded_objects["model"]
        feature_names = loaded_objects["feature_names"]

        pipeline = Pipeline([
            ('skewness_handler', SkewnessHandler()),
            ('binning_num_to_yn', BinningNumToYN()),
            ('one_hot_with_feat_names', OneHotWithFeatNames()),
            ('ordinal_feat_names', OrdinalFeatNames())
        ])

        column_list = [
            "person_age", "person_gender", "person_education", "person_income",
            "person_emp_exp", "person_home_ownership", "loan_amnt",
            "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score",
            "previous_loan_defaults_on_file"
        ]
        new_data = pd.DataFrame([data], columns=column_list)
        transformed_data = pipeline.fit_transform(new_data)
        transformed_data = transformed_data.reindex(
            columns=feature_names, fill_value=0)

        prediction = model.predict(transformed_data)[0]
        result = "True" if prediction == 1 else "False"
        return {"loan_approved": result}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
