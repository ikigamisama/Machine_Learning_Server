import pandas as pd
import uvicorn
from fastapi import FastAPI
from training import LoanApprovalPredictionProcess

app = FastAPI()


@app.post('/predict/loan-application/')
def predict_loan_application(data: dict):
    try:
        # Initialize the LoanApprovalPrediction class
        model_path = 'deploy/LoanApprovalPrediction/complete_model_loan_prediction.joblib'
        loan_predictor = LoanApprovalPredictionProcess(
            'dataset/LoanApprovalPrediction/train.csv', 'dataset/LoanApprovalPrediction/test.csv', model_path)
        loan_predictor.load_data()
        loan_predictor.train_model()

        column_list = [
            "person_age", "person_gender", "person_education", "person_income",
            "person_emp_exp", "person_home_ownership", "loan_amnt",
            "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score",
            "previous_loan_defaults_on_file"
        ]

        new_data = pd.DataFrame([data], columns=column_list)

        result = loan_predictor.predict(new_data)

        return {"loan_approved": result}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
