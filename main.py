import pandas as pd
import uvicorn
import spacy
from fastapi import FastAPI
from training import LoanApprovalPredictionProcess
from textblob import TextBlob

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


@app.post('/nlp/grammar-spelling-checker')
def nlp_grammar_spelling(data: dict):
    try:
        nlp = spacy.load('en_core_web_sm')
        text = data.get("text", "")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        grammar_errors = []
        if not text.strip().endswith(('.', '!', '?')):
            grammar_errors.append(
                f"Sentence should end with punctuation at position {len(text)}")
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        spelling_errors = []
        for word, corrected_word in zip(text.split(), corrected_text.split()):
            if word.lower() != corrected_word.lower():
                spelling_errors.append(
                    f"'{word}' should be '{corrected_word}'")

        words = [token.text for token in doc if token.is_alpha]
        word_count = len(words)
        sentence_count = len(list(doc.sents))
        avg_word_length = sum(len(word) for word in words) / \
            word_count if word_count > 0 else 0

        return {
            "entities_found": entities,
            "grammar_errors": grammar_errors,
            "spelling_errors": spelling_errors,
            "statistics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_word_length": avg_word_length
            },
            "corrected_text": corrected_text
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
