import pandas as pd
import uvicorn
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from training import LoanApprovalPredictionProcess
from spellchecker import SpellChecker
from textblob import TextBlob
from io import BytesIO, StringIO
import base64

app = FastAPI()
spell = SpellChecker()


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
        text = data.get("text", "")

        # Grammar check: basic punctuation rule
        grammar_errors = []
        if not text.strip().endswith(('.', '!', '?')):
            grammar_errors.append(
                f"Sentence should end with punctuation at position {len(text)}")

        # Spelling check with pyspellchecker
        words = text.split()
        misspelled = spell.unknown(words)
        spelling_errors = []
        for word in misspelled:
            corrected_word = spell.correction(word)
            spelling_errors.append(f"'{word}' should be '{corrected_word}'")

        # TextBlob for grammar correction
        blob = TextBlob(text)
        corrected_text = str(blob.correct())

        # Word and sentence statistics
        word_count = len(words)
        # Estimate number of sentences based on periods
        sentence_count = len(text.split('.'))
        avg_word_length = sum(len(word) for word in words) / \
            word_count if word_count > 0 else 0

        return {
            "spelling_errors": spelling_errors,
            "grammar_errors": grammar_errors,
            "statistics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_word_length": avg_word_length
            },
            "corrected_text": corrected_text
        }

    except Exception as e:
        return {"error": str(e)}


@app.post('/visualization/data-chart/')
def data_visualization(data: dict):
    try:
        x_axis = data.get("xAxis")
        y_axis = data.get("yAxis")
        plot_type = data.get("plotType")
        plot_data = pd.DataFrame(data.get("data"))

        # Create the plot
        plt.figure(figsize=(12, 8))

        match plot_type:
            case "Scatter Plot":
                sns.scatterplot(data=plot_data, x=x_axis, y=y_axis)
            case "Line Plot":
                sns.lineplot(data=plot_data, x=x_axis, y=y_axis)
            case "Bar Plot":
                sns.barplot(data=plot_data, x=x_axis, y=y_axis)
            case "Count Plot":
                sns.countplot(data=plot_data, x=x_axis, hue=y_axis)
            case "Heatmap":
                pivot_data = plot_data.pivot_table(
                    index=x_axis, columns=y_axis, aggfunc="size", fill_value=0)
                sns.heatmap(pivot_data, annot=True, fmt="d", cmap="Blues")
            case "Box Plot":
                sns.boxplot(data=plot_data, x=x_axis, y=y_axis)
            case "Violin Plot":
                sns.violinplot(data=plot_data, x=x_axis, y=y_axis)
            case "Swarm Plot":
                sns.swarmplot(data=plot_data, x=x_axis, y=y_axis)
            case "Joint Plot":
                sns.jointplot(data=plot_data, x=x_axis,
                              y=y_axis, kind="scatter")
            case "Pair Plot":
                sns.pairplot(plot_data, vars=[x_axis, y_axis])
            case "Hexbin Plot":
                plt.hexbin(plot_data[x_axis], plot_data[y_axis],
                           gridsize=25, cmap="Blues")
                plt.colorbar(label="Counts")
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
            case _:
                return {"error": f"Unsupported plot type: {plot_type}"}

        plt.title(f"{plot_type} of {x_axis} vs {y_axis}")
        plt.xticks(rotation=90)
        # Save the plot to a BytesIO stream
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": base64_image}

    except Exception as e:
        return {"error": str(e)}


@app.post('/data-visualization/csv_file')
async def csv_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        content = await file.read()
        csv_data = StringIO(content.decode("utf-8"))
        df = pd.read_csv(csv_data)

        num_rows, num_columns = df.shape
        column_names = list(df.columns)
        data = df.to_dict(orient="records")

        return {
            "message": "CSV file processed successfully.",
            "file_name": file.filename,
            "file_size": len(content),
            "num_rows": num_rows,
            "num_columns": num_columns,
            "columns": column_names,
            "data": data,
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
