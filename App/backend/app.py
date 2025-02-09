import pandas as pd
import joblib
from flask import Flask, request, send_file
from flask_cors import CORS
import socket
import json
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
CORS(app)

# Load pre-trained models
models = {
    "logistic_regression": joblib.load(r"C:\Users\la7tim\Desktop\loanBigData\backend\models\Logistic_Regression_model.pkl"),
    "random_forest": joblib.load(r"C:\Users\la7tim\Desktop\loanBigData\backend\models\Random_Forest_model.pkl"),
    "xgboost": joblib.load(r"C:\Users\la7tim\Desktop\loanBigData\backend\models\XGBoost_model.pkl"),
    "catboost": joblib.load(r"C:\Users\la7tim\Desktop\loanBigData\backend\models\CatBoost_model.pkl"),
}

# Load the pre-trained scaler
scaler_path = r"C:\Users\la7tim\Desktop\loanBigData\backend\models\scaler.pkl"
scaler = joblib.load(scaler_path)

# Required columns
required_columns = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file', 'person_home_ownership_OTHER',
    'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# Preprocessing function
def preprocess_new_data(data):
    # Binary Encoding
    data['person_gender'] = data['person_gender'].map({'female': 0, 'male': 1})
    data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

    # Ordinal Encoding for 'person_education'
    education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
    data['person_education'] = data['person_education'].map(education_order)

    # One-Hot Encoding for 'person_home_ownership' and 'loan_intent'
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Add missing columns with default values
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns based on 'required_columns'
    data = data[required_columns]
    return data

# Function to send data to Logstash
def send_to_logstash(data):
    logstash_host = 'localhost'
    logstash_port = 5044

    # Convert the row data to JSON
    json_data = json.dumps(data)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((logstash_host, logstash_port))
            sock.sendall(json_data.encode('utf-8'))
    except Exception as e:
        print(f"Failed to send data to Logstash: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    df = pd.read_csv(file)

    # Check the selected model
    model_choice = request.form.get('model')
    if model_choice not in models:
        return "Invalid model selected", 400

    model = models[model_choice]

    try:
        # Preprocess the data
        processed_data = preprocess_new_data(df)

        # Scale the data using the pre-trained scaler
        scaled_data = scaler.transform(processed_data)

        # Predict results
        predictions = model.predict(scaled_data)
        df['Prediction'] = predictions

        # Send each row of predictions to Logstash
        for _, row in df.iterrows():
            send_to_logstash(row.to_dict())

        # Save the results to a CSV file
        output_file = 'output_predictions.csv'
        df.to_csv(output_file, index=False)

        # Respond with the file
        return send_file(output_file, as_attachment=True, download_name='output_predictions.csv')

    except Exception as e:
        return f"Prediction failed: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
