from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the form
            input_data = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            
            # Convert input data to numpy array
            input_data_np = np.asarray(input_data).reshape(1, -1)
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data_np)
            
            # Make prediction
            prediction = model.predict(input_data_scaled)
            
            result = "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have Heart Disease"
            
            return render_template('index.html', prediction_text=result)
        except ValueError:
            return render_template('index.html', prediction_text="Invalid input. Please enter valid numeric values.")

if __name__ == '__main__':
    app.run(debug=True)
