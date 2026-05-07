import os
import logging
import warnings
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define feature names
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Load model and scaler
try:
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading page", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file: {str(e)}")
        return "File not found", 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.form.to_dict()
        
        # Validate input
        for field in FEATURE_NAMES:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({'error': f'Invalid value for {field}'}), 400
        
        # Prepare input data
        input_data = np.array([data[field] for field in FEATURE_NAMES]).reshape(1, -1)
        
        # Scale input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        
        # Format response
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': {
                'diabetic': float(probability[1]),
                'non_diabetic': float(probability[0])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Failed to make prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 