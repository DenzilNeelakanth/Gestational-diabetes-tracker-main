import os
import logging
import warnings
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash
import joblib
import numpy as np
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, TrackingHistory
from datetime import datetime, timedelta
import shap
import random
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning)
 
# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key'  # Use a long, random string in production!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Define feature names
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'Age'
]

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model and scaler
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    logger.info(f"Loading scaler from: {SCALER_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Initialize SHAP explainer
try:
    # Create a background dataset for SHAP
    background_data = np.zeros((100, len(FEATURE_NAMES)))
    background_data_scaled = scaler.transform(background_data)
    explainer = shap.TreeExplainer(model, background_data_scaled)
    logger.info("SHAP explainer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing SHAP explainer: {str(e)}")
    raise

@app.route('/')
@login_required
def home():
    try:
        return render_template('dashboard.html')
    except Exception as e:
        import traceback
        logger.error(f"Error rendering template: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error loading page: {str(e)}", 500

@app.route('/prediction')
@login_required
def prediction_page():
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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
            
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
            
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        message = request.json.get('message', '').lower()
        response = generate_chat_response(message)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_chat_response(message):
    # Predefined responses for common questions
    responses = {
        'what is diabetes': 'Diabetes is a chronic condition that affects how your body turns food into energy. There are two main types: Type 1 and Type 2.',
        'type 1 diabetes': 'Type 1 diabetes occurs when your body doesn\'t produce insulin. It\'s usually diagnosed in children and young adults.',
        'type 2 diabetes': 'Type 2 diabetes occurs when your body doesn\'t use insulin properly. It\'s the most common type of diabetes.',
        'symptoms': 'Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, and blurred vision.',
        'prevention': 'You can reduce your risk by maintaining a healthy weight, eating a balanced diet, exercising regularly, and avoiding smoking.',
        'treatment': 'Treatment typically includes lifestyle changes, monitoring blood sugar, and may involve insulin or other medications.',
        'diet': 'A healthy diabetes diet includes whole grains, lean proteins, vegetables, and fruits. Limit processed foods and sugary drinks.',
        'exercise': 'Regular exercise helps control blood sugar levels. Aim for 150 minutes of moderate activity per week.',
        'blood sugar': 'Normal blood sugar levels are typically between 70-140 mg/dL. However, target ranges may vary based on individual circumstances.',
        'complications': 'Long-term complications can include heart disease, kidney damage, nerve damage, and eye problems.',
        'insulin': 'Insulin is a hormone that helps glucose enter your cells. People with Type 1 diabetes need insulin injections, while some with Type 2 may also require it.',
        'monitoring': 'Regular monitoring of blood sugar levels is crucial for diabetes management. This can be done through blood glucose meters or continuous glucose monitors.',
        'help': 'I can provide information about diabetes types, symptoms, prevention, treatment, diet, exercise, and more. Just ask your question!'
    }
    
    # Check for keywords in the message
    for key in responses:
        if key in message:
            return responses[key]
    
    # Default response if no keywords are found
    return "I'm a diabetes assistant. You can ask me about diabetes types, symptoms, prevention, treatment, diet, exercise, or monitoring. How can I help you?"

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get input data
        data = request.form.to_dict()
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate input
        for field in FEATURE_NAMES:
            if field not in data:
                error_msg = f'Missing field: {field}'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
            try:
                data[field] = float(data[field])
            except ValueError:
                error_msg = f'Invalid value for {field}. Please enter a valid number.'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
        
        # Prepare input data
        input_data = np.array([data[field] for field in FEATURE_NAMES]).reshape(1, -1)
        logger.info(f"Prepared input data: {input_data}")
        
        # Scale input data
        try:
            scaled_data = scaler.transform(input_data)
            logger.info(f"Scaled data: {scaled_data}")
        except Exception as e:
            error_msg = f"Error scaling input data: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Make prediction
        try:
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            
            prediction_result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
            confidence = float(probability[1] if prediction == 1 else probability[0]) * 100
            
            # Save to user history
            history = TrackingHistory(
                user_id=current_user.id,
                pregnancies=data['Pregnancies'],
                glucose=data['Glucose'],
                blood_pressure=data['BloodPressure'],
                skin_thickness=data['SkinThickness'],
                insulin=data['Insulin'],
                bmi=data['BMI'],
                age=data['Age'],
                prediction=prediction_result,
                probability_diabetic=float(probability[1]),
                probability_non_diabetic=float(probability[0]),
                timestamp=datetime.utcnow()
            )
            db.session.add(history)
            db.session.commit()
            
            # Generate SHAP values
            shap_values = explainer.shap_values(scaled_data)
            feature_importance = dict(zip(FEATURE_NAMES, np.abs(shap_values[0])))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Generate SHAP explanations
            shap_explanations = []
            for feature, importance in sorted_features:
                value = float(data[feature])
                impact = float(shap_values[0][FEATURE_NAMES.index(feature)])
                direction = "increases" if impact > 0 else "decreases"
                shap_explanations.append({
                    'feature': feature,
                    'value': value,
                    'impact': abs(impact),
                    'direction': direction,
                    'importance': float(importance)
                })
            
            # Generate health insights
            insights = generate_health_insights(data, prediction_result, probability)
            
            # Format response
            result = {
                'prediction': prediction_result,
                'probability': {
                    'diabetic': float(probability[1]),
                    'non_diabetic': float(probability[0])
                },
                'feature_importance': [
                    {
                        'feature': feature,
                        'importance': float(importance)
                    } for feature, importance in sorted_features
                ],
                'shap_explanations': shap_explanations,
                'health_insights': insights
            }
            
            return jsonify(result)
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

def generate_health_insights(data, prediction_result, probability):
    insights = []
    
    # Glucose Analysis
    glucose = float(data['Glucose'])
    # GDM-specific thresholds
    if glucose >= 95 and glucose < 140:
        insights.append({
            'category': 'Glucose',
            'level': 'warning',
            'message':'Dawn phenomenon (natural rise in blood sugar in the early morning), insufficient insulin overnight (if on medication), or evening snack not adequate or appropriate'
        })
    elif glucose >= 140 and glucose < 200:
        insights.append({
            'category': 'Glucose',
            'level': 'warning',
            'message': 'Too many carbohydrates in the meal, high glycemic index foods, or insufficient insulin response to that particular meal.'

        })
    elif glucose >= 120:
        insights.append({
            'category': 'Glucose',
            'level': 'warning',
            'message': 'Similar to high 1-hour post-meal readings, but may also indicate a delayed insulin response.'

        })
    
    # BMI Analysis
    bmi = float(data['BMI'])
    if bmi < 18.5:
        insights.append({
            'category': 'BMI',
            'level': 'info',
            'message': 'Your BMI indicates that you are underweight (BMI < 18.5) Being underweight can be associated with health risks such as nutritional deficiencies, weakened immune system, and potential complications with bones and fertility.'
             'Its important to maintain a balanced diet and consult with a healthcare provider or a registered dietitian to assess your overall health and create a suitable nutrition plan.'
        })
    elif bmi > 25:
        insights.append({
            'category': 'BMI',
            'level': 'warning',
            'message': 'Your BMI indicates that you are overweight (BMI > 25). Carrying excess weight can increase the risk of several health conditions, including heart disease, type 2 diabetes, high blood pressure, and certain types of cancer.Its recommended to adopt a healthy lifestyle with regular physical activity, a balanced diet, and routine health checkups.Please consider consulting a healthcare professional for personalized guidance.'
        })
    
    # Blood Pressure Analysis
    bp = float(data['BloodPressure'])
    if bp < 90: 
        insights.append({
            'category': 'Blood Pressure',
            'level': 'warning',
            'message': 'Your blood pressure is low . Low blood pressure during pregnancy may cause symptoms like dizziness, fainting, fatigue, blurred vision, and nausea. It can also reduce blood flow to the placenta, which may affect the baby development.'
             'Please stay hydrated, avoid sudden movements, and consult your healthcare provider for further evaluation.'
        })
    elif bp > 120:
        insights.append({
            'category': 'Blood Pressure',
            'level': 'warning',
            'message': 'Your blood pressure is above normal range. Regular exercise and reduced salt intake may help.'
        })
    
    # Age and Risk Factors
    age = float(data['Age'])
    if age > 45:
        insights.append({
            'category': 'Age',
            'level': 'info',
            'message': 'As you are over 45, regular health check-ups are recommended for early detection of any health issues.'
        })
    
    # Pregnancy Analysis
    pregnancies = float(data['Pregnancies'])
    if pregnancies > 0:
        insights.append({
            'category': 'Pregnancy History',
            'level': 'info',
            'message': 'Previous pregnancies can affect diabetes risk. Regular monitoring of blood sugar levels is recommended.'
        })
    
    # Insulin Analysis
    insulin = float(data['Insulin'])
    if insulin < 16:
        insights.append({
            'category': 'Insulin',
            'level': 'warning',
            'message': 'Your insulin level is below normal range. This might indicate insulin resistance.'
        })
    elif insulin > 166:
        insights.append({
            'category': 'Insulin',
            'level': 'warning',
            'message': 'Your insulin level is above normal range. This might indicate insulin resistance or metabolic syndrome.'
        })
    
    # Overall Risk Assessment
    if prediction_result == 'Diabetic':
        risk_level = 'high' if probability[1] > 0.8 else 'moderate'
        insights.append({
            'category': 'Overall Risk',
            'level': 'warning',
            'message': f'Based on your health parameters, you have a {risk_level} risk of diabetes. Regular monitoring and lifestyle modifications are recommended.'
        })
    else:
        insights.append({
            'category': 'Overall Risk',
            'level': 'info',
            'message': 'Your current health parameters are within normal ranges. Maintain a healthy lifestyle to reduce future risk.'
        })
    
    # Lifestyle Recommendations
    if prediction_result == 'Diabetic':
        insights.extend([
            {
                'category': 'Dietary Recommendations',
                'level': 'info',
                'message': 'Focus on a balanced diet with controlled portions. Include whole grains, lean proteins, and plenty of vegetables. Limit refined carbohydrates and sugary foods.'
            },
            {
                'category': 'Physical Activity',
                'level': 'info',
                'message': 'Aim for 150 minutes of moderate exercise weekly. Include both aerobic activities (walking, swimming) and strength training. Exercise helps improve insulin sensitivity.'
            },
            {
                'category': 'Blood Sugar Monitoring',
                'level': 'info',
                'message': 'Regularly monitor your blood sugar levels. Keep a log of your readings and share them with your healthcare provider during check-ups.'
            },
            {
                'category': 'Stress Management',
                'level': 'info',
                'message': 'Practice stress-reduction techniques like meditation, deep breathing, or yoga. Stress can affect blood sugar levels.'
            },
            {
                'category': 'Sleep Hygiene',
                'level': 'info',
                'message': 'Maintain a regular sleep schedule. Aim for 7-8 hours of quality sleep. Poor sleep can affect blood sugar control.'
            },
            {
                'category': 'Hydration',
                'level': 'info',
                'message': 'Stay well-hydrated by drinking plenty of water throughout the day. Limit sugary drinks and alcohol.'
            }
        ])
    else:
        insights.extend([
            {
                'category': 'Preventive Measures',
                'level': 'info',
                'message': 'Maintain a healthy lifestyle to prevent diabetes. Regular check-ups and monitoring of blood sugar levels are recommended.'
            },
            {
                'category': 'Dietary Habits',
                'level': 'info',
                'message': 'Follow a balanced diet rich in whole foods, vegetables, and lean proteins. Limit processed foods and sugary beverages.'
            },
            {
                'category': 'Physical Activity',
                'level': 'info',
                'message': 'Engage in regular physical activity. Aim for at least 30 minutes of moderate exercise most days of the week.'
            },
            {
                'category': 'Weight Management',
                'level': 'info',
                'message': 'Maintain a healthy weight through balanced diet and regular exercise. Even small weight loss can significantly reduce diabetes risk.'
            },
            {
                'category': 'Regular Check-ups',
                'level': 'info',
                'message': 'Schedule regular health check-ups to monitor blood sugar levels and other health parameters.'
            },
            {
                'category': 'Healthy Habits',
                'level': 'info',
                'message': 'Avoid smoking and limit alcohol consumption. These habits can increase the risk of developing diabetes and other health conditions.'
            }
        ])
    
    return insights

@app.route('/history')
@login_required
def history():
    user_history = TrackingHistory.query.filter_by(user_id=current_user.id).order_by(TrackingHistory.timestamp.desc()).all()
    return render_template('history.html', history=user_history)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
