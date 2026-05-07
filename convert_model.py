import joblib
import xgboost as xgb

# Load the pickle model
model = joblib.load('diabetes_model.pkl')

# Save in XGBoost's native format
model.save_model('diabetes_model.json')

print("Model converted successfully!") 