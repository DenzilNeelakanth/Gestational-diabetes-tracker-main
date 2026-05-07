import joblib
import numpy as np

try:
    print("Loading model...")
    model = joblib.load('diabetes_model.pkl')
    print("Model type:", type(model))
    
    print("\nLoading scaler...")
    scaler = joblib.load('scaler.pkl')
    print("Scaler type:", type(scaler))
    
    # Test data (same as in test_request.py)
    test_data = np.array([[2, 120, 70, 20, 79, 32.0, 25]])
    print("\nTest data shape:", test_data.shape)
    
    print("\nScaling data...")
    scaled_data = scaler.transform(test_data)
    print("Scaled data shape:", scaled_data.shape)
    
    print("\nMaking prediction...")
    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    
    print("\nResults:")
    print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
    print("Probabilities:", probabilities[0])
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    print(f"Error type: {type(e)}")
    raise 