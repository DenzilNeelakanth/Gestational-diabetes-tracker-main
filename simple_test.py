import requests

# Sample 1: High risk case
data1 = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'Age': 50
}

# Sample 2: Low risk case
data2 = {
    'Pregnancies': 1,
    'Glucose': 85,
    'BloodPressure': 66,
    'SkinThickness': 29,
    'Insulin': 0,
    'BMI': 26.6,
    'Age': 31
}

# Sample 3: Borderline case
data3 = {
    'Pregnancies': 3,
    'Glucose': 120,
    'BloodPressure': 80,
    'SkinThickness': 30,
    'Insulin': 100,
    'BMI': 28.5,
    'Age': 45
}

def test_prediction(data, sample_name):
    print(f"\nTesting {sample_name} case:")
    print("Input values:")
    for key, value in data.items():
        print(f"{key}: {value}")
    
    response = requests.post('http://127.0.0.1:5000/predict', data=data)
    result = response.json()
    
    print("\nPrediction Result:")
    print(f"Status Code: {response.status_code}")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of being Diabetic: {result['probability']['diabetic']:.2%}")
    print(f"Probability of being Non-Diabetic: {result['probability']['non_diabetic']:.2%}")
    print("-" * 50)

# Test all samples
test_prediction(data1, "High Risk")
test_prediction(data2, "Low Risk")
test_prediction(data3, "Borderline") 