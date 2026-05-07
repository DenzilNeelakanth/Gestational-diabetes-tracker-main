import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Pregnancies": "2",
    "Glucose": "120",
    "BloodPressure": "70",
    "SkinThickness": "20",
    "Insulin": "79",
    "BMI": "32.0",
    "Age": "25"
}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print("Response Headers:", response.headers)
    print("\nResponse Body:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
