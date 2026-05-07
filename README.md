# Gestational-diabetes-tracker

🌟 Overview
Gestational Diabetes Tracker is an intelligent, user-friendly web application designed to help pregnant women and healthcare providers predict, track, and manage Gestational Diabetes Mellitus (GDM). By combining advanced machine learning with a secure and intuitive interface, this project empowers users to monitor their health, receive actionable insights, and make informed decisions throughout pregnancy.

🚀 Features
Real-Time GDM Risk Prediction:
Enter clinical and demographic data to instantly receive a risk assessment powered by a trained XGBoost machine learning model.

Personalized Health Insights:
Get clear explanations and recommendations based on your unique health profile, with SHAP-based interpretability for every prediction.

History and Trend Tracking:
Securely store and review your previous predictions, enabling both patients and clinicians to monitor trends and intervene early.

User Authentication:
Register and log in to keep your data private and accessible only to you.

Modern, Responsive Interface:
Clean design with Bootstrap, accessible on any device and browser.

Security First:
Passwords are hashed, sessions are managed securely, and all data is validated for integrity and privacy.

🩺 Why Gestational Diabetes?
Gestational diabetes affects millions of pregnancies worldwide each year. Early detection and regular monitoring are crucial to prevent complications for both mother and child. This tracker aims to bridge the gap between periodic clinical screenings and the need for continuous, personalized health management.

🛠️ Tech Stack
Backend: Python, Flask, Flask-SQLAlchemy, Flask-Login, Flask-CORS
Machine Learning: XGBoost, Scikit-learn, SHAP, Joblib
Frontend: HTML5, CSS3 (Bootstrap), JavaScript
Database: SQLite
Other: Jupyter Notebook, Visual Studio Code


🖥️ Installation & Setup


1.Clone the repository:

bash
git clone https://github.com/AnuAK03/Gestational-diabetes-tracker.git
cd Gestational-diabetes-tracker

2.Create a virtual environment & activate it:

bash
python -m venv venv


3.Install dependencies:

bash
pip install -r requirements.txt

4.Run the application:

bash
python app.py

The app will be available at http://localhost:5000.



📄 License
This project is licensed under the MIT License.

💬 Acknowledgements
Thanks to the open-source community, healthcare professionals, and all contributors for making this project possible!

Empowering mothers and clinicians—one prediction at a time.

Happy tracking! 👶🩺🌸
