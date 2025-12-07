# 💤 DreamMetrics – Sleep Quality Prediction Web App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![Firebase](https://img.shields.io/badge/Firebase-Auth%20%26%20Firestore-orange)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**DreamMetrics** is a full-stack machine learning web application designed to predict a user’s sleep quality score based on lifestyle, health, and demographic inputs. Beyond simple prediction, it provides a detailed, health-focused breakdown with personalized recommendations to help users improve their sleep hygiene.

The system combines trained regression models (Random Forest, Decision Tree, Linear Regression), a Flask backend, and a secure Firebase infrastructure.

🔗 **Repository:** [https://github.com/pprbkt/DreamMetrics](https://github.com/pprbkt/DreamMetrics)

---

## 🚀 Key Features

* **Multi-Model ML Engine:** Trains and evaluates Linear Regression, Decision Trees, and Random Forest models. The app automatically selects the best-performing model (highest $R^2$) for production.
* **Deep Insights:**
    * Predicts "Quality of Sleep" score.
    * Classifies sleep as **Excellent**, **Good**, **Fair**, or **Poor**.
    * Provides sub-scores for **Sleep**, **Lifestyle**, and **Health**.
* **Smart Recommendations:** Generates categorized advice (Critical Issues, Needs Improvement, Good Habits) based on input data.
* **Secure Architecture:** User authentication via **Firebase Auth** and persistent history storage using **Cloud Firestore**.
* **Robust Validation:** Double-layer validation (Client-side JS & Server-side Python) ensures data integrity (e.g., Age 18-100, BP ranges).
* **Interactive UI:** A modern, multi-step wizard frontend with real-time feedback and responsive design.

---

## 🛠 Tech Stack

### Backend & ML
* **Language:** Python
* **Framework:** Flask
* **Machine Learning:** Scikit-learn, Pandas, Numpy, Joblib
* **Environment:** Python-dotenv

### Cloud & Database
* **Authentication:** Firebase Auth
* **Database:** Firebase Firestore (Native Mode)
* **SDK:** Firebase Admin SDK

### Frontend
* **Core:** HTML5, Custom CSS
* **Logic:** Vanilla JavaScript (ES6+)

---

## 📂 Project Structure

```text
DreamMetrics/
├── app.py                      # Main Flask application entry point
├── train_model.py              # ML Pipeline: Preprocessing, Training, Evaluation
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (Not committed)
├── firebase-credentials.json   # Firebase Admin key (Not committed)
├── Sleep_health_and_lifestyle_dataset.csv  # Source dataset
│
├── models/ (Generated)
│   ├── models_all.pkl          # Dictionary of trained models
│   ├── scaler.pkl              # StandardScaler object
│   ├── label_encoders.pkl      # Encoders for categorical data
│   └── model_info.json         # Metadata and performance metrics
│
└── templates/
    ├── index.html              # Main prediction wizard UI
    ├── login.html              # Auth page
    ├── signup.html             # Auth page
    └── history.html            # User prediction history
```

## 📊 Data & Modeling

The application uses the Sleep Health and Lifestyle Dataset.

1. Preprocessing
   Cleaning: Missing sleep disorders are labeled "None"; numeric NaNs filled with medians.
   Feature Engineering: Blood Pressure is split into Systolic and Diastolic.
   Encoding: Label encoding for Gender, Occupation, BMI Category, and Sleep Disorder.
   Scaling: StandardScaler applied to numerical features.

2. Model Evaluation

The `train_model.py` script splits data (80/20 train/test) and performs 5-fold cross-validation. Models are evaluated on:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* $R^2$ Score (Coefficient of Determination)

The system defaults to the model with the highest Test $R^2$.

## ⚙️ Installation & Setup

1. Clone the Repository

```bash
git clone https://github.com/pprbkt/DreamMetrics.git
cd DreamMetrics
```

2. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Configure Firebase

Create a project in the Firebase Console.

Enable Authentication (Email/Password).

Enable Firestore Database (Create in Native mode).

Generate a Service Account Private Key:

Go to Project Settings -> Service Accounts -> Generate New Private Key.

Rename the downloaded JSON file to `firebase-credentials.json` and place it in the project root.

5. Environment Variables

Create a `.env` file in the project root and add your Firebase Web Config (found in Project Settings -> General -> Your Apps):

```env
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_project_id.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
SECRET_KEY=your_flask_secret_key
```

6. Train the Models

Run the training script to generate the `.pkl` files and model metadata:

```bash
python train_model.py
```

7. Run the Application

```bash
python app.py
```

Access the app at http://127.0.0.1:5000/.

## 📝 Usage

Sign Up/Login: Create an account to access the prediction tools.

Select Model: (Optional) On the dashboard, view model performance metrics and switch between Random Forest, Decision Tree, etc.

Input Data: Complete the 4-step wizard (Personal -> Lifestyle -> Health -> Review).

Note: Inputs like Age (18-100) and Sleep Duration (3-12h) are validated.

View Results: Get your Sleep Quality Score, Health Breakdown, and tailored recommendations.

History: Visit the `/history` page to track how your sleep predictions have changed over time.

## 🔮 Future Improvements

* Integrate Gradient Boosting (XGBoost/LightGBM) models.
* Add data visualization charts to the User History page.
* Implement Dark Mode.
* Email notifications for "Critical" sleep scores.