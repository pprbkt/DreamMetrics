from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import numpy as np
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from functools import wraps
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase-credentials.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load all trained models and preprocessing objects
models_all = joblib.load('models_all.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print("=" * 60)
print("DREAMMETRICS - Flask App Started")
print(f"Models loaded: {', '.join(models_all.keys())}")
print(f"Best Model: {model_info['best_model_name']}")
print("=" * 60)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/login')
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/signup')
def signup():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/home')
@login_required
def home():
    user_email = session.get('user_email', 'User')
    return render_template('index.html', model_info=model_info, user_email=user_email)

@app.route('/history')
@login_required
def history():
    user_id = session.get('user_id')
    
    # Get user's prediction history from Firestore
    predictions_ref = db.collection('predictions').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20)
    predictions = []
    
    for doc in predictions_ref.stream():
        pred_data = doc.to_dict()
        pred_data['id'] = doc.id
        predictions.append(pred_data)
    
    return render_template('history.html', predictions=predictions, user_email=session.get('user_email'))

@app.route('/auth/verify', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data.get('idToken')
        
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        user_email = decoded_token.get('email', '')
        
        # Store in session
        session['user_id'] = user_id
        session['user_email'] = user_email
        
        # Create or update user document in Firestore
        user_ref = db.collection('users').document(user_id)
        user_ref.set({
            'email': user_email,
            'last_login': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return jsonify({'success': True, 'message': 'Authentication successful'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        user_id = session.get('user_id')
        
        # Get data from form
        data = request.get_json()
        
        # Get selected model (default to best model)
        selected_model_name = data.get('model_name', model_info['best_model_name'])
        model = models_all[selected_model_name]
        
        # Extract features in correct order
        gender = data['gender']
        age = int(data['age'])
        occupation = data['occupation']
        sleep_duration = float(data['sleep_duration'])
        physical_activity = int(data['physical_activity'])
        stress_level = int(data['stress_level'])
        bmi_category = data['bmi_category']
        heart_rate = int(data['heart_rate'])
        daily_steps = int(data['daily_steps'])
        sleep_disorder = data['sleep_disorder']
        systolic_bp = int(data['systolic_bp'])
        diastolic_bp = int(data['diastolic_bp'])
        
        # Encode categorical variables
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        occupation_encoded = label_encoders['Occupation'].transform([occupation])[0]
        bmi_encoded = label_encoders['BMI Category'].transform([bmi_category])[0]
        disorder_encoded = label_encoders['Sleep Disorder'].transform([sleep_disorder])[0]
        
        # Create feature array in the correct order
        features = np.array([[
            gender_encoded,
            age,
            occupation_encoded,
            sleep_duration,
            physical_activity,
            stress_level,
            bmi_encoded,
            heart_rate,
            daily_steps,
            disorder_encoded,
            systolic_bp,
            diastolic_bp
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Round to 1 decimal place
        prediction = round(prediction, 1)
        
        # Generate interpretation
        if prediction >= 8:
            quality = "Excellent"
            message = "Your sleep quality is outstanding! Keep maintaining these healthy habits."
            color = "#10b981"
        elif prediction >= 7:
            quality = "Good"
            message = "Your sleep quality is good. Small improvements can make it even better."
            color = "#3b82f6"
        elif prediction >= 6:
            quality = "Fair"
            message = "Your sleep quality is fair. Consider adjusting your lifestyle habits."
            color = "#f59e0b"
        else:
            quality = "Poor"
            message = "Your sleep quality needs attention. Focus on improving sleep hygiene."
            color = "#ef4444"
        
        # Get model metrics
        model_metrics = model_info['results'][selected_model_name]
        
        # ============ NEW: Generate Detailed Breakdown ============
        
        # Sleep Duration Analysis
        sleep_status = "Optimal"
        sleep_score = 100
        sleep_recommendation = "You're getting the recommended amount of sleep. Great job!"
        
        if sleep_duration < 6:
            sleep_status = "Too Short"
            sleep_score = 40
            sleep_recommendation = f"You're sleeping only {sleep_duration} hours. Aim for 7-9 hours for optimal health."
        elif sleep_duration < 7:
            sleep_status = "Below Optimal"
            sleep_score = 70
            sleep_recommendation = f"Try to increase your sleep by {round(7 - sleep_duration, 1)} hours for better rest."
        elif sleep_duration > 9:
            sleep_status = "Too Long"
            sleep_score = 75
            sleep_recommendation = "Oversleeping can also affect sleep quality. Aim for 7-9 hours."
        
        # Lifestyle Score (Activity + Stress + Steps)
        lifestyle_score = 0
        lifestyle_recommendations = []
        
        # Physical Activity
        if physical_activity >= 30:
            lifestyle_score += 35
            activity_status = "Good"
        else:
            activity_status = "Low"
            lifestyle_recommendations.append(f"Increase physical activity to 30-60 min/day (currently {physical_activity} min)")
            lifestyle_score += (physical_activity / 30) * 35
        
        # Stress Level
        if stress_level <= 3:
            lifestyle_score += 35
            stress_status = "Low"
        elif stress_level <= 6:
            lifestyle_score += 20
            stress_status = "Moderate"
            lifestyle_recommendations.append(f"Your stress level is {stress_level}/10. Try meditation or relaxation techniques.")
        else:
            stress_status = "High"
            lifestyle_recommendations.append(f"High stress ({stress_level}/10) significantly impacts sleep. Consider stress management strategies.")
            lifestyle_score += 10
        
        # Daily Steps
        if daily_steps >= 8000:
            lifestyle_score += 30
            steps_status = "Excellent"
        elif daily_steps >= 5000:
            lifestyle_score += 20
            steps_status = "Good"
        else:
            steps_status = "Low"
            lifestyle_recommendations.append(f"Increase daily steps to 8,000-10,000 (currently {daily_steps})")
            lifestyle_score += (daily_steps / 8000) * 30
        
        # Health Metrics Score (Heart Rate + Blood Pressure + BMI)
        health_score = 0
        health_recommendations = []
        
        # Heart Rate
        if 60 <= heart_rate <= 80:
            health_score += 40
            hr_status = "Normal"
        elif 50 <= heart_rate < 60 or 80 < heart_rate <= 90:
            health_score += 25
            hr_status = "Borderline"
            health_recommendations.append(f"Heart rate is {heart_rate} bpm. Consider monitoring regularly.")
        else:
            hr_status = "Concerning"
            health_recommendations.append(f"Heart rate ({heart_rate} bpm) is outside normal range. Consult a healthcare provider.")
            health_score += 10
        
        # Blood Pressure
        if systolic_bp < 120 and diastolic_bp < 80:
            health_score += 40
            bp_status = "Normal"
        elif systolic_bp < 130 and diastolic_bp < 85:
            health_score += 25
            bp_status = "Elevated"
            health_recommendations.append("Blood pressure is slightly elevated. Monitor regularly.")
        else:
            bp_status = "High"
            health_recommendations.append("Blood pressure is high. Consult a healthcare provider.")
            health_score += 10
        
        # BMI
        if bmi_category in ["Normal", "Normal Weight"]:
            health_score += 20
            bmi_status = "Healthy"
        elif bmi_category == "Overweight":
            health_score += 12
            bmi_status = "Overweight"
            health_recommendations.append("BMI indicates overweight. Consider balanced diet and exercise.")
        else:
            bmi_status = bmi_category
            health_recommendations.append(f"BMI category: {bmi_category}. Consult a healthcare provider for personalized advice.")
            health_score += 8
        
        # Sleep Disorder Impact
        disorder_impact = "None"
        disorder_recommendation = ""
        if sleep_disorder != "None":
            disorder_impact = sleep_disorder
            disorder_recommendation = f"You have {sleep_disorder}. This significantly affects sleep quality. Follow your doctor's treatment plan."
        
        # Generate Priority Recommendations
        critical_recommendations = []
        improvement_recommendations = []
        good_habits = []
        
        # Critical (Red flags)
        if sleep_duration < 6:
            critical_recommendations.append({
                'category': 'Sleep Duration',
                'value': f'{sleep_duration} hours',
                'recommendation': 'Increase sleep by at least 1-2 hours'
            })
        
        if stress_level >= 7:
            critical_recommendations.append({
                'category': 'Stress Level',
                'value': f'{stress_level}/10',
                'recommendation': 'High stress - try meditation, yoga, or therapy'
            })
        
        if systolic_bp >= 130 or diastolic_bp >= 85:
            critical_recommendations.append({
                'category': 'Blood Pressure',
                'value': f'{systolic_bp}/{diastolic_bp}',
                'recommendation': 'Consult a doctor about blood pressure management'
            })
        
        # Needs Improvement (Yellow flags)
        if 6 <= sleep_duration < 7:
            improvement_recommendations.append({
                'category': 'Sleep Duration',
                'value': f'{sleep_duration} hours',
                'recommendation': f'Add {round(7.5 - sleep_duration, 1)} more hours for optimal sleep'
            })
        
        if physical_activity < 30:
            improvement_recommendations.append({
                'category': 'Physical Activity',
                'value': f'{physical_activity} min/day',
                'recommendation': 'Aim for 30-60 minutes of daily exercise'
            })
        
        if daily_steps < 8000:
            improvement_recommendations.append({
                'category': 'Daily Steps',
                'value': f'{daily_steps} steps',
                'recommendation': 'Target 8,000-10,000 steps per day'
            })
        
        if 4 <= stress_level < 7:
            improvement_recommendations.append({
                'category': 'Stress Level',
                'value': f'{stress_level}/10',
                'recommendation': 'Practice stress-reduction techniques regularly'
            })
        
        # Good Habits (Green - keep doing)
        if sleep_duration >= 7 and sleep_duration <= 9:
            good_habits.append({
                'category': 'Sleep Duration',
                'value': f'{sleep_duration} hours',
                'note': 'Perfect sleep duration!'
            })
        
        if physical_activity >= 30:
            good_habits.append({
                'category': 'Physical Activity',
                'value': f'{physical_activity} min/day',
                'note': 'Great activity level!'
            })
        
        if daily_steps >= 8000:
            good_habits.append({
                'category': 'Daily Steps',
                'value': f'{daily_steps} steps',
                'note': 'Excellent daily movement!'
            })
        
        if stress_level <= 3:
            good_habits.append({
                'category': 'Stress Level',
                'value': f'{stress_level}/10',
                'note': 'Well-managed stress!'
            })
        
        if 60 <= heart_rate <= 80:
            good_habits.append({
                'category': 'Heart Rate',
                'value': f'{heart_rate} bpm',
                'note': 'Healthy heart rate!'
            })
        
        if systolic_bp < 120 and diastolic_bp < 80:
            good_habits.append({
                'category': 'Blood Pressure',
                'value': f'{systolic_bp}/{diastolic_bp}',
                'note': 'Optimal blood pressure!'
            })
        
        # Breakdown data
        breakdown = {
            'sleep': {
                'score': round(sleep_score),
                'status': sleep_status,
                'recommendation': sleep_recommendation
            },
            'lifestyle': {
                'score': round(lifestyle_score),
                'activity_status': activity_status,
                'stress_status': stress_status,
                'steps_status': steps_status,
                'recommendations': lifestyle_recommendations
            },
            'health': {
                'score': round(health_score),
                'hr_status': hr_status,
                'bp_status': bp_status,
                'bmi_status': bmi_status,
                'recommendations': health_recommendations
            },
            'disorder': {
                'impact': disorder_impact,
                'recommendation': disorder_recommendation
            }
        }
        
        recommendations = {
            'critical': critical_recommendations,
            'improvement': improvement_recommendations,
            'good_habits': good_habits
        }
        
        # Save prediction to Firestore
        prediction_data = {
            'user_id': user_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'model_used': selected_model_name,
            'prediction': prediction,
            'quality': quality,
            'breakdown': breakdown,
            'recommendations': recommendations,
            'input_data': {
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'sleep_duration': sleep_duration,
                'physical_activity': physical_activity,
                'stress_level': stress_level,
                'bmi_category': bmi_category,
                'heart_rate': heart_rate,
                'daily_steps': daily_steps,
                'sleep_disorder': sleep_disorder,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp
            }
        }
        
        db.collection('predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'quality': quality,
            'message': message,
            'color': color,
            'model_used': selected_model_name,
            'model_metrics': model_metrics,
            'breakdown': breakdown,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/get_options')
def get_options():
    """Return available options for dropdown fields"""
    options = {
        'occupations': label_encoders['Occupation'].classes_.tolist(),
        'bmi_categories': label_encoders['BMI Category'].classes_.tolist(),
        'sleep_disorders': label_encoders['Sleep Disorder'].classes_.tolist(),
        'models': model_info['model_names'],
        'model_results': model_info['results']
    }
    return jsonify(options)

@app.route('/config')
def get_firebase_config():
    """Return Firebase config for client-side"""
    config = {
        'apiKey': os.getenv('FIREBASE_API_KEY'),
        'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
        'projectId': os.getenv('FIREBASE_PROJECT_ID'),
        'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
        'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
        'appId': os.getenv('FIREBASE_APP_ID')
    }
    return jsonify(config)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
