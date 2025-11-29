from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print("=" * 60)
print("DREAMMETRICS - Flask App Started")
print(f"Model: {model_info['model_name']}")
print(f"R2 Score: {model_info['best_metrics']['R2']}")
print("=" * 60)

@app.route('/')
def home():
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.get_json()
        
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
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'quality': quality,
            'message': message,
            'color': color
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
        'sleep_disorders': label_encoders['Sleep Disorder'].classes_.tolist()
    }
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
