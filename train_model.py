import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json

print("=" * 80)
print("DREAMMETRICS - MODEL TRAINING")
print("=" * 80)

# Load dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print(f"\n[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create a copy for processing
data = df.copy()

# Drop Person ID (not useful for prediction)
data = data.drop('Person ID', axis=1)

# Handle Blood Pressure - split into systolic and diastolic
print("\n[OK] Processing Blood Pressure column...")
if 'Blood Pressure' in data.columns:
    bp_split = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
    data['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
    data = data.drop('Blood Pressure', axis=1)

# Handle missing values
print("[OK] Handling missing values...")
# Fill NaN in Sleep Disorder with 'None'
if 'Sleep Disorder' in data.columns:
    data['Sleep Disorder'] = data['Sleep Disorder'].fillna('None')

# Fill any remaining numeric NaN with median
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical variables
print("[OK] Encoding categorical variables...")
label_encoders = {}
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = data.drop('Quality of Sleep', axis=1)
y = data['Quality of Sleep']

print(f"\n[OK] Features: {list(X.columns)}")
print(f"[OK] Target: Quality of Sleep")
print(f"[OK] Target range: {y.min()} to {y.max()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n[OK] Train set: {X_train.shape[0]} samples")
print(f"[OK] Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
trained_models = {}
best_model_name = ""
best_score = -float('inf')

for name, model in models.items():
    print(f"\n-> Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'CV_R2': round(cv_mean, 4)
    }
    
    # Save trained model
    trained_models[name] = model
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  CV R2: {cv_mean:.4f}")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model_name = name

# Save all trained models
print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name} (R2 = {best_score:.4f})")
print("=" * 80)

# Save all models
joblib.dump(trained_models, 'models_all.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save model info
model_info = {
    'best_model_name': best_model_name,
    'features': list(X.columns),
    'categorical_columns': categorical_columns,
    'results': results,
    'model_names': list(models.keys())
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print("\n[OK] All models saved: models_all.pkl")
print("[OK] Scaler saved: scaler.pkl")
print("[OK] Label encoders saved: label_encoders.pkl")
print("[OK] Model info saved: model_info.json")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
