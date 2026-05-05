from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

# Global variables to store model and data
MODEL_PATH = 'model.joblib'
ENCODER_PATH = 'encoder.joblib'
FEATURES_PATH = 'features.joblib'

def train_and_save_model():
    print("Training model...")
    if not os.path.exists('Kaggle_Disease_Dataset.csv'):
        # Just in case, though the previous script should have generated it
        # I'll import the generator logic if needed, but for now assuming it exists
        pass
    
    df = pd.read_csv('Kaggle_Disease_Dataset.csv')
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Feature selection (top 15)
    temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    temp_rf.fit(X, y_encoded)
    
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': temp_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top_15_features = feature_importances['Feature'].head(15).tolist()
    X_selected = X[top_15_features]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_selected, y_encoded)
    
    # Save everything
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(top_15_features, FEATURES_PATH)
    print("Model trained and saved!")
    return rf_model, le, top_15_features

# Initial load or train
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURES_PATH):
    rf_model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    top_15_features = joblib.load(FEATURES_PATH)
else:
    rf_model, le, top_15_features = train_and_save_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': top_15_features})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    selected_symptoms = data.get('symptoms', [])
    
    if not selected_symptoms:
        return jsonify({'error': 'No symptoms selected'}), 400
    
    # Prepare input array
    user_input = np.zeros(len(top_15_features))
    for sym in selected_symptoms:
        if sym in top_15_features:
            idx = top_15_features.index(sym)
            user_input[idx] = 1
            
    # Predict
    pred_encoded = rf_model.predict(user_input.reshape(1, -1))
    pred_disease = le.inverse_transform(pred_encoded)[0]
    
    # Get probability (optional but cool for UI)
    probs = rf_model.predict_proba(user_input.reshape(1, -1))[0]
    confidence = np.max(probs) * 100
    
    return jsonify({
        'disease': pred_disease,
        'confidence': f"{confidence:.2f}%",
        'description': f"Our model suggests a high probability of {pred_disease}. Please consult a professional for a formal diagnosis."
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
