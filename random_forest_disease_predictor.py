import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def generate_mock_dataset():
    """Generates a mock Kaggle-style dataset if one doesn't exist."""
    print("Creating a mock Kaggle-style dataset for demonstration...")
    np.random.seed(42)
    diseases = ['Flu', 'Common Cold', 'Malaria', 'Typhoid', 'Covid-19', 'Dengue', 'Pneumonia']
    
    # List of ~20 symptoms
    symptoms = [
        'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', 
        'chest_pain', 'muscle_aches', 'loss_of_taste', 'chills', 
        'runny_nose', 'sore_throat', 'diarrhea', 'sweating', 'joint_pain',
        'shortness_of_breath', 'dizziness', 'stomach_pain', 'rash', 'weakness'
    ]
    
    data = []
    for _ in range(2000): # 2000 samples
        disease = np.random.choice(diseases)
        row = {sym: 0 for sym in symptoms}
        row['prognosis'] = disease
        
        # Add realistic symptoms for each disease
        if disease == 'Flu':
            for s in ['fever', 'cough', 'fatigue', 'muscle_aches', 'weakness']: row[s] = 1
        elif disease == 'Common Cold':
            for s in ['cough', 'runny_nose', 'sore_throat', 'headache']: row[s] = 1
        elif disease == 'Malaria':
            for s in ['fever', 'chills', 'sweating', 'headache', 'nausea']: row[s] = 1
        elif disease == 'Typhoid':
            for s in ['fever', 'headache', 'nausea', 'diarrhea', 'stomach_pain']: row[s] = 1
        elif disease == 'Covid-19':
            for s in ['fever', 'cough', 'loss_of_taste', 'shortness_of_breath', 'fatigue']: row[s] = 1
        elif disease == 'Dengue':
            for s in ['fever', 'joint_pain', 'headache', 'vomiting', 'rash']: row[s] = 1
        elif disease == 'Pneumonia':
            for s in ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'chills']: row[s] = 1
            
        # Add 5% noise (random 0s and 1s) to make it realistic
        for sym in symptoms:
            if np.random.rand() < 0.05:
                row[sym] = 1 if row[sym] == 0 else 0
                
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv('Kaggle_Disease_Dataset.csv', index=False)
    print("[SUCCESS] Mock 'Kaggle_Disease_Dataset.csv' generated successfully!\n")

def main():
    print("="*50)
    print("DISEASE PREDICTION PIPELINE (RANDOM FOREST)")
    print("="*50)

    # Prepare data if not exists
    if not os.path.exists('Kaggle_Disease_Dataset.csv'):
        generate_mock_dataset()

    # --- Step 1: Data Load ---
    print("\n[Step 1] Loading Data...")
    df = pd.read_csv('Kaggle_Disease_Dataset.csv')
    print(f"Dataset Shape: {df.shape}")
    print(f"Total columns: {len(df.columns)} (Symptoms + Prognosis)")

    # --- Step 2: Data Cleaning ---
    print("\n[Step 2] Cleaning Data...")
    null_count = df.isnull().sum().sum()
    print(f"Found {null_count} missing values.")
    if null_count > 0:
        df.dropna(inplace=True)
        print(f"Missing values dropped. New Shape: {df.shape}")
    
    # Note: We usually drop duplicates, but in this medical dataset, 
    # many patients can have the exact same symptoms, so duplicates are valid!
    print("Data cleaning complete.")

    # --- Step 3: Feature & Target Split ---
    print("\n[Step 3] Splitting Features (X) and Target (y)...")
    X = df.drop('prognosis', axis=1) # All symptom columns (0/1)
    y = df['prognosis']              # The disease name
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # --- Step 4: Label Encoding (IMPORTANT) ---
    print("\n[Step 4] Label Encoding Target Variable...")
    print(f"Sample targets before encoding: {y.unique()[:3]}")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Sample targets after encoding: {np.unique(y_encoded)[:3]}")

    # --- Step 5: Feature Selection ---
    print("\n[Step 5] Performing Feature Selection (Top 15 Features)...")
    # Train a quick model just to check importance
    temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    temp_rf.fit(X, y_encoded)
    
    # Extract importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': temp_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top_15_features = feature_importances['Feature'].head(15).tolist()
    print(f"Top 15 Important Symptoms Selected:\n {top_15_features}")
    
    # Filter X to only keep top 15 features to reduce complexity
    X_selected = X[top_15_features]

    # --- Step 6: Train-Test Split ---
    print("\n[Step 6] Train-Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.20, random_state=42
    )
    print(f"Training Data: {X_train.shape[0]} rows")
    print(f"Testing Data: {X_test.shape[0]} rows")

    # --- Step 7: Model Training (CORE) ---
    print("\n[Step 7] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # 100 decision trees
        max_depth=None,    # Trees expand until pure
        random_state=42,
        n_jobs=-1          # Use all CPU cores for speed
    )
    rf_model.fit(X_train, y_train)
    print("[SUCCESS] Model Training Complete!")

    # --- Step 8: Model Evaluation ---
    print("\n[Step 8] Evaluating Model Performance...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    report_text = f"""
==================================================
        MODEL PERFORMANCE REPORT
==================================================
Accuracy Score: {accuracy * 100:.2f}%

Confusion Matrix:
{conf_matrix}

Selected Symptoms (Top 15):
{', '.join(top_15_features)}
==================================================
"""
    print(report_text)
    
    # Save to file for the user
    with open('last_run_report.txt', 'w') as f:
        f.write(report_text)
    print("[INFO] Results have been saved to 'last_run_report.txt'")

    # --- Step 9: Interactive Prediction Flow ---
    print("\n[Step 9] Interactive User Prediction Flow...")
    print("Available symptoms to choose from:")
    print(", ".join(top_15_features))
    print("\nEnter your symptoms separated by commas (e.g., fever, cough, headache)")
    print("Or type 'exit' to quit.")
    
    while True:
        user_input_str = input("\n-> Enter symptoms: ").strip().lower()
        if user_input_str == 'exit':
            print("Exiting...")
            break
            
        if not user_input_str:
            continue
            
        symptoms_user_has = [s.strip() for s in user_input_str.split(',')]
        
        user_input_array = np.zeros(len(top_15_features))
        valid_symptoms_entered = []
        
        for symptom in symptoms_user_has:
            if symptom in top_15_features:
                idx = top_15_features.index(symptom)
                user_input_array[idx] = 1
                valid_symptoms_entered.append(symptom)
            else:
                print(f"  [Warning] Symptom '{symptom}' not recognized.")
                
        if not valid_symptoms_entered:
            print("  No valid symptoms recognized. Please try again.")
            continue
            
        # Reshape input for prediction: [1, number_of_features]
        user_input_reshaped = user_input_array.reshape(1, -1)
        
        # Predict numeric class
        pred_encoded = rf_model.predict(user_input_reshaped)
        
        # Reverse label encoding to get the string disease name
        pred_disease_name = le.inverse_transform(pred_encoded)[0]
        
        print(f"\n[RESULT] Based on '{', '.join(valid_symptoms_entered)}', the Predicted Disease is: *** {pred_disease_name.upper()} ***")
        
    print("="*50)

if __name__ == "__main__":
    main()
