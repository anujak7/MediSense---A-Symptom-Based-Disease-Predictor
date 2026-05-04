# Disease Prediction System (ML)

This project is a Machine Learning based application that predicts diseases based on symptoms using the Random Forest algorithm.

## 🚀 Features
- **Accurate Prediction:** Uses Random Forest Classifier for high accuracy.
- **Top Feature Selection:** Automatically selects the most important 15 symptoms.
- **Interactive Console:** User-friendly command-line interface for real-time prediction.
- **Data Preprocessing:** Includes data cleaning, label encoding, and missing value handling.

## 🛠️ Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn

## 📂 Project Structure
- `random_forest_disease_predictor.py`: Main execution script.
- `Project_Report.md`: Formal academic project report.
- `Kaggle_Disease_Dataset.csv`: Dataset used for training.
- `requirements.txt`: Python dependencies.

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the predictor:
   ```bash
   python random_forest_disease_predictor.py
   ```
3. Enter your symptoms when prompted (e.g., `fever, cough, headache`).

## 📊 Model Performance
- **Accuracy:** ~95%
- **Algorithm:** Random Forest (100 Trees)
- **Split:** 80% Train, 20% Test
