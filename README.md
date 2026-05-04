#  MediSense: A Symptom-Based Disease Predictor


[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.25%25-brightgreen.svg)](#-model-evaluation)

**MediSense** is an intelligent healthcare assistant powered by Machine Learning that predicts potential diseases based on user symptoms. Using a robust **Random Forest Classifier**, it analyzes patterns in clinical data to provide accurate preliminary diagnoses.

---

##  Key Features

- **High Accuracy:** Achieves **99.25% accuracy** on clinical test data.
- **Intelligent Feature Selection:** Automatically identifies the top 15 most significant symptoms using Importance Scores.
- **Interactive Interface:** Easy-to-use Command Line Interface (CLI) for real-time symptom input.
- **Detailed Analytics:** Generates a comprehensive performance report and Confusion Matrix after every training session.
- **Knowledge Base:** Includes a secondary descriptive dataset with treatment info and specialist recommendations.

---

## Tech Stack

- **Language:** Python 3.x
- **Data Handling:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-Learn` (Random Forest Classifier)
- **Deployment:** Git & GitHub

---

##  Project Structure

```text
├── random_forest_disease_predictor.py  # Main ML Pipeline & CLI
├── Kaggle_Disease_Dataset.csv         # Training Dataset (0/1 Symptoms)
├── dataset.csv                        # Informational Dataset (Treatments/Doctors)
├── Project_Report.md                  # Comprehensive Academic Report
├── Performance_Metrics.md              # Evaluation Results
├── requirements.txt                   # Dependency List
└── banner.png                         # Project Visuals
```

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python installed. You can install the required libraries using:
```bash
pip install -r requirements.txt
```

### 2. Run the Predictor
Launch the interactive prediction tool:
```bash
python random_forest_disease_predictor.py
```

### 3. Usage
Enter your symptoms separated by commas when prompted:
> `-> Enter symptoms: fever, cough, chills`

---

##  Model Evaluation

The model was evaluated using a **Confusion Matrix** on a 20% test split:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 99.25% |
| **Algorithm** | Random Forest (100 Estimators) |
| **Test Samples** | 400 Rows |

### Confusion Matrix
```text
[[63  1  0  0  0  0  0]
 [ 0 62  0  0  0  0  0]
 [ 0  0 58  0  0  0  0]
 [ 0  1  0 51  0  0  0]
 [ 0  0  0  0 56  0  0]
 [ 0  0  0  0  0 51  0]
 [ 0  0  1  0  0  0 56]]
```

---

##  Dataset Source
The primary training data is based on the [Disease Prediction using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) dataset from Kaggle.

---

##  Disclaimer
*MediSense is intended for educational and preliminary screening purposes only. It is not a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for medical concerns.*

---

##  Author
**Anuj Khan**  
*Computer Science Student*

---
 **If you find this project helpful, don't forget to give it a star!**
