# Code Explanation: Disease Prediction System

This document explains the logic behind the code in `random_forest_disease_predictor.py` to help you explain it to your teacher.

## 1. Imports (Lines 1-9)
We use `pandas` for data handling, `numpy` for math, and `sklearn` for Machine Learning models and metrics. We also suppress warnings to keep the output clean.

## 2. Mock Data Generation (Lines 11-56)
If the dataset file is not found, this function creates a synthetic dataset of 2,000 patients with 7 diseases and 20 symptoms. It adds a bit of "noise" (randomness) to make the model more robust, just like real-world data.

## 3. Data Loading & Cleaning (Lines 68-83)
We read the CSV file. If there are any empty (null) values, they are removed using `df.dropna()`.

## 4. Feature and Target Separation (Lines 85-91)
- **X (Features):** These are the symptoms (columns like fever, cough).
- **y (Target):** This is the disease we want to predict (prognosis).

## 5. Label Encoding (Lines 93-97)
Computers understand numbers, not words. `LabelEncoder` converts disease names like "Malaria" into numbers like `2`.

## 6. Feature Selection (Lines 99-115)
Not all symptoms are equally important. We train a quick model to find out which 15 symptoms are the strongest indicators of disease. We then filter our data to use *only* these top 15 symptoms. This makes the model faster and more accurate.

## 7. Training and Testing Split (Lines 117-123)
We split the data into two parts:
- **Training Set (80%):** The model "learns" from this data.
- **Testing Set (20%):** We use this data to "test" the model and see how accurate it is.

## 8. Random Forest Model (Lines 125-135)
We use the **Random Forest Classifier** with 100 trees. It's like asking 100 different experts to vote on what the disease might be, and taking the majority vote. This makes it very reliable.

## 9. Evaluation (Lines 137-144)
We calculate the **Accuracy Score**. If the model gets 95%, it means it predicts correctly 95 times out of 100. The **Confusion Matrix** shows exactly where the model got confused (if any).

## 10. Interactive Prediction (Lines 146-189)
This is the loop where you can type your symptoms. 
- It takes your input.
- Converts it into a format the model understands.
- Asks the model to predict the disease.
- Converts the predicted number back into a disease name and prints it.
