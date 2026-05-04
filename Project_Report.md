# Project Report: Disease Prediction System using Machine Learning

![Project Banner](file:///C:/Users/SVSU/.gemini/antigravity/brain/86e9e6c4-1a41-478e-973b-541559774fa6/disease_prediction_banner_1777887713271.png)

## 1. Introduction
The **Disease Prediction System** is a machine learning-based application designed to predict potential diseases based on user-provided symptoms. By leveraging the power of predictive analytics, this system can assist in early screening and provide a preliminary diagnosis, which can be crucial for timely medical intervention.

## 2. Objective
*   To develop an accurate predictive model for multiple diseases (e.g., Flu, Malaria, Covid-19).
*   To process and clean medical data for machine learning.
*   To identify the most significant symptoms (Feature Selection).
*   To provide an interactive interface for users to check their symptoms.

## 3. Technology Stack
*   **Language:** Python 3.x
*   **Libraries:** 
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical computations.
    *   `scikit-learn`: For implementing the Random Forest algorithm and preprocessing.

## 4. Methodology
The project follows a standard Data Science lifecycle:

### A. Data Collection
We used a medical dataset (Kaggle-style) containing symptoms as binary features (0/1) and the corresponding prognosis (disease) as the target variable.

### B. Data Preprocessing
*   **Cleaning:** Removing null values and handling duplicates.
*   **Label Encoding:** Converting categorical disease names (e.g., "Covid-19") into numerical values (e.g., 4) using `LabelEncoder`.
*   **Feature Selection:** We calculated feature importance using a preliminary Random Forest model and selected the **top 15 symptoms** to reduce noise and improve model efficiency.

### C. Algorithm: Random Forest
We chose the **Random Forest Classifier** because:
*   It is an ensemble method (multiple decision trees) which reduces overfitting.
*   It handles high-dimensional data (many symptoms) very well.
*   It provides high accuracy for classification tasks.

## 5. Implementation Details
The core logic is implemented in `random_forest_disease_predictor.py`. Key stages include:
1.  **Data Loading:** Reading the CSV file.
2.  **Splitting:** Dividing the data into **Training (80%)** and **Testing (20%)** sets.
3.  **Training:** Fitting the Random Forest model on the training data.
4.  **Prediction:** Taking user input and mapping it to the model's feature vector for a real-time prediction.

## 6. Performance & Results
*   **Accuracy:** The model achieves an impressive **99.25% accuracy** on the test dataset.
*   **Evaluation:** We used a **Confusion Matrix** to visualize performance:

```text
[[63  1  0  0  0  0  0]
 [ 0 62  0  0  0  0  0]
 [ 0  0 58  0  0  0  0]
 [ 0  1  0 51  0  0  0]
 [ 0  0  0  0 56  0  0]
 [ 0  0  0  0  0 51  0]
 [ 0  0  1  0  0  0 56]]
```
*   **Result:** The high diagonal values indicate that the model is extremely reliable in distinguishing between different diseases based on the selected symptoms.

## 7. Conclusion
The Disease Prediction System successfully demonstrates the application of Machine Learning in the healthcare domain. While it is not a replacement for professional medical advice, it serves as a powerful tool for preliminary diagnosis and awareness.

---
**Developed by:** [Your Name]
**Project Category:** Machine Learning / Healthcare AI
