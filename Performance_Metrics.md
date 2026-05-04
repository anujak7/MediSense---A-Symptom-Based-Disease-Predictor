# Model Performance Metrics

This document contains the evaluation results of the Disease Prediction Model.

## 1. Accuracy
**Accuracy Score:** `99.25%`
*This means the model correctly predicted the disease in 99.25 out of 100 cases.*

## 2. Confusion Matrix
The confusion matrix shows the performance of the classification model on the test data.

```text
[[63  1  0  0  0  0  0]
 [ 0 62  0  0  0  0  0]
 [ 0  0 58  0  0  0  0]
 [ 0  1  0 51  0  0  0]
 [ 0  0  0  0 56  0  0]
 [ 0  0  0  0  0 51  0]
 [ 0  0  1  0  0  0 56]]
```

### Interpretation:
- **Diagonal Elements:** These represent the number of correct predictions for each disease.
- **Off-diagonal Elements:** These represent misclassifications (e.g., one case of Disease A was predicted as Disease B).
- With very few off-diagonal values, the model is performing exceptionally well.

## 3. Training Details
- **Algorithm:** Random Forest Classifier
- **Total Samples:** 2,000
- **Training Set:** 1,600 (80%)
- **Testing Set:** 400 (20%)
- **Features Used:** Top 15 Symptoms (Selected via Importance Score)
