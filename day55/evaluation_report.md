
# Advanced Customer Churn Prediction - Evaluation Report

## 1. Executive Summary

The best model based on test-set F1-score was **Logistic Regression**.

- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-score: 1.0000

These results describe performance on the held-out test set.
They should not be treated as guaranteed business outcomes.

## 2. Data Journey

The churn dataset was cleaned and prepared.

Feature engineering included:
- Total charges = Monthly charges multiplied by tenure.
- Support call rate = Support calls divided by tenure plus one.
- Tenure groups: New, Short Term, Medium Term and Long Term.

Numeric features were imputed and scaled.
Categorical features were imputed and one-hot encoded.

SMOTE was applied inside the training pipeline.
The test data was not oversampled.

## 3. Model Leaderboard

                    Model  Accuracy  Precision  Recall  F1 Score  Training Time
      Logistic Regression       1.0        1.0     1.0       1.0       3.485726
            Decision Tree       1.0        1.0     1.0       1.0       0.037142
            Random Forest       1.0        1.0     1.0       1.0       0.155021
                      KNN       1.0        1.0     1.0       1.0       0.023508
                      SVM       1.0        1.0     1.0       1.0       0.024489
        Gradient Boosting       1.0        1.0     1.0       1.0       0.116781
                  XGBoost       1.0        1.0     1.0       1.0       1.852996
    Random Forest (Tuned)       1.0        1.0     1.0       1.0       6.143912
Gradient Boosting (Tuned)       1.0        1.0     1.0       1.0       0.374488
          XGBoost (Tuned)       1.0        1.0     1.0       1.0       0.851505

## 4. Winning Model - Confusion Matrix

- True Negatives: 4
- False Positives: 0
- False Negatives: 0
- True Positives: 4

The model correctly identified 4 churn customers.
It incorrectly flagged 0 customers who stayed as churn risks.
It missed 0 actual churn customers.

## 5. Feature Importance

The selected model does not expose built-in feature importance. Use permutation importance for further analysis.

## 6. Limitations

- This project uses a small educational dataset.
- Test metrics may change substantially with different data splits.
- SMOTE after one-hot encoding can create fractional encoded
  categorical values; SMOTENC is a better choice for mixed
  numerical and categorical data in production.
- Feature importance indicates model reliance, not causation.
- A larger, representative dataset and further validation are
  needed before using this model for business decisions.

## 7. Conclusion

The model was selected using test-set F1-score.
The saved pipeline includes preprocessing, SMOTE and the
trained classifier, allowing consistent predictions on new data.
