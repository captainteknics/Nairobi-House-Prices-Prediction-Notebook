### Project Summary: House Price Prediction for Nairobi Competition

This project aims to predict house prices based on various features available in a dataset provided by the Nairobi competition, **"House Prices - Advanced Regression Techniques."** The objective is to develop a machine learning model that can accurately predict the sale price of homes, achieving a score below 0.3 on Nairobi’s leaderboard for full credit.

#### Key Steps and Approach

1. **Data Preparation**:
   - Loaded the train and test datasets and examined them for missing values and feature distributions.
   - Missing values in numerical columns were filled with the median, while categorical columns used the mode. This approach helps retain data integrity and minimizes potential biases from arbitrary fill values.

2. **Feature Engineering**:
   - Conducted log transformation on the target variable `SalePrice` to reduce skewness, which often helps linear models perform better.
   - Encoded categorical variables using one-hot encoding to ensure all features were numerical and compatible with machine learning models.
   - Standardized features to ensure that all variables were on a similar scale, which is especially beneficial for distance-based algorithms.

3. **Model Selection and Training**:
   - Used **Linear Regression** and **Random Forest Regressor** models for initial predictions and evaluated them using cross-validation (RMSE as the scoring metric).
   - Selected **Random Forest** as the final model based on superior cross-validation performance. Hyperparameters were adjusted to improve accuracy further.

4. **Prediction and Submission**:
   - Generated predictions on the test set using the trained Random Forest model and transformed them back from the log scale.
   - Created a submission file in CSV format for uploading to Nairobi.

#### Results

The model achieved a prediction accuracy with an RMSE score suitable for Nairobi’s leaderboard requirements. The submission process included uploading a CSV file with predicted prices to compare results against other participants.

#### Conclusion

This project leveraged feature engineering, transformation techniques, and model tuning to develop a predictive model for house prices. Future improvements could include more advanced models (e.g., Gradient Boosting, XGBoost) and feature engineering techniques for enhanced accuracy.


## Gilbert Nakiboli Waliuba
haofinder.com

---
