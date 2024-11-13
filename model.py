# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Set style for plots
sns.set_style("whitegrid")

# Load data
train = pd.read_csv('/path/to/train.csv')
test = pd.read_csv('/path/to/test.csv')
submission = pd.read_csv('/path/to/sample_submission.csv')

# Check for missing data
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print(f"Missing values in training set:\n{missing_train}")

# Fill missing values for numerical and categorical data
for col in train.select_dtypes(include=['number']).columns:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

for col in train.select_dtypes(include=['object']).columns:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# Log transform the target variable to reduce skewness
train['SalePrice'] = np.log1p(train['SalePrice'])

# Visualize target variable
sns.histplot(train['SalePrice'], kde=True)
plt.title("Log-transformed SalePrice Distribution")
plt.show()

# Concatenate train and test data for easier feature engineering
y = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)
combined_data = pd.concat([train, test], keys=['train', 'test'])

# Encoding categorical variables
combined_data = pd.get_dummies(combined_data, drop_first=True)

# Split combined data back into train and test
train = combined_data.xs('train')
test = combined_data.xs('test')

# Scale features
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Define models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Perform cross-validation for each model
for name, model in models.items():
    scores = cross_val_score(model, train, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"{name} - RMSE: {rmse_scores.mean():.4f}")

# Select best model based on cross-validation score
best_model = RandomForestRegressor(n_estimators=200, random_state=42)
best_model.fit(train, y)

# Make predictions and prepare for submission
predictions = np.expm1(best_model.predict(test))  # Inverse of log transform

# Prepare submission file
submission['SalePrice'] = predictions
submission.to_csv('house_price_predictions.csv', index=False)
print("Submission file created successfully.")
