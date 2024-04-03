import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Load the dataset
df = pd.read_csv("BIKE DETAILS.csv")

# Select relevant features
final_dataset = df[['selling_price', 'year', 'seller_type', 'owner', 'km_driven']]

# Create new feature: age of the bike
final_dataset['Current_Year'] = 2020
final_dataset['no_year'] = final_dataset['Current_Year'] - final_dataset['year']

# Drop unnecessary columns
final_dataset.drop(['Current_Year', 'year'], axis=1, inplace=True)

# One-hot encode categorical variables
final_dataset = pd.get_dummies(final_dataset, drop_first=True)

# Split data into features (X) and target (y)
X = final_dataset.drop('selling_price', axis=1)
y = final_dataset['selling_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define hyperparameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(5, 30, num=6)],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Instantiate RandomForestRegressor
rf = RandomForestRegressor()

# Perform RandomizedSearchCV for hyperparameter tuning
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                               n_iter=10, cv=5, scoring='neg_mean_squared_error', 
                               verbose=2, random_state=42, n_jobs=1)

# Fit RandomizedSearchCV to training data
rf_random.fit(X_train, y_train)

# Print best hyperparameters and best score
print("Best Hyperparameters:", rf_random.best_params_)
print("Best Score:", rf_random.best_score_)

# Make predictions on test data
predictions = rf_random.predict(X_test)

# Evaluate model performance
mae = np.mean(abs(predictions - y_test))
mse = np.mean((predictions - y_test)**2)
rmse = np.sqrt(mse)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# Save the trained model to a file
with open('random_forest_regression_model.pkl', 'wb') as file:
    pickle.dump(rf_random, file)
