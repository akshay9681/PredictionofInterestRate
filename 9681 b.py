# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:38:05 2025

@author: aksha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Load the datasets
inflation = pd.read_csv(r"C:\Users\aksha\Downloads\INFLATION.csv")
unemployment = pd.read_csv(r"C:\Users\aksha\Downloads\UNEMPLOYMENT RATE.csv")
iip = pd.read_csv(r"C:\Users\aksha\Downloads\IIP.csv")
business_confidence = pd.read_csv(r"C:\Users\aksha\Downloads\BUISNESS CONFIDENCE.csv")

# Display the first few rows of each dataset
print("Inflation Data:")
print(inflation.head(), "\n")

print("Unemployment Data:")
print(unemployment.head(), "\n")

print("IIP Data:")
print(iip.head(), "\n")

print("Business Confidence Data:")
print(business_confidence.head(), "\n")

# Display basic info about each dataset
print("Inflation Data Info:")
print(inflation.info(), "\n")

print("Unemployment Data Info:")
print(unemployment.info(), "\n")

print("IIP Data Info:")
print(iip.info(), "\n")

print("Business Confidence Data Info:")
print(business_confidence.info(), "\n")
# Merge datasets on common time index (Assuming 'Date' column)
# Merge datasets on 'DATE'
import pandas as pd

# Load datasets
inflation = pd.read_csv(r"C:\Users\aksha\Downloads\INFLATION.csv")
unemployment = pd.read_csv(r"C:\Users\aksha\Downloads\UNEMPLOYMENT RATE.csv")
iip = pd.read_csv(r"C:\Users\aksha\Downloads\IIP.csv")
business_confidence = pd.read_csv(r"C:\Users\aksha\Downloads\BUISNESS CONFIDENCE.csv")
interest_rate = pd.read_excel(r"C:\Users\aksha\OneDrive\Documents\INTEREST RATE.xlsx")

import pandas as pd

# Standardize column names (strip spaces and make uppercase)
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    df.columns = df.columns.str.strip().str.upper()  # Convert to uppercase

# Ensure 'DATE' column exists in all datasets
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    if 'DATE' not in df.columns and 'Date' in df.columns:
        df.rename(columns={'Date': 'DATE'}, inplace=True)

# Convert 'DATE' to datetime format for consistency
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    df['DATE'] = pd.to_datetime(df['DATE'])

# Merge datasets on 'DATE' with unique suffixes to avoid duplicate column names
data = interest_rate.merge(inflation, on="DATE", how="inner", suffixes=("_INTEREST", "_INFLATION"))\
    .merge(unemployment, on="DATE", how="inner", suffixes=("", "_UNEMPLOYMENT"))\
    .merge(iip, on="DATE", how="inner", suffixes=("", "_IIP"))\
    .merge(business_confidence, on="DATE", how="inner", suffixes=("", "_BUSINESS_CONFIDENCE"))

# Display merged dataset info
print("Merged Data:")
print(data.head())
print(data.info())
# Convert DATE column to datetime format
data["DATE"] = pd.to_datetime(data["DATE"])

# Set DATE as the index
data.set_index("DATE", inplace=True)
# Check for missing values
data.fillna(method='ffill', inplace=True)  # Forward-fill missing values
# Define features and target variable
# Define features and target variable
# Define features and target variable
X = data.drop(columns=["VALUE"])
y = data["VALUE"]
# Train-test split
from sklearn.model_selection import train_test_split  # Ensure import

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)  # No need for scaling in Random Forest
from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)  # No need for scaling in Random Forest
print(model)  # This prints the model details
print(model.feature_importances_)  # Shows which features are most important
y_pred = model.predict(X_test)  # Predict on test data
print(y_pred[:5])  # Print first 5 predictions
score = model.score(X_test, y_test)  # R² (coefficient of determination)
print(f"Model R² Score: {score:.4f}")
# Evaluate the model with R² score on test data
score = model.score(X_test, y_test)  # R² score
print(f"Model R² Score: {score:.4f}")
# Predict on test data
y_pred = model.predict(X_test)  
print("First 5 predictions:", y_pred[:5])  # Display first 5 predictions
import matplotlib.pyplot as plt

# Sample predictions and actual values (replace with your actual data)
predictions = [5.823, 7.2, 3.556, 5.997, 5.845]
actual_values = [5.8, 7.1, 3.5, 6.0, 5.85]  # Replace with the actual values from your dataset

# Create an index for the data points
x = range(1, len(predictions) + 1)

# Plotting the predictions and actual values
plt.figure(figsize=(10, 6))
plt.plot(x, predictions, label='Predictions', marker='o', linestyle='-', color='b')
plt.plot(x, actual_values, label='Actual Values', marker='x', linestyle='--', color='r')

# Adding labels and title
plt.xlabel('Data Point')
plt.ylabel('Target Variable')
plt.title('Model Predictions vs Actual Values')
plt.legend()

# Display the chart
plt.grid(True)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
import numpy as np
import matplotlib.pyplot as plt

# Sample predictions and actual values (replace with your actual data)
predictions = [5.823, 7.2, 3.556, 5.997, 5.845]
actual_values = [5.8, 7.1, 3.5, 6.0, 5.85]  # Replace with the actual values from your dataset

# Example years (replace with actual years from your data)
years = [2023, 2024, 2025, 2026, 2027]

# Plotting the predictions and actual values
plt.figure(figsize=(10, 6))
plt.plot(years, predictions, label='Predictions', marker='o', linestyle='-', color='b')
plt.plot(years, actual_values, label='Actual Values', marker='x', linestyle='--', color='r')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Target Variable')
plt.title('Model Predictions vs Actual Values')
plt.legend()

# Display the chart
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Example historical predictions and actual values (replace with your actual data)
predictions = [5.823, 7.2, 3.556, 5.997, 5.845]
actual_values = [5.8, 7.1, 3.5, 6.0, 5.85]  # Replace with actual historical values

# Example years (replace with actual years from your data)
years = [2023, 2024, 2025, 2026, 2027]
# Predictions
y_pred = model.predict(X_test_scaled)
# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend()
plt.title("Actual vs Predicted Interest Rate")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the datasets (assuming you already did this)
# inflation = pd.read_csv(r"C:\Users\aksha\Downloads\INFLATION.csv")
# unemployment = pd.read_csv(r"C:\Users\aksha\Downloads\UNEMPLOYMENT RATE.csv")
# iip = pd.read_csv(r"C:\Users\aksha\Downloads\IIP.csv")
# business_confidence = pd.read_csv(r"C:\Users\aksha\Downloads\BUISNESS CONFIDENCE.csv")
# interest_rate = pd.read_excel(r"C:\Users\aksha\OneDrive\Documents\INTEREST RATE.xlsx")

# Merged dataset from the previous code (ensure you've already done the merging part)
# Example:
# data = ...

# Fill missing values if any
data.fillna(method='ffill', inplace=True)

# Normalize the features (Use MinMaxScaler)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["VALUE"]))  # Features (without target)

# Define features (X) and target variable (y)
X = scaled_data
y = data["VALUE"].values  # Target variable

# Convert data into 3D shape [samples, timesteps, features] for LSTM
def create_sequences(X, y, sequence_length=3):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])  # Sequence of previous timesteps
        y_seq.append(y[i])  # Current target variable
    return np.array(X_seq), np.array(y_seq)

sequence_length = 3  # Use previous 3 time steps to predict the next value
X_seq, y_seq = create_sequences(X, y, sequence_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend()
plt.title("Actual vs Predicted Interest Rate")
plt.show()

# Load and preprocess data (assuming it's done already)
data.fillna(method='ffill', inplace=True)

# Normalize the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["VALUE"]))  # Features (without target)

# Define features (X) and target variable (y)
X = scaled_data
y = data["VALUE"].values  # Target variable

# Create sequences for LSTM
def create_sequences(X, y, sequence_length=3):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])  # Sequence of previous timesteps
        y_seq.append(y[i])  # Current target variable
    return np.array(X_seq), np.array(y_seq)

sequence_length = 3  # Use previous 3 time steps to predict the next value
X_seq, y_seq = create_sequences(X, y, sequence_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend()
plt.title("Actual vs Predicted Interest Rate")
plt.show()



# Plot histogram for actual vs predicted values
plt.figure(figsize=(10, 5))

# Plot histogram for actual values
plt.hist(y_test, bins=30, alpha=0.5, label='Actual Values', color='blue')

# Plot histogram for predicted values
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Values', color='red')

# Adding labels and title
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Interest Rates')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb  # Import XGBoost

# Load datasets
inflation = pd.read_csv(r"C:\Users\aksha\Downloads\INFLATION.csv")
unemployment = pd.read_csv(r"C:\Users\aksha\Downloads\UNEMPLOYMENT RATE.csv")
iip = pd.read_csv(r"C:\Users\aksha\Downloads\IIP.csv")
business_confidence = pd.read_csv(r"C:\Users\aksha\Downloads\BUISNESS CONFIDENCE.csv")
interest_rate = pd.read_excel(r"C:\Users\aksha\OneDrive\Documents\INTEREST RATE.xlsx")

# Standardize column names (strip spaces and make uppercase)
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    df.columns = df.columns.str.strip().str.upper()

# Ensure 'DATE' column exists in all datasets
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    if 'DATE' not in df.columns and 'Date' in df.columns:
        df.rename(columns={'Date': 'DATE'}, inplace=True)

# Convert 'DATE' to datetime format for consistency
for df in [inflation, unemployment, iip, business_confidence, interest_rate]:
    df['DATE'] = pd.to_datetime(df['DATE'])

# Merge datasets on 'DATE'
data = interest_rate.merge(inflation, on="DATE", how="inner", suffixes=("_INTEREST", "_INFLATION"))\
    .merge(unemployment, on="DATE", how="inner", suffixes=("", "_UNEMPLOYMENT"))\
    .merge(iip, on="DATE", how="inner", suffixes=("", "_IIP"))\
    .merge(business_confidence, on="DATE", how="inner", suffixes=("", "_BUSINESS_CONFIDENCE"))

# Handle missing values and set DATE as the index
data.fillna(method='ffill', inplace=True)
data.set_index("DATE", inplace=True)

# Define features and target variable
X = data.drop(columns=["VALUE"])
y = data["VALUE"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=100)

# Train the model
xg_reg.fit(X_train_scaled, y_train)

# Predict using the test data
y_pred = xg_reg.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend()
plt.title("Actual vs Predicted Interest Rate")
plt.show()

# Plot histogram for actual vs predicted values
plt.figure(figsize=(10, 5))

# Plot histogram for actual values
plt.hist(y_test, bins=30, alpha=0.5, label='Actual Values', color='blue')

# Plot histogram for predicted values
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Values', color='red')

# Adding labels and title
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Interest Rates')

# Show legend
plt.legend()

# Display the plot
plt.show()



import numpy as np
import scipy.optimize as opt

# Define the model parameters (simplified example)
alpha = 0.33   # Capital share in output
beta = 0.99    # Discount factor
delta = 0.025  # Depreciation rate
rho = 0.95     # Autoregressive shock to productivity
sigma = 2.0    # Risk aversion parameter

# Define the production function: Y = A * K^alpha * L^(1-alpha)
def production(K, A, L):
    return A * K**alpha * L**(1-alpha)

# Define the utility function: U(C) = C^(1-sigma)/(1-sigma)
def utility(C):
    return C**(1-sigma) / (1-sigma)

# Define the equation for next period capital given by K' = (1 - delta) * K + I
def capital_evolution(K, I):
    return (1 - delta) * K + I

# Define the objective function to minimize (simplified to one period loss)
def objective(x, A, L, K):
    C, I = x
    Y = production(K, A, L)
    U = utility(C)
    K_next = capital_evolution(K, I)
    return -U + beta * utility(C)  # Negative utility for maximization

# Set initial conditions
A = 1.0   # Productivity level
L = 1.0   # Labor supply
K = 10.0  # Initial capital stock

# Solve the optimization problem using Scipy
result = opt.minimize(objective, [1.0, 1.0], args=(A, L, K), bounds=((0.1, 5), (0.1, 5)))
print(result)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = linear_model.predict(X_test_scaled)

# Evaluate model performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}')


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = linear_model.predict(X_test_scaled)

# Create a DataFrame to display years and predictions together
predictions_df = pd.DataFrame({
    'Year': X_test['Year'],  # Assuming 'Year' is a column in your X_test dataset
    'Predicted Inflation': y_pred_lr
})

# Evaluate model performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Print performance metrics
print(f'Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}')

# Display the predictions along with the years
print(predictions_df)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for years and predicted inflation values
predictions_df = pd.DataFrame({
    'Year': X_test['Year'],  # Assuming 'Year' is a column in your X_test dataset
    'Predicted Inflation': y_pred_lr
})

# Plot the histogram with 'Year' on the x-axis and 'Predicted Inflation' on the y-axis
plt.figure(figsize=(10, 6))
sns.histplot(predictions_df, x='Year', weights='Predicted Inflation', kde=False)

# Labels and title
plt.xlabel('Year')
plt.ylabel('Predicted Inflation')
plt.title('Histogram of Predicted Inflation by Year')

# Show the plot
plt.show()


# Define features and target variable
X = data.drop(columns=["VALUE"])
y = data["VALUE"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)







