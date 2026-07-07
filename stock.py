import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# STEP 1: Download stock data (Example: Reliance - Indian stock)
print("Downloading stock data...")
data = yf.download("RELIANCE.NS", start="2020-01-01", end="2026-01-01")

# STEP 2: Feature Engineering - create simple indicators
data['MA10'] = data['Close'].rolling(window=10).mean()   # 10-day moving average
data['MA50'] = data['Close'].rolling(window=50).mean()   # 50-day moving average
data['Price_Change'] = data['Close'].pct_change()         # daily % change

# Target: next day's closing price
data['Target'] = data['Close'].shift(-1)

# Remove rows with missing values (first 50 rows will have NaN due to MA50)
data = data.dropna()

# STEP 3: Prepare features (X) and target (y)
features = ['Close', 'MA10', 'MA50', 'Price_Change']
X = data[features]
y = data['Target']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# STEP 4: Train 3 models and compare
print("\nTraining models...\n")

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_error = mean_absolute_error(y_test, lr_pred)

# Model 2: Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_error = mean_absolute_error(y_test, dt_pred)

# Model 3: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_error = mean_absolute_error(y_test, rf_pred)

# STEP 5: Show results
print("===== MODEL COMPARISON (Lower error = Better) =====")
print(f"Linear Regression MAE: {lr_error:.2f}")
print(f"Decision Tree MAE: {dt_error:.2f}")
print(f"Random Forest MAE: {rf_error:.2f}")

# STEP 6: Show sample predictions vs actual
print("\n===== SAMPLE PREDICTIONS (Random Forest) =====")
results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': rf_pred[:10]
})
print(results)

print("\nDone! Project executed successfully.")