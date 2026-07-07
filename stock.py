import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Downloading stock data...")
data = yf.download("RELIANCE.NS", start="2020-01-01", end="2026-01-01")

data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Price_Change'] = data['Close'].pct_change()
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

features = ['Close', 'MA10', 'MA50', 'Price_Change']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("\nTraining models...\n")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_error = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_error = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_error = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("===== MODEL COMPARISON (Lower error = Better) =====")
print(f"Linear Regression MAE: {lr_error:.2f} | R2: {lr_r2:.3f}")
print(f"Decision Tree MAE: {dt_error:.2f} | R2: {dt_r2:.3f}")
print(f"Random Forest MAE: {rf_error:.2f} | R2: {rf_r2:.3f}")

print("\n===== SAMPLE PREDICTIONS (Random Forest) =====")
results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': rf_pred[:10]
})
print(results)

print("\nDone! Project executed successfully.")

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(rf_pred, label='Predicted Price (Random Forest)', color='red', linestyle='--')
plt.title('Stock Price: Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.savefig('prediction_chart.png')
print("\nChart saved as prediction_chart.png")
plt.show()
