# Stock Price Prediction using Machine Learning

## Project Overview
This project predicts next-day stock closing prices using historical data and technical indicators.

## Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Features Used
- Closing Price
- 10-day Moving Average (MA10)
- 50-day Moving Average (MA50)
- Daily Price Change %

## Results (Mean Absolute Error)
| Model | MAE |
|-------|-----|
| Linear Regression | 12.52 |
| Random Forest | 18.91 |
| Decision Tree | 32.69 |

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- yfinance

## Key Learning
Linear Regression performed best on this dataset, showing that stock prices often follow near-linear short-term trends. Random Forest and Decision Tree showed higher error due to overfitting on limited features.
