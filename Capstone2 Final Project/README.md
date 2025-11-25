# Predicting FAANG Stock Price Movements

This project is my second capstone for the Data Science Career Track.  
The goal is to predict **next-day price direction** (up vs. not up) for FAANG stocks
using historical OHLCV data and engineered features such as returns, moving averages,
volatility, and lagged returns.

## Repository Structure

- `notebooks/` – Jupyter notebook with data cleaning, feature engineering, modeling, and evaluation.
- `metrics/model_metrics.csv` – Final model hyperparameters and performance metrics.
- `reports/Capstone_Final_Report.pdf` – Project report in PDF format.
- `slides/` – (Optional) Short slide deck summarizing the project.
- `data/` – Input CSV files (if sharable).

## Models

I compare three models:

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (final chosen model)

Gradient Boosting achieves the best balance of recall and F1-score on the test set and is selected as the final model.
