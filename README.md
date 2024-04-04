AMD Stock price forecasting
# LSTM Stock Price Prediction

## Overview
This repository contains code for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The LSTM model is trained on historical stock price data fetched from Yahoo Finance. The project includes data preprocessing, model training, and making predictions.

## Project Logic and Code Flow
1. **Data Retrieval**: Historical stock price data is fetched from Yahoo Finance using the `yfinance` library.

2. **Data Preprocessing**: 
   - The fetched data is preprocessed, including adding technical indicators such as Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and Moving Average (MA).
   - The data is split into training and testing sets. 

3. **Model Definition**: 
   - The LSTM model architecture is defined in the `LSTMModel` class, which includes an LSTM layer followed by fully connected layers.
   - Hyperparameters such as dropout rate, number of hidden units, number of layers, etc., are defined.

4. **Model Training**: 
   - The model is trained using the training data.
   - Grid search is performed over different hyperparameter combinations to find the best model based on validation loss (commented out in the code).

5. **Model Evaluation**: 
   - The best model based on validation loss is selected.
   - The model is evaluated on the testing data to calculate training and validation loss.
   - Training and validation loss curves are plotted to visualize model performance.

6. **Prediction**: 
   - Future stock price prediction is performed using the trained model.
   - The model is used to make sequential predictions for a specified number of future time steps.
   - The predicted prices are inverse-transformed to obtain the actual stock prices.

## Dependencies
- Python 3
- PyTorch
- pandas
- numpy
- yfinance
- matplotlib
- scikit-learn
- tqdm
- talib

## Forecasting Future Stock Prices
The trained LSTM model in this project is capable of forecasting future stock prices for a specified number of days. By default, the model forecasts 24 days of future stock prices for the AMD (Advanced Micro Devices) stock.

To change the forecast duration or the stock symbol:
1. Open the `lstm_stock_prediction.py` file.
2. Modify the `forecast_days` variable to change the number of days to forecast.
3. Modify the `ticker_symbol` variable to change the stock symbol (e.g., "AMD", "INTC", "NVDA").

After making the changes, rerun the code to generate new forecasts.
