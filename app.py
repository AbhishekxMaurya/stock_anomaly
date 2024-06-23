from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import time

app = Flask(__name__)

def fetch_data(stock_code, start_date, end_date):
    data = yf.download(stock_code, start=start_date, end=end_date)
    return data

def process_data_for_anomaly_detection(data):
    data['Return'] = data['Adj Close'].pct_change().dropna()
    window = 20
    data['Moving_Avg'] = data['Adj Close'].rolling(window).mean()
    data['Volatility'] = data['Adj Close'].rolling(window).std()
    features = data[['Return', 'Moving_Avg', 'Volatility']].dropna()
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    features['Anomaly'] = model.predict(features)
    features['Anomaly_Score'] = model.decision_function(features[['Return', 'Moving_Avg', 'Volatility']])
    return features, data

def process_data_for_lstm(data):
    data = data[['Adj Close']]
    data = data.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    training_size = int(len(data_scaled) * 0.70)
    test_size = len(data_scaled) - training_size
    train_data, test_data = data_scaled[0:training_size, :], data_scaled[training_size:len(data_scaled), :1]
    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Add debug statements
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of Y_train: {Y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of Y_test: {Y_test.shape}")

    # Handle case where dataset is too small for specified time_step
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return None, None, None

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=1, epochs=1)
    return model, scaler, data_scaled

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def forecast_future(model, data, time_step, forecast_period, scaler):
    forecast_list = []
    input_data = data[-time_step:]
    input_data = input_data.reshape((1, time_step, 1))

    for _ in range(forecast_period):
        pred = model.predict(input_data, verbose=0)
        forecast_list.append(pred[0, 0])
        input_data = np.append(input_data[:, 1:, :], pred.reshape((1, 1, 1)), axis=1)

    forecast_list = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1))
    return forecast_list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_code = request.form['stock_code']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    data = fetch_data(stock_code, start_date, end_date)
    features, processed_data = process_data_for_anomaly_detection(data)
    
    # Simulate progress for the progress bar
    time.sleep(2)  # Simulate initial processing delay
    
    model, scaler, data_scaled = process_data_for_lstm(data)

    # Check if model is None
    if model is None:
        return "Dataset is too small for the specified time_step. Please choose a larger date range."

    # Generate anomaly detection plot
    plt.figure(figsize=(14, 7))
    plt.plot(processed_data.index, processed_data['Adj Close'], label='Stock Price')
    anomalies = features[features['Anomaly'] == -1]
    plt.scatter(anomalies.index, processed_data.loc[anomalies.index, 'Adj Close'], color='red', label='Anomaly', marker='x')
    plt.title(f'{stock_code} Stock Price with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    anomaly_img = io.BytesIO()
    plt.savefig(anomaly_img, format='png')
    anomaly_img.seek(0)
    anomaly_img_url = base64.b64encode(anomaly_img.getvalue()).decode()
    plt.close()

    # Forecast for the next 3 months (approximately 60 trading days)
    forecast_period = 60
    forecasted_prices = forecast_future(model, data_scaled, 100, forecast_period, scaler)

    # Generate future prediction plot
    plt.figure(figsize=(14, 7))
    plt.plot(scaler.inverse_transform(data_scaled), label='Original Data')
    plt.plot(range(len(data_scaled), len(data_scaled) + forecast_period), forecasted_prices, label='Forecasted Prices', color='orange')
    plt.legend()
    future_img = io.BytesIO()
    plt.savefig(future_img, format='png')
    future_img.seek(0)
    future_img_url = base64.b64encode(future_img.getvalue()).decode()
    plt.close()

    return render_template('result.html', anomaly_img_url=anomaly_img_url, future_img_url=future_img_url)

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

if __name__ == "__main__":
    app.run(debug=True)
