from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import time


app = Flask(__name__)

def fetch_stock_data(stock_code, start_date, end_date):
    data = yf.download(stock_code, start=start_date, end=end_date)
    return data

def plot_anomalies(data, stock_code):
    try:
        data['Return'] = data['Adj Close'].pct_change().dropna()
        window = 20
        data['Moving_Avg'] = data['Adj Close'].rolling(window).mean()
        data['Volatility'] = data['Adj Close'].rolling(window).std()
        features = data[['Return', 'Moving_Avg', 'Volatility']].dropna()

        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(features)
        features['Anomaly'] = model.predict(features)
        features['Anomaly_Score'] = model.decision_function(features[['Return', 'Moving_Avg', 'Volatility']])

        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Adj Close'], label='Stock Price')
        anomalies = features[features['Anomaly'] == -1]
        plt.scatter(anomalies.index, data.loc[anomalies.index, 'Adj Close'], color='red', label='Anomaly', marker='x')
        plt.title(f'{stock_code} Stock Price with Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    except Exception as e:
        print(f"Error in plot_anomalies: {e}")
        return None

def plot_predictions(data, stock_code):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Adj Close']].dropna())

        training_size = int(len(data_scaled) * 0.70)
        train_data, test_data = data_scaled[0:training_size, :], data_scaled[training_size:len(data_scaled), :1]

        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 100
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, Y_train, batch_size=1, epochs=1)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        Y_train = scaler.inverse_transform([Y_train])
        Y_test = scaler.inverse_transform([Y_test])

        look_back = time_step
        train_predict_plot = np.empty_like(data_scaled)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

        test_predict_plot = np.empty_like(data_scaled)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(data_scaled) - 1, :] = test_predict

        plt.figure(figsize=(14, 7))
        plt.plot(scaler.inverse_transform(data_scaled), label='Original Data')
        plt.plot(train_predict_plot, label='Train Prediction')
        plt.plot(test_predict_plot, label='Test Prediction')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    except Exception as e:
        print(f"Error in plot_predictions: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url_anomalies = None
    plot_url_predictions = None
    eta = None

    if request.method == 'POST':
        stock_code = request.form['stock_code']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_time = time.time()

        data = fetch_stock_data(stock_code, start_date, end_date)
        if not data.empty:
            plot_url_anomalies = plot_anomalies(data, stock_code)
            plot_url_predictions = plot_predictions(data, stock_code)
        
        end_time = time.time()
        eta = end_time - start_time

    return render_template('index.html', plot_url_anomalies=plot_url_anomalies, plot_url_predictions=plot_url_predictions, eta=eta)

