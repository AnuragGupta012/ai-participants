# Import necessary libraries
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Download historical data from Yahoo Finance
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Prepare data for training
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data

# Create training and testing datasets
def create_datasets(scaled_data, train_size):
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]
    return train_data, test_data

# Split the data into input and output
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[(i + time_step), 0])
    return np.array(dataX), np.array(dataY)

# Create the LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Plot the predictions
def plot_predictions(train_data, train_predict, test_data, test_predict):
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, color='blue', label='Original Data')
    plt.plot(train_predict, color='red', label='Predicted Data')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Main function
def main():
    # Download historical data
    ticker = 'BTC-USD'
    start_date = '2010-01-01'
    end_date = '2022-02-26'
    data = download_data(ticker, start_date, end_date)

    # Prepare data for training
    scaled_data = prepare_data(data)

    # Create training and testing datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = create_datasets(scaled_data, train_size)

    # Split the data into input and output
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create the LSTM model
    model = create_model(X_train)

    # Train the model
    train_model(model, X_train, y_train)

    # Make predictions
    train_predict = make_predictions(model, X_train)
    test_predict = make_predictions(model, X_test)

    # Plot the predictions
    plot_predictions(train_data, train_predict, test_data, test_predict)

if __name__ == "__main__":
    main()
