# import the libraries
import math
# import pandas_datareader as web
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

plt.style.use('fivethirtyeight')

yf.pdr_override()

st.title("Stock price prediction")

# Text input
#user_input = st.text_input("Enter the Stock Code", 'AAPL')

@st.cache_data
def fun(user_input):
    # Get the stock qoute
    df = pdr.get_data_yahoo(user_input, start="2010-1-15", end="2023-03-21")
    # df = web.DataReader('AAPL', data_source='yahoo', start='2007-10-01', end='2021-05-04')
    # show the data

    # df

    st.subheader('Date from 2010 to 2023')
    st.write(df.describe())

    # df.shape
    # Visualize the closing price history
    plt.figure(figsize=(16, 8))
    plt.title('Closing Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD $', fontsize=18)
    st.pyplot(plt)

    # Create a new Dataframe with only the Close Coloumn
    data = df.filter(['Close'])
    # Convert the dataframe to numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create the Scaled Training Data Set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Create the testing data set
    # Create a new array containing scaled values from index 1746 to 2257
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean sqaured error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD $', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # plt.show()
    st.pyplot(plt)
    return model, scaler

user_input = st.text_input("Enter the Stock Code", 'AAPL')

if st.button("Submit"):
    model, scaler = fun(user_input)
    #model, scaler = fun(user_input)
    date = date.today()
    date200 = date-timedelta(days = 200)
    print(date)
    apple_quote = pdr.get_data_yahoo(
        user_input, start=date200, end=date)
    # Create a new dataFrame
    new_df = apple_quote.filter(['Close'])
    # Get the last 60 day closing price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # Create an empty list
    X_test = [last_60_days_scaled]
    # Append the past 60 days
    # Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    # Undo the Scaling
    pred_price = scaler.inverse_transform(pred_price)
    st.write("# Predicted price:")
    st.write(pred_price[0][0])
    apple_quote2 = pdr.get_data_yahoo(
        user_input, start=date-timedelta(days=2), end=date)
    st.write("""
        # Original price:
        """)
    st.write(apple_quote2['Close'][0])

st.write("""
# Stock prices codes with company name 

|               Stock code                     |          Stock company   |
|----------------------------------------------|:------------------------:|
|TSLA                                          |Tesla                     |
|GOOG                                          |Alphabet Inc.             |
|AMZN                                          |Amazon.com, Inc.          |
|V                                             |Visa Inc                  |
|ENPH                                          |Enphase Energy, Inc.      |
|META                                          |Meta Platforms, Inc.      |
|NVDA                                          |NVIDIA Corporation        |
|NFLX                                          |Netflix, Inc.             |
|AAPL                                          |APPLE                     |

for more code consider the link ("https://finance.yahoo.com/lookup/") 

 """)
# streamlit run app.py
