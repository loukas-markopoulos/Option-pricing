import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
"""
Gets stock data from yahoo finance.

Three functions:
    - Returns a pandas dataframe of the underlying stock price varying with time
    - Plots historical data for the underlying stock price
    - Returns the most recent stock price
Both have one parameter:
    - ticker: stock's ticker symbol
"""

def get_data(ticker):
    data = yf.download(ticker, period='1y')
    df = data[[('Adj Close', ticker)]]
    return df

def plot_price_hist(ticker):
    df = get_data(ticker)
    df['Adj Close'].plot()
    plt.show()

def recent_price(ticker):
    df = get_data(ticker)
    return df['Adj Close'].iloc[-1]

# apple_ticker = 'AAPL'
# print(get_data(apple_ticker).head())
# print(get_data(apple_ticker).columns)
# plot_price_hist(apple_ticker)
# recent_stock_price = recent_price(apple_ticker)
# print(recent_stock_price)

