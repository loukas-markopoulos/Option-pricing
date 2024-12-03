from BlackScholesModel import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


"""
Function returning the price for a call or put option calculated using a neural network. 
Network has been trained and tested on data generated from functions in BlackScholesModel.py

Class created for the Neural Network method with attributes of the required inputs.
Methods created to generate a dataset and predict the option price.
"""

class NeuralNetworkModel:
    def __init__(self, stock_price, strike_price, risk_free_rate, time_to_expiration, volatility):
        """
        Initialises variables used in the Black-Scholes formula and Neural Network model.

        stock_price: underlying current stock price
        strike_price: strike price of option
        risk_free_rate: returns on risk-free assets (constant until expiry)
        time_to_expiration: length of time option contract is valid for
        volatility: volatility of underlying stock (standard deviation of the stock's log returns)
        """
        
        self.S = stock_price            # underlying stock price
        self.X = strike_price
        self.r = risk_free_rate
        self.T = time_to_expiration     # in years
        self.volatility = volatility
        self.scaler = StandardScaler()
        self.nn_model = None
    

    def generate_dataset(self, num_samples, option_type):
        """
        Method to create a dataframe that can be used to train and test the neueral network created.
        Random values will be produced for parameters and the Black Scholes formula will be used to calculate the option price.
        """
        
        S_data = np.random.uniform(50, 150, num_samples)         # Stock price between £50 and £150
        X_data = np.random.uniform(50, 150, num_samples)         # Strike price between £50 and £150
        r_data = np.random.uniform(0.01, 0.05, num_samples)      # Risk-free rate between 0.01 and 0.05
        T_data = np.random.uniform(0.1, 2, num_samples)          # Time to expiration between 0.1 and 2 years
        volatility_data = np.random.uniform(0.1, 0.5, num_samples)    # Volatility between 0.1 and 0.5

        option_prices_data = np.zeros(num_samples)
        for i in range(num_samples):
            bsm_asset = BlackScholesModel(S_data[i], X_data[i], r_data[i], T_data[i], volatility_data[i])
            option_prices_data[i] = bsm_asset.bsm_option_price(option_type)

        x = np.vstack([S_data, X_data, r_data, T_data, volatility_data]).T
        y = option_prices_data

        return x, y

    def initialise_model(self, option_type):
        """
        Method to create, train and test a neural network using tensorflow. 
        A predetermined architecture has been implemented for simplicity.
        """

        x_train, y_train = self.generate_dataset(1000, option_type)
        self.scaler.fit(x_train)
        scaled_x_train = self.scaler.transform(x_train)

        self.nn_model = keras.models.Sequential([
            keras.layers.Input(shape=(5,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1),
        ])
        
        self.nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        history = self.nn_model.fit(scaled_x_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
        
        x_test, y_test = self.generate_dataset(200, option_type)
        scaled_x_test = self.scaler.transform(x_test)

        loss = self.nn_model.evaluate(scaled_x_test, y_test)
        print(f'Test Loss: {loss}')
    
    def nn_option_pricing(self, option_type):
        """
        Method that uses the neural network to predict the option price.
        """
        if self.nn_model is None:
            self.initialise_model(option_type)

        input = np.array([[self.S, self.X, self.r, self.T, self.volatility]])
        scaled_input = self.scaler.transform(input)

        option_price = self.nn_model.predict(scaled_input)
        return option_price
    
    # A SIMILAR THING CAN BE DONE TO PREDICT THE 5 GREEKS !!!