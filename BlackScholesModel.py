import numpy as np 
from scipy.stats import norm

"""
Function returning price for a call or put option calculated using the Black-Scholes model

Call/Put option price is calculated with following assumptions:
    - European option can be exercised only on maturity date.
    - Underlying stock does not pay divident during option's lifetime.  
    - The risk free rate and volatility are constant.
    - Efficient Market Hypothesis - market movements cannot be predicted.
    - Lognormal distribution of underlying returns.

Class created for BlackScholesModel with attributes of the required inputs
Method created to calculate the option price depending on if it is a call or put

"""

class BlackScholesModel:
    def __init__(self, stock_price, strike_price, risk_free_rate, time_to_expiration, volatility):
        """
        Initialises variables used in the Black-Scholes formula.

        stock_price: underlying current stock price
        strike_price: strike price of option
        risk_free_rate: returns on risk-free assets (constant until expiry)
        time_to_expiration: length of time option contract is valid for
        volatility: volatility of underlying stock (standard deviation of the stock's log returns)
        """
        self.S = stock_price        # underlying stock price
        self.X = strike_price
        self.r = risk_free_rate
        self.T = time_to_expiration / 365       # In days
        self.volatility = volatility
    
    def bsm_option_price(self, option_type):
        """
        Method to calculate option price

        Call Formula: S*N(d1) - X*exp(-rT)*N(d2)
        Put Formula: X*exp(-rT)*N(-d2) - S*N(-d1)
        """

        # d1 measures the distance between the underlying asset price and the strike price, adjusted for volatility and time to expiration.
        d1 = (np.log(self.S / self.X) + (self.r + 0.5 * self.volatility**2) * self.T) / (self.volatility * np.sqrt(self.T))
        # d2 is similar to d1, but it also factors in the volatility decay over time
        d2 = (np.log(self.S / self.X) + (self.r - 0.5 * self.volatility**2) * self.T) / (self.volatility * np.sqrt(self.T))
        
        try:
            if option_type == "Call":
                call_price = self.S * norm.cdf(d1, 0.0, 1.0) - self.X * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)
                return call_price
            elif option_type == "Put":
                put_price = self.X * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0)
                return put_price
        except:
            return print('Invalid input')
        
    