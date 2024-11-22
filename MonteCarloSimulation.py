import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Function returning price for a call or put option calculated using the Monte Carlo simulation

Underlying stock price at option's expiry date is simulated using a random stochastic process - Brownian Motion
The option price is found by averaging all simulated stock prices at maturity and discounting the final value.
"""

class MonteCarloSimulation:
    def __init__(self, stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps):
        """
        Initialises variables used in the Monte Carlo Simulation formula.

        stock_price: underlying current stock price
        strike_price: strike price of option
        risk_free_rate: returns on risk-free assets (constant until expiry)
        time_to_expiration: length of time option contract is valid for
        volatility: volatility of underlying stock (standard deviation of the stock's log returns)
        num_simulations: number of potential random underlying price movements
        num_steps: 
        """

        self.S = stock_price        # underlying stock price at option's buy date (most recent stock price)
        self.X = strike_price
        self.r = risk_free_rate
        self.T = time_to_expiration / 365       # In days
        self.volatility = volatility
        self.N = num_simulations
        # If you want to plot the individual steps of each simulation then these atributes are necessary:
        # self.num_steps = num_steps
        # self.step_size = time_to_expiration / self.num_steps
    
    def mcs_option_price(self, option_type):
        """
        Method to calculate the option price

        Both call and put options require calculating payoffs for each simulated price, averaging these and discounting them
        Payoffs for call options: max(simulated_price - strike_price, 0)
        Payoffs for put options: max(strike_price - simulated_price, 0)
        """
        
        simulated_prices = np.zeros(self.N)

        for i in range(self.N):
            # Generate a random sample from standard normal distribution
            Z = np.random.standard_normal()

            simulated_prices[i] = self.S * np.exp((self.r - 0.5 * self.volatility ** 2) * self.T + self.volatility * np.sqrt(self.T) * Z)
        
        total_payoff = 0

        try:
            if option_type == "Call":
                for simulated_value in simulated_prices:
                    payoff = max(simulated_value - self.X, 0)
                    total_payoff += payoff
                call_price = np.exp(-self.r * self.T) * (total_payoff / self.N)
                return call_price
            elif option_type == "Put":
                for simulated_value in simulated_prices:
                    payoff = max(self.X - simulated_value, 0)
                    total_payoff += payoff
                put_price = np.exp(-self.r * self.T) * (total_payoff / self.N)
                return put_price
        except:
            return print('Invalid input')

    # function to visualise the monte carlo simulation