import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Function returning price for a call or put option calculated using the Monte Carlo simulation

Underlying stock price at option's expiry date is simulated using a random stochastic process - Brownian Motion
The option price is found by averaging all simulated stock prices at maturity and discounting the final value.
"""

class MonteCarloSimulation:
    
    @staticmethod
    def mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps,  option_type):
        """
        Method to calculate the option price

        stock_price: underlying current stock price
        strike_price: strike price of option
        risk_free_rate: returns on risk-free assets (constant until expiry)
        time_to_expiration: length of time option contract is valid for
        volatility: volatility of underlying stock (standard deviation of the stock's log returns)
        num_simulations: number of potential random underlying price movements
        num_steps: steps for each simulation

        A random seed is set for reproducibility

        Both call and put options require calculating payoffs for each simulated price, averaging these and discounting them
        Payoffs for call options: max(simulated_price - strike_price, 0)
        Payoffs for put options: max(strike_price - simulated_price, 0)
        """

        np.random.seed(42)

        T = time_to_expiration / 365
        step_size = T / num_steps

        simulated_prices = np.zeros((num_simulations, num_steps + 1))
        simulated_prices[:, 0] = stock_price

        for step in range(1, num_steps + 1):
            Z = np.random.standard_normal(num_simulations)
            simulated_prices[:,step] = simulated_prices[:, step - 1] * np.exp((risk_free_rate - 0.5 * volatility**2) * step_size + volatility * np.sqrt(step_size) * Z)
        
        # MonteCarloSimulation.visualise_mcs(simulated_prices, 100, T, num_steps, option_type)

        try:
            if option_type == "Call":
                payoffs = np.maximum(simulated_prices[:, -1] - strike_price, 0)
            
            elif option_type == "Put":
                payoffs = np.maximum(strike_price - simulated_prices[:, -1], 0)
            
            option_price = np.exp(-risk_free_rate * T) * np.mean(payoffs)
            return option_price
        except:
            print("Invalid input")

    @staticmethod
    def visualise_mcs(simulated_prices, num_plot_paths, T, num_steps, option_type):
        plot_indices = np.random.choice(simulated_prices.shape[0], size = num_plot_paths, replace=False)
        selected_plots = simulated_prices[plot_indices]

        for i in range(num_plot_paths):
            plt.plot(np.linspace(0, T, num_steps + 1), selected_plots[i], color='red', alpha = 0.1)
        plt.title(f"Monte Carlo Simulation of {option_type} Option Price")
        plt.xlabel("Time to Maturity (years)")
        plt.ylabel("Option Price")
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def mcs_greeks(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps,  option_type):
        """
        Method to calculate the 5 greeks of options using the monte carlo simulation.

        Delta: measures how the option price changes with a small change (epsilon) in the underlying price
        Gamma: measures how delta changes with a small change (epsilon) in the underlying price
        Theta: measures how the option price changes with a small change (epsilon) in the time to maturity
        Vega: measures how the option price changes with a small change (epsilon) in the volatility
        Rho: measures how the option price changes with a small change (epsilon) in the risk-free interest
        """

        delta_gamma_epsilon = 0.005 * stock_price
        vega_epsilon = 0.01 * volatility
        theta_epsilon = 0.01 * time_to_expiration
        rho_epsilon = 0.001 * risk_free_rate

        try:
            if option_type == "Call":
                delta = (MonteCarloSimulation.mcs_option_price((stock_price + delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call") - MonteCarloSimulation.mcs_option_price((stock_price - delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call")) / (2 * delta_gamma_epsilon)
                gamma = (MonteCarloSimulation.mcs_option_price((stock_price + delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call") + MonteCarloSimulation.mcs_option_price((stock_price - delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call") - 2 * MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call")) / (delta_gamma_epsilon**2)
                vega = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, (volatility + vega_epsilon), num_simulations, num_steps, "Call") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, (volatility - vega_epsilon), num_simulations, num_steps, "Call")) / (2 * vega_epsilon)
                theta = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, (time_to_expiration + theta_epsilon), volatility, num_simulations, num_steps, "Call") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, (time_to_expiration - theta_epsilon), volatility, num_simulations, num_steps, "Call")) / (2 * theta_epsilon)
                rho = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, (risk_free_rate + rho_epsilon), time_to_expiration, volatility, num_simulations, num_steps, "Call") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, (risk_free_rate - rho_epsilon), time_to_expiration, volatility, num_simulations, num_steps, "Call")) / (2 * rho_epsilon)

            elif option_type == "Put":
                delta = (MonteCarloSimulation.mcs_option_price((stock_price + delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put") - MonteCarloSimulation.mcs_option_price((stock_price - delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put")) / (2 * delta_gamma_epsilon)
                gamma = (MonteCarloSimulation.mcs_option_price((stock_price + delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put") + MonteCarloSimulation.mcs_option_price((stock_price - delta_gamma_epsilon), strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put") - 2 * MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put")) / (delta_gamma_epsilon**2)
                vega = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, (volatility + vega_epsilon), num_simulations, num_steps, "Put") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, (volatility - vega_epsilon), num_simulations, num_steps, "Put")) / (2 * vega_epsilon)
                theta = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, (time_to_expiration + theta_epsilon), volatility, num_simulations, num_steps, "Put") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, risk_free_rate, (time_to_expiration - theta_epsilon), volatility, num_simulations, num_steps, "Put")) / (2 * theta_epsilon)
                rho = (MonteCarloSimulation.mcs_option_price(stock_price, strike_price, (risk_free_rate + rho_epsilon), time_to_expiration, volatility, num_simulations, num_steps, "Put") - MonteCarloSimulation.mcs_option_price(stock_price, strike_price, (risk_free_rate - rho_epsilon), time_to_expiration, volatility, num_simulations, num_steps, "Put")) / (2 * rho_epsilon)
        except:
            print('Invalid input')
        
        return delta, gamma, vega, theta, rho