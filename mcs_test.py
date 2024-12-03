from data import recent_price
from MonteCarloSimulation import *

stock_price = 100
strike_price = 100
risk_free_rate = 0.05
time_to_expiration = 1
volatility = 0.2
num_simulations = 50000
num_steps = 365

mcs_asset = MonteCarloSimulation()
call_price = mcs_asset.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call")
put_price = mcs_asset.mcs_option_price(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put")
call_delta, call_gamma, call_theta, call_vega, call_rho = mcs_asset.mcs_greeks(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Call")
put_delta, put_gamma, put_theta, put_vega, put_rho = mcs_asset.mcs_greeks(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, num_simulations, num_steps, "Put")

print(f'call price: {call_price}, put price: {put_price}')
print(f'call delta: {call_delta}, put delta: {put_delta}')
print(f'call gamma: {call_gamma}, put gamma: {put_gamma}')
print(f'call theta: {call_theta}, put theta: {put_theta}')
print(f'call vega: {call_vega}, put vega: {put_vega}')
print(f'call rho: {call_rho}, put rho: {put_rho}')