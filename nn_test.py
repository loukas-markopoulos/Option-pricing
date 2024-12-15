from NeuralNetwork import *
from BlackScholesModel import *

stock_price = 100
strike_price = 100
risk_free_rate = 0.05
time_to_expiration = 1
volatility = 0.2

nn_asset = NeuralNetworkModel(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility)
bsm_asset = BlackScholesModel(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility)

# call_option_price = nn_asset.nn_option_pricing("Call")
# put_option_price = nn_asset.nn_option_pricing("Put")
# print(f'Call option price: {call_option_price}')
# print(f'Put option price: {put_option_price}')

delta, gamma, theta, vega, rho = nn_asset.nn_greeks_prediction("Call")
call_delta, call_gamma, call_theta, call_vega, call_rho = bsm_asset.bsm_greeks("Call")
print(f'{delta}, {call_delta}')
print(f'{gamma}, {call_gamma}')
print(f'{theta}, {call_theta}')
print(f'{vega}, {call_vega}')
print(f'{rho}, {call_rho}')