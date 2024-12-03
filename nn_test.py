from NeuralNetwork import *

stock_price = 100
strike_price = 100
risk_free_rate = 0.05
time_to_expiration = 1
volatility = 0.2

nn_asset = NeuralNetworkModel(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility)

call_option_price = nn_asset.nn_option_pricing("Call")
put_option_price = nn_asset.nn_option_pricing("Put")

print(f'Call option price: {call_option_price}')
print(f'Put option price: {put_option_price}')