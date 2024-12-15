# Option-pricing

In this project I have explored how options are valued based on given financial parameters. <br/>
Three modelling techniques have been implemented: <br/>
- Black-Scholes calculation,
- Monte Carlo simulation,
- Neural network predeiction (Tensorflow). <br/>

Each technique calculates the price of either a call or put option when given the following financial parameters: <br/>
- underlying stock price,
- strike price of the option,
- risk free rate,
- time to expiration,
- volatility of the stock. <br/>

Each technique also has functions that return the 5 greeks of the desired option that are used by traders to indicate an option's price's sensitivity.
