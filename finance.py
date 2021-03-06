import pandas as pd
import numpy as np
import scipy.stats as ss

def yield_calculator(prices):
    yields = np.array(prices['Price'].iloc[:-1])/np.array(prices['Price'].iloc[1:])
    dates = prices['Date'].iloc[:-1]
    return pd.DataFrame(np.transpose([dates,yields]),columns=['Date','Yield'])

def log_yield_calculator(prices):
    yields = np.log(np.array(prices['Price'].iloc[:-1])) - np.log(np.array(prices['Price'].iloc[1:]))
    dates = prices['Date'].iloc[:-1]
    return pd.DataFrame(np.transpose([dates,yields]),columns=['Date','Yield'])


def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def black_scholes(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))