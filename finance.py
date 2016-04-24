import pandas as pd
import numpy as np

def yield_calculator(prices):
    yields = np.array(prices['Price'].iloc[:-1])/np.array(prices['Price'].iloc[1:])
    dates = prices['Date'].iloc[:-1]
    return pd.DataFrame(np.transpose([dates,yields]),columns=['Date','Yield'])

def log_yield_calculator(prices):
    yields = np.log(np.array(prices['Price'].iloc[:-1])) - np.log(np.array(prices['Price'].iloc[1:]))
    dates = prices['Date'].iloc[:-1]
    return pd.DataFrame(np.transpose([dates,yields]),columns=['Date','Yield'])