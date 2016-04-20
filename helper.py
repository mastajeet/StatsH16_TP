import pandas as pd
import finance as fin
def yf_price_extractor(path, *, number_datapoint=None):
    pricelist = pd.read_csv(path)
    pricelist = pricelist.rename(columns={'Close': 'Price'})
    fields_to_return = ['Date', 'Price']

    if number_datapoint is None:
        return pricelist[fields_to_return]
    elif number_datapoint is not None:
        if number_datapoint == 1:
            return pricelist[fields_to_return].iloc[0]
        if number_datapoint < 1:
            raise ValueError('Number of data point should at least be equal to one')
        else:
            return pricelist[fields_to_return].iloc[:number_datapoint+1]

def yf_yield_extractor(path, *, number_datapoint=None):
    prices = yf_price_extractor(path, number_datapoint=number_datapoint)
    return fin.yield_calculator(prices)