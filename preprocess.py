import pandas as pd
import os
import datetime


def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


os.chdir('..')
os.chdir('btc data')
data = pd.read_csv('bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv', parse_dates=True, date_parser=dateparse, index_col=[0])

data['Open'].isnull().sum()

data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

# standardize the data
df = (data - data.mean())/ data.std()

return df