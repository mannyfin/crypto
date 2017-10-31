import pandas as pd
import numpy as np
import datetime
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from Models import Model

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

# lr = Model(df, 0.025).linear_regression()
nn1 = Model(df, 0.025).nn_1_keras()
print('hi')
print('hi')
print('hi')
print('hi')




























# # Make train set the first 95% of the total dataset
# train = df.iloc[:-2*int(np.floor(len(df)*0.025))]
# # make CV set 2.5% of total dataset and after the train set
# cv = df.iloc[-2*int(np.floor(len(df)*0.025)):-int(np.floor(len(df)*0.025))]
# # Make test set 2.5% of total dataset and at the very end
# test = df.iloc[-int(np.floor(len(df)*0.025)):]
#
# """
# Here is a stupid way to try to predict the price using LR. It is stupid because it assumes you know the OHLV before you
# know the price, but is presented here for illustration.
# """
# trainX = train.loc[:, df.columns !='Weighted_Price']
# trainY = train['Weighted_Price']
# testcvX = cv.loc[:, df.columns !='Weighted_Price']
# testcvY = cv['Weighted_Price']
#
# model = LinearRegression()
# model.fit(trainX,trainY)
#
# prediction = model.predict(testcvX)
# prediction = pd.Series(data=prediction, index=testcvY.index, name='Prediction')
#
# # MSE
# np.mean((prediction -  testcvY) ** 2)
# # R^2
# model.score(testcvX,testcvY)
# # plot
# ax = testcvY.plot()
# prediction.plot(ax=ax)
# # plt.show()
# # if desired...
# # model.coef_
# # model.intercept_
#
# """
# A barely more realistic example
# """
#
# pap_train_prediction = Model(trainY).pap(5)
# pap_test_prediction = Model(testcvY).pap(5)
#
# pap_train_mse = np.mean((pap_train_prediction - trainY) ** 2)
# pap_test_mse = np.mean((pap_test_prediction - testcvY) ** 2)
#
# print('pap_train_mse= '+str(pap_train_mse))
# print('pap_test_mse= '+str(pap_test_mse))
#
# plt.show()