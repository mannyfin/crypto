import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Model(object):
    """
    Mo' models mo' problems
    """


    def __init__(self, data, percent_test_set):
        self.data = data
        self.percent_test_set = percent_test_set
        self.end_index = -int(np.floor(len(self.data) * self.percent_test_set))
        try:
            assert type(self.data) == pd.core.frame.DataFrame
        except AssertionError:
            "Data is not a DataFrame"


    def pap(self, shifted=1):
        """
        Predict that a value in the past is going to happen next
        :param self: df
        :param shifted: int, how many time periods to shift by
        :return: shifted and subtracted data
        """
        data_pap = self.data.shift(shifted).dropna()

        return data_pap

    def linear_regression(self, n_features=None):
        # will add n_features later
        # looks like the price derives heavily from the previous price based on model.coef_ . There appears to be a lag.
        # based on the previous price.
        """
        Here are example coefs:
        Index(['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',
       'Weighted_Price'],
        dtype='object')
        [[  3.03396920e-01  -2.96159416e-01  -3.07348359e-01   2.92622466e-01
        -2.17157330e-06   3.00848962e-04   1.00714852e+00]]
        """

        from sklearn.linear_model import LinearRegression
        window = 200000
        se = []
        # mse = pd.DataFrame([])
        mse = []
        guesses = []
        while window < len(self.data)+self.end_index:
            train_set, cv_set = self.walkforward_validation(data=self.data, end_index=self.end_index, window=window)

            # use all the cols for training
            trainX = train_set.iloc[:-1]
            # predict future price
            trainY = train_set.iloc[1:, train_set.columns == 'Weighted_Price']
            # below is if look ahead is 1
            testcvX = train_set.iloc[-1]
            testcvY = cv_set.iloc[:, train_set.columns == 'Weighted_Price']
            # testcvX = cv_set.iloc[:-1]
            # testcvY = cv_set.iloc[1:, train_set.columns == 'Weighted_Price']

            model = LinearRegression()
            model.fit(trainX, trainY)

            # prediction = model.predict(testcvX)
            # for 1 sample use below
            prediction = model.predict(testcvX.values.reshape(1, -1))[0]
            prediction = pd.DataFrame(data=prediction, index=testcvY.index, columns=['Weighted_Price'])
            guesses.append(prediction)
            # MSE
            vals= (prediction - testcvY) ** 2
            se.append(vals)
            # R^2
            # model.score(testcvX.values.reshape(1, -1), testcvY)
            # plot

            window += 1

            if window % 1000 == 0:
                print('window=' + str(window))
                # temp = pd.concat(se[i] for i in range(len(se)))
                # mse.append(temp.mean())
                # print('mse = ' + str(mse[-1].values[0]))
                if window % 25000 == 0:
                    ahem = [se[i].values[0][0] for i in range(len(se))]
                    mean_ahem = np.mean(ahem)
                    print('mse = ' + str(mean_ahem))
                    # mov_avg = self.moving_average(ahem)


                    # yummy = pd.concat(se[i] for i in range(len(se)))
                    # yummy.mean()
                    # rolls = yummy.rolling(5000).mean()
        # will fix ref before assignment at another time
        se = pd.concat(se[i] for i in range(len(se)))
        mse = pd.concat(mse[i] for i in range(len(mse)))
        ax = testcvY.plot()
        prediction.plot(ax=ax)

    def nn_1_keras(self):
        from keras.models import Sequential
        from keras.layers import Dense, Activation

        # Make train set the first 95% of the total dataset
        train = self.data.iloc[:-2 * int(np.floor(len(self.data) * 0.025))]
        # make CV set 2.5% of total dataset and after the train set
        cv = self.data.iloc[-2 * int(np.floor(len(self.data) * 0.025)):-int(np.floor(len(self.data) * 0.025))]
        # Make test set 2.5% of total dataset and at the very end
        test = self.data.iloc[-int(np.floor(len(self.data) * 0.025)):]
        trainX = train.iloc[:-1]
        # predict future price
        trainY = train.iloc[1:, train.columns == 'Weighted_Price']
        cv_x = cv.iloc[:-1]
        cv_y = cv.iloc[1:, cv.columns == 'Weighted_Price']

        model = Sequential()
        model.add(Dense(12, input_dim=7))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # For a mean squared error regression problem
        model.compile(optimizer='rmsprop', loss='mse')

        # Train the model, iterating on the data in batches of 32 samples
        model.fit(trainX.as_matrix(), trainY.as_matrix(), epochs=1, batch_size=32, validation_data=(cv_x.as_matrix(), cv_y.as_matrix()))
        model.fit(trainX.as_matrix(), trainY.as_matrix(), epochs=1, batch_size=32)

    @classmethod
    def walkforward_validation(cls, data, end_index, window=1, look_ahead_window=1):

        # TODO handle if beginning window > size of train set
        # TODO give option for beginning window to be float --> percent of train set, or int --> specific number of observations


        # if end_index <= window:
        #     train = data.iloc[:end_index]
        #     cv = data.iloc[end_index:end_index + look_ahead_window]
        # else:
        #     train = data.iloc[:window]
        #     cv = data.iloc[window:window + look_ahead_window]
        train = data.iloc[:window]
        cv = data.iloc[window:window + look_ahead_window]
        return train, cv

    @classmethod
    def moving_average(cls, a, n=5000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
