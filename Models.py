import pandas as pd
import numpy as np


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
        from sklearn.linear_model import LinearRegression
        window = 20000
        se = []
        # mse = pd.DataFrame([])
        mse = []
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

            # MSE
            vals= pd.DataFrame((prediction - testcvY) ** 2)
            se.append((prediction - testcvY) ** 2)
            # R^2
            # model.score(testcvX.values.reshape(1, -1), testcvY)
            # plot

            window += 1

            if window % 1000 == 0:
                print('window=' + str(window))
                temp = pd.concat(se[i] for i in range(len(se)))
                mse.append(temp.mean())
                print('mse = ' + str(mse[-1].values[0]))
        # will fix ref before assignment at another time
        se = pd.concat(se[i] for i in range(len(se)))
        mse = pd.concat(mse[i] for i in range(len(mse)))
        ax = testcvY.plot()
        prediction.plot(ax=ax)

    @classmethod
    def walkforward_validation(self,data, end_index, window=1, look_ahead_window=1):

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