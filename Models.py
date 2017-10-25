class Model(object):
    """
    Mo' models mo' problems
    """
    def __init__(self, data):
        self.data = data

    def pap(self, shifted=1):
        """
        Predict that a value in the past is going to happen next
        :param self: df
        :param shifted: int, how many time periods to shift by
        :return: shifted and subtracted data
        """
        data_pap = self.data.shift(shifted).dropna()

        return data_pap
