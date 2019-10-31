import numpy as np
from tqdm import tqdm, trange


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, Xtr, ytr, yte):
        # X is N x D where each row is an example. Y is 1-dimension of size N
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = Xtr
        self.ytr = ytr
        self.yte = yte

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        # find the nearest training image to the i'th test image
        # using the L1 distance (sum of absolute value differences)
        print('The length of X test', num_test)
        range_of_Xtest = trange(num_test, desc='Complete')
        for i in range_of_Xtest:
            # Updating the accuracy during learning process
            tqdm.set_postfix_str(range_of_Xtest, s='Accuracy: %3.3f ' % (np.mean(self.yte == Ypred) * 100) + "%",
                                 refresh=True)

            # L1 distance with the sum of absolute value. Accuracy: 38.6%
            distances = np.sum(np.abs(X[i, :] - self.Xtr), axis=1)

            # L2 distance with the sum of square root value
            # distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))

            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred
