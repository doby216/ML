import numpy as np
import keras
from keras.datasets import cifar10

from nearestNeighbor import NearestNeighbor

if __name__ == "__main__":
    (Xtr, Ytr), (Xte, Yte) = cifar10.load_data()  # a magic function we provide
    # flatten out all images to be one-dimensional

    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3).astype("float32")  # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3).astype("float32")  # Xte_rows becomes 10000 x 3072

    # Flatten Y train and Y test
    Ytr = Ytr.flatten()
    Yte = Yte.flatten()

    nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, Ytr, Yte)  # train the classifier on the training images and labels
    Yte_predict = nn.predict(Xte_rows)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)

    print('accuracy: %3.3f ' % (np.mean(Yte_predict == Yte)*100), "%")
