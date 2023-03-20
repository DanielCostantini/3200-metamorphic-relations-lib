import numpy as np
import random


class Data:

    def __init__(self, train_x: np.array, train_y: np.array, test_x: np.array, test_y: np.array, max_y: int):
        """
        Stores the data

        :param train_x: a numpy array, the first dimension is the index of elements for training
        :param train_y: a 1D numpy array of label indices for training
        :param test_x: a numpy array, the first dimension is the index of elements for testing
        :param test_y: a 1D numpy array of label indices for testing
        :param max_y: the largest y value possible
        """

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.train = (train_x, train_y)
        self.test = (test_x, test_y)

        self.max_y = max_y

        self.shuffle_train()

    def update_train(self, train_x: np.array, train_y: np.array):
        """
        Updates the training data

        :param train_x: new training x data
        :param train_y: new training y data
        """

        self.train_x = train_x
        self.train_y = train_y
        self.train = (train_x, train_y)

    def shuffle_train(self):
        """
        Rearranges the order of the training data
        """

        train_data = list(zip(self.train))
        random.shuffle(train_data)

        train_x = [train[0] for train in train_data]
        train_y = [train[1] for train in train_data]

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.update_train(train_x, train_y)

    def update_test(self, test_x: np.array, test_y: np.array):
        """
        Updates the testing data

        :param test_x: new testing x data
        :param test_y: new testing y data
        """

        self.test_x = test_x
        self.test_y = test_y
        self.train = (test_x, test_y)

    @staticmethod
    def concat_lists(lists: [(np.array, np.array)]) -> (np.array, np.array):
        """
        Takes a list of pairs of numpy arrays of xs and ys and makes them a single xs and ys list

        :param lists: a list of pairs of numpy arrays of xs and ys e.g. [(xs1, ys1), (xs2, ys2)]
        :return: a tuple of xs and ys e.g. (xs1 + xs2, ys1 + ys2)
        """

        xs = np.zeros(tuple([0] + list(lists[0][0].shape)[1:]))
        ys = np.zeros((0,), dtype=int)

        for i in range(len(lists)):
            xs = np.concatenate((xs, lists[i][0]))
            ys = np.concatenate((ys, lists[i][1]))

        return np.array(xs), np.array(ys)

    @staticmethod
    def group_by_label(y: np.array, max_y: int) -> [[int]]:
        """
        Groups an array of ints by their values.
        E.g. ([3, 3, 2, 3, 1, 0], 4) -> [[5], [4], [2], [0, 1, 3], []]

        :param y: a numpy array of ints
        :param max_y: the maximum possible value
        :return: a list of y indices for each possible y value
        """

        group_indices = [[] for _ in range(max_y)]

        for i in range(y.shape[0]):
            group_indices[y[i]] += i

        return group_indices

    def get_train_subset(self, i_min=0, i_max=-1):

        return self.train_x[i_min:i_max], self.train_y[i_min:i_max]
