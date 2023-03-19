import numpy as np


class Data:

    @staticmethod
    def concat_lists(lists):

        xs = np.zeros(tuple([0] + list(lists[0][0].shape)[1:]))
        ys = np.zeros((0,), dtype=int)

        for i in range(len(lists)):
            xs = np.concatenate((xs, lists[i][0]))
            ys = np.concatenate((ys, lists[i][1]))

        return np.array(xs), np.array(ys)

    @staticmethod
    def groupByLabel(y, max_y):

        group_indices = [[] for i in range(max_y)]

        for i in range(y.shape[0]):
            group_indices[y[i]].append(i)

        return group_indices
