from keras import Model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from statistics import mean
import numpy as np
import math

from metamorphic_relations import Transform, Data, Results, MR, Info


class MRModel:

    def __init__(self, data: Data, model: Model, transform_x=None, transform_y=None, GMRs: [Transform] = None, DSMRs: [Transform] = None):

        self.data = data
        self.model = model
        self.GMRs = GMRs
        self.DSMRs = DSMRs
        self.all_MRs = GMRs + DSMRs

        if transform_x is not None:
            self.transform_x = transform_x
        else:
            self.transform_x = lambda x: x

        if transform_y is not None:
            self.transform_y = transform_y
        else:
            self.transform_y = self.y_1D_to_2D

        #     The randomly generated initial weights of the model are saved to file so the model can be quickly reset without
        #     recompiling the whole model
        self.model.save_weights("Output/initial_weights.h5")

        test_x, test_y = self.transform_data(self.data.test_x, self.data.test_y)
        self.data.update_test(test_x, test_y)

    def compare_MR_sets_counts(self, max_composite: int = 1, min_i: int = 4) -> Results:
        """
        Trains the model on each set of MRs using increasing proportions of the data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :param min_i: the smallest set of data to test calculated as 2**min_i
        """

        GMRs = MR(self.GMRs, max_composite)
        DSMRs = MR(self.DSMRs, max_composite)
        both = MR(self.all_MRs, max_composite)

        max_i = int(math.ceil(math.log2(len(self.data.train_x))))

        i_vals = [int(2 ** i) for i in range(min_i, max_i)]

        results = Results()

        results.original_results = self.get_results(MR([], 1), i_vals)
        results.GMR_results = self.get_results(GMRs, i_vals)
        results.DSMR_results = self.get_results(DSMRs, i_vals)
        results.all_MR_results = self.get_results(both, i_vals)

        return results

    def compare_MR_sets(self, max_composite: int = 1) -> Results:
        """
        Trains the model on each set of MRs using increasing proportions of the data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :param min_i: the smallest set of data to test calculated as 2**min_i
        """

        GMRs = MR(self.GMRs, max_composite)
        DSMRs = MR(self.DSMRs, max_composite)
        both = MR(self.all_MRs, max_composite)

        results = Results()

        results.original_results = self.get_results(MR([], 1))
        results.GMR_results = self.get_results(GMRs)
        results.DSMR_results = self.get_results(DSMRs)
        results.all_MR_results = self.get_results(both)

        return results

    def get_results(self, MR_tree: MR, i_vals: [int] = None) -> Info:
        """
        Returns the results of training the data on the model with the MRs
        In the form of a list of (Number of training elements used, k-fold training macro f1, testing macro f1)
        """

        if i_vals is None:
            i_vals = [-1]

        results = []

        for i in i_vals:

            #         Takes the first sample of elements
            new_train_x, new_train_y = self.data.get_train_subset(i_max=i)

            #         Performs the MRs and returns only the new data
            MR_train_x, MR_train_y = MR_tree.perform_MRs_tree(new_train_x, new_train_y, self.data.max_y)
            new_train_x, new_train_y = self.transform_data(MR_train_x, MR_train_y)

            #         Trains and tests the model given the collected data
            train_f1 = self.train_model(new_train_x, new_train_y)
            test_f1 = self.test_model()

            print(len(new_train_x), train_f1, test_f1)
            results.append((i, len(new_train_x), train_f1, test_f1))

        set_result = Info.list_to_info(results)

        return set_result

    def train_model(self, train_x, train_y, k=5):
        # Classic deep learning classification
        # https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

        kf = KFold(n_splits=k, shuffle=True)
        f1 = []
        i = 0

        #     Splits the data into k folds, trains the model with k-1 folds and evaluates the macro f1 with the remaining fold
        #     The mean of these are taken to give a better estimate of performance than taking out a single validation set
        for train_index, val_index in kf.split(train_x):
            train_xk, val_xk = train_x[train_index], train_x[val_index]
            train_yk, val_yk = train_y[train_index], train_y[val_index]

            #         Resets the model weights
            self.model.load_weights("Output/initial_weights.h5")
            self.model.fit(train_xk, train_yk, epochs=20, batch_size=100, verbose=0)

            self.model.save("Output/train_weights" + str(i) + ".h5")
            f1.append(self.test_model(val_xk, val_yk))

            i += 1

        #     After the f1s have been found for each fold the model is trained using all the available data for the best performance

        best_train = np.argmax(np.array(f1))
        self.model.load_weights("Output/train_weights" + str(best_train) + ".h5")

        return mean(f1)

    def test_model(self, test_x=None, test_y=None):

        if test_x is None:
            test_x = self.data.test_x

        if test_y is None:
            test_y = self.data.test_y

        f1 = f1_score(self.y_2D_to_1D(self.model.predict(test_x, verbose=0)), self.y_2D_to_1D(test_y),
                      average='macro')

        return f1

    @staticmethod
    def concat(x, y, new_x, new_y):
        x = np.concatenate((x, new_x))
        y = np.concatenate((y, new_y))

        return x, y

    def transform_data(self, x, y):

        return self.transform_x(x), self.transform_y(y, self.data.max_y)

    @staticmethod
    def y_2D_to_1D(predictions):
        """
        Reshapes a 2D array to a 1D array containing the index of the highest value.
        E.g. [[0, 1], [1, 0]] -> [1, 0]
        """

        new_pred = np.zeros(predictions.shape[0])

        for i in range(predictions.shape[0]):
            new_pred[i] = np.argmax(predictions[i])

        return new_pred

    @staticmethod
    def y_1D_to_2D(y, max_y):

        new_y = np.zeros((y.shape[0], max_y))

        #     Reshapes a 1D array to a 2D array containing the 1 at the index of the y value
        #     E.g. [1, 0] -> [[0, 1], [1, 0]]
        for i in range(len(y)):
            new_y[i][y[i]] = 1

        return new_y
