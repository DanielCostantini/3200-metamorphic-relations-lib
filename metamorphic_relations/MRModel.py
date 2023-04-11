import copy

from keras import Model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from statistics import mean
import numpy as np
import math

from metamorphic_relations.Results import Results
from metamorphic_relations.MR import MR
from metamorphic_relations.Info import Info
from metamorphic_relations.Data import Data
from metamorphic_relations.Transform import Transform

import logging

LOG_FORMAT = '%(levelname) -s %(asctime)s %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger.info('logging started')


class MRModel:
    """
    Creates and MRModel object

    :param data: the data to be used with the model
    :param model: the ML model
    :param function transform_x: the transform of the x from the data representation to the expected input to the model
    :param function transform_y: the transform of the y from the data representation to the expected output from the model
    :param GMRs: list of Generic Metamorphic Relations (GMRs)
    :param DSMRs: list of Domain Specific Metamorphic Relations (DSMRs)
    """

    def __init__(self, data: Data, model: Model, transform_x=None, transform_y=None, GMRs: list[Transform] = None,
                 DSMRs: list[Transform] = None):

        self.data = data
        self.model = model
        self.GMRs = MR(GMRs)
        self.DSMRs = MR(DSMRs)
        self.all_MRs = MR(GMRs + DSMRs)

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

    def compare_MR_sets_counts(self, max_composite: int = 1, min_i: int = 4,
                               compare_sets: tuple[bool, bool, bool, bool] = (True, True, True, True)) -> tuple[Results, list[Model]]:
        """
        Trains the model on each set of MRs using increasing proportions of the data

        :param max_composite: determines the max number of MRs that can be applied consecutively to produce new training data
        :param min_i: the smallest set of data to test calculated as 2**min_i
        :param compare_sets:the sets of results to compare of ["original", "GMRs", "DSMRs", "all_MRs"]. E.g. [True, False, True, False] compares ["original", "DSMRs"]
        :return: a Results object and list of the best model for each set (in the same order as the input to compare_sets)
        """

        if len(compare_sets) != 4:
            raise Exception("compare_sets must have four boolean values")

        if min_i < 0:
            raise Exception("min_i must be greater than or equal to 0")

        self.GMRs.update_composite(max_composite)
        self.DSMRs.update_composite(max_composite)
        self.all_MRs.update_composite(max_composite)

        max_i = int(math.ceil(math.log2(len(self.data.train_x))))

        i_vals = [int(2 ** i) for i in range(min_i, max_i)]

        results = Results()
        models = []

        if compare_sets[0]:
            results.original_results, model = self.get_results(MR([]), i_vals=i_vals)
            models.append(model)
        if compare_sets[1]:
            results.GMR_results, model = self.get_results(self.GMRs, i_vals=i_vals)
            models.append(model)
        if compare_sets[2]:
            results.DSMR_results, model = self.get_results(self.DSMRs, i_vals=i_vals)
            models.append(model)
        if compare_sets[3]:
            results.all_MR_results, model = self.get_results(self.all_MRs, i_vals=i_vals)
            models.append(model)

        return results, models

    def compare_MR_sets(self, max_composite: int = 1, compare_sets: tuple[bool, bool, bool, bool] = (True, True, True, True)) -> tuple[Results, list[Model]]:
        """
        Trains the model on each set of MRs using all the training data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :param compare_sets:the sets of results to compare of ["original", "GMRs", "DSMRs", "all_MRs"]. E.g. [True, False, True, False] compares ["original", "DSMRs"]
        :return: a Results object and list with the model for each set (in the same order as the input to compare_sets)
        """

        if len(compare_sets) != 4:
            raise Exception("compare_sets must have four boolean values")

        self.GMRs.update_composite(max_composite)
        self.DSMRs.update_composite(max_composite)
        self.all_MRs.update_composite(max_composite)

        results = Results()
        models = []

        if compare_sets[0]:
            results.original_results, model = self.get_results(MR([]))
            models.append(model)
        if compare_sets[1]:
            results.GMR_results, model = self.get_results(self.GMRs)
            models.append(model)
        if compare_sets[2]:
            results.DSMR_results, model = self.get_results(self.DSMRs)
            models.append(model)
        if compare_sets[3]:
            results.all_MR_results, model = self.get_results(self.all_MRs)
            models.append(model)

        return results, models

    def compare_MR_counts(self, max_composite: int = 1, min_i: int = 4) -> tuple[Results, list[Model]]:
        """
        Trains the model on each MR individually using increasing proportions of the data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :param min_i: the smallest set of data to test calculated as 2**min_i
        :return: a Results object and list of the best model for each individual MR (in the same order as results)
        """

        if min_i < 0:
            raise Exception("min_i must be greater than or equal to 0")

        self.GMRs.update_composite(max_composite)
        self.DSMRs.update_composite(max_composite)
        self.all_MRs.update_composite(max_composite)

        max_i = int(math.ceil(math.log2(len(self.data.train_x))))

        i_vals = [int(2 ** i) for i in range(min_i, max_i)]

        results = Results()
        models = []

        results.original_results = self.get_results(MR([]), i_vals=i_vals)

        all_results = []

        if max_composite == 1:
            for i in range(len(self.GMRs.MR_list)):
                res, model = self.get_results(i_vals=i_vals, MR_list=self.GMRs.MR_list[i])
                res.set_name(self.GMRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

            for i in range(len(self.DSMRs.MR_list)):
                res, model = self.get_results(i_vals=i_vals, MR_list=self.DSMRs.MR_list[i])
                res.set_name(self.DSMRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

        if max_composite != 1:
            for i in range(len(self.all_MRs.MR_list)):
                res, model = self.get_results(i_vals=i_vals, MR_list=self.all_MRs.MR_list[i])
                res.set_name(self.all_MRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

        results.individual_results = all_results

        return results, models

    def compare_MRs(self, max_composite: int = 1) -> tuple[Results, list[Model]]:
        """
        Trains the model on each MR individually using all the training data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :return: a Results object and list of the best model for each individual MR (in the same order as results)
        """

        self.GMRs.update_composite(max_composite)
        self.DSMRs.update_composite(max_composite)
        self.all_MRs.update_composite(max_composite)

        results = Results()
        models = []

        results.original_results, model = self.get_results(MR([]))
        models.append(model)

        all_results = []

        if max_composite == 1:
            for i in range(len(self.GMRs.MR_list)):
                res, model = self.get_results(MR_list=self.GMRs.MR_list[i])
                res.set_name(self.GMRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

            for i in range(len(self.DSMRs.MR_list)):
                res, model = self.get_results(MR_list=self.DSMRs.MR_list[i])
                res.set_name(self.DSMRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

        if max_composite != 1:
            for i in range(len(self.all_MRs.MR_list)):
                res, model = self.get_results(MR_list=self.all_MRs.MR_list[i])
                res.set_name(self.all_MRs.MR_list_names[i])
                all_results.append(res)
                models.append(model)

        results.individual_results = all_results

        return results, models

    def get_results(self, MR_obj: MR = None, MR_list: list[tuple[Transform, list]] = None, i_vals: list[int] = None) -> tuple[Info, Model]:
        """
        Returns the results of training the data on the model with the MRs

        :param MR_obj: the MR object - this should be given if the MR tree should be used i.e. to get the results when using set of MRs
        :param MR_list: a list of transforms to be performed compositely - this should be used when getting results for an individual (potentially composite) MR
        :param i_vals: the intervals to get results for
        :return: an Info object containing the results for this MR
        """

        if i_vals is None:
            i_vals = [len(self.data.train_x)]

        if (MR_obj is None and MR_list is None) or (MR_obj is not None and MR_list is not None):
            raise Exception("Exactly one of: MR_obj or MR_list must be provided")

        results = []
        best_model = None
        best_test_f1 = 0.0

        for i in i_vals:
            #         Takes the first sample of elements
            new_train_x, new_train_y = self.data.get_train_subset(i_max=i)

            #         Performs the MRs and returns only the new data
            if MR_list is None:
                MR_train_x, MR_train_y = MR_obj.perform_MRs_tree(new_train_x, new_train_y, self.data.max_y)
            else:
                MR_train_x, MR_train_y = MR.perform_MRs_list(MR_list, new_train_x, new_train_y, self.data.max_y)

            new_train_x, new_train_y = self.transform_data(MR_train_x, MR_train_y)
            logging.info("got new data")

            #         Trains and tests the model given the collected data
            train_f1 = self.train_model(new_train_x, new_train_y)
            logging.info("trained")

            test_f1 = self.test_model()

            logging.info((i, len(new_train_x), train_f1, test_f1))
            results.append((i, len(new_train_x), train_f1, test_f1))

            if test_f1 > best_test_f1:
                best_model = copy.deepcopy(self.model)

        set_result = Info.list_to_info(results)

        return set_result, best_model

    def train_model(self, train_x: np.array, train_y: np.array, k: int = 5) -> float:
        """
        Trains the model and sets it to the best performing model of the k folds

        :param train_x: the x data
        :param train_y: the y data
        :param k: the number of folds for k fold validation
        :return: the mean macro f1 score over the training folds
        """

        if len(train_x) != len(train_y):
            raise Exception("Training data and labels must have the same length")

        if k <= 0:
            raise Exception("k must be a positive integer")

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

    def test_model(self, test_x: np.array = None, test_y: np.array = None) -> float:
        """
        Tests the model to find macro f1 scores

        :param test_x: the x data (default uses the data.train_x)
        :param test_y: the y data (default uses the data.train_y)
        :return: the macro f1 score
        """

        if test_x is None:
            test_x = self.data.test_x

        if test_y is None:
            test_y = self.data.test_y

        if len(test_x) != len(test_y):
            raise Exception("Testing data and labels must have the same length")

        f1 = f1_score(self.y_2D_to_1D(self.model.predict(test_x, verbose=0)), self.y_2D_to_1D(test_y),
                      average='macro')

        return f1

    @staticmethod
    def concat(x: np.array, y: np.array, new_x: np.array, new_y: np.array) -> tuple[np.array, np.array]:
        """
        Concatenates 2 arrays

        :param x: x data
        :param y: y data
        :param new_x: data to add to x
        :param new_y: data to add to y
        :return: (x + new_x, y + new_y)
        """

        x = np.concatenate((x, new_x))
        y = np.concatenate((y, new_y))

        return x, y

    def transform_data(self, x: np.array, y: np.array) -> tuple[np.array, np.array]:
        """
        Transforms the data to be used in the ML model

        :param x: x data
        :param y: y data
        :return: (new_x_data, new_y_data)
        """

        return self.transform_x(x), self.transform_y(y, self.data.max_y)

    @staticmethod
    def y_2D_to_1D(y: np.array) -> np.array:
        """
        Reshapes a 2D array to a 1D array containing the index of the highest value.
        E.g. [[0, 1], [1, 0]] -> [1, 0]

        :param y: the original array
        :return: the 1D array
        """

        if len(y.shape) != 2:
            raise Exception("y must have two dimensions")

        new_y = np.zeros(y.shape[0])

        for i in range(y.shape[0]):
            new_y[i] = np.argmax(y[i])

        return new_y

    @staticmethod
    def y_1D_to_2D(y: np.array, max_y: int) -> np.array:
        """
        Reshapes a 1D array to a 2D array containing the index of the highest value.
        E.g. [1, 0] -> [[0, 1], [1, 0]]

        :param y: the original array
        :param max_y: the largest possible value in the array
        :return: the 2D array
        """

        if len(y.shape) != 1:
            raise Exception("y must have one dimensions")

        new_y = np.zeros((y.shape[0], max_y))

        #     Reshapes a 1D array to a 2D array containing the 1 at the index of the y value
        #     E.g. [1, 0] -> [[0, 1], [1, 0]]
        for i in range(len(y)):

            if y[i] >= max_y:
                raise Exception("max_y must be larger than all values in y")

            new_y[i][y[i]] = 1

        return new_y

