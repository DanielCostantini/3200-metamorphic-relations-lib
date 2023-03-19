from keras import Model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from statistics import mean
import numpy as np
import random
import math

from metamorphic_relations import Transform, Data, Results, MR, Info


class MRModel:

    def __init__(self, data: Data, model: Model, GMRs: [Transform] = None, DSMRs: [Transform] = None):

        self.data = data
        self.model = model
        self.GMRs = GMRs
        self.DSMRs = DSMRs
        self.all_MRs = GMRs + DSMRs

        self.model.save_weights("Output/initial_weights.h5")

    def compare_MR_sets(self, max_composite: int = 1, min_i: int = 4, flatten: bool = True) -> Results:
        """
        Trains the model on each set of MRs using increasing proportions of the data

        :param max_composite: default = 1, determines the max number of MRs that can be applied consecutively to produce new training data
        :param min_i: the smallest set of data to test calculated as 2**min_i
        :param flatten: determines whether the data should be flattened before being fed into the model
        """

        #     The randomly generated initial weights of the model are saved to file so the model can be quickly reset without
        #     recompiling the whole model

        GMRs = MR(self.GMRs, max_composite)
        DSMRs = MR(self.DSMRs, max_composite)
        both = MR(self.all_MRs, max_composite)

        max_i = int(math.ceil(math.log2(len(self.data.train_x))))

        i_vals = [int(2 ** i) for i in range(min_i, max_i)]

        results = Results()

        results.original_results = self.get_results(MR([], 1), i_vals, flatten)
        results.GMR_results = self.get_results(GMRs, i_vals, flatten)
        results.DSMR_results = self.get_results(DSMRs, i_vals, flatten)
        results.all_MR_results = self.get_results(both, i_vals, flatten)

        return results

    def get_results(self, MR_tree: MR, i_vals: [int], flatten: bool) -> Info:
        """
        Returns the results of training the data on the model with the MRs
        In the form of a list of (Number of training elements used, k-fold training macro f1, testing macro f1)
        """

        test_x, test_y = transform_data(test_x, test_y, max_y, flatten)

        results = Info()

        for i in i_vals:

            #         Takes the first sample of elements
            new_train_x, new_train_y = train_x[:max_index], train_y[:max_index]

            #         Performs the MRs and returns only the new data
            MR_train_x, MR_train_y = MRs.perform_MRs_list(MR_list, new_train_x, new_train_y, max_y)
            new_train_x, new_train_y = transform_data(MR_train_x, MR_train_y, max_y, flatten)

            #         Trains and tests the model given the collected data
            trained_model, train_f1 = train_model(model, new_train_x, new_train_y)
            test_f1 = test_model(trained_model, test_x, test_y)

            print(len(new_train_x), train_f1, test_f1)
            results.append((len(new_train_x), train_f1, test_f1))

        return results

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

        return self.model, mean(f1)

    def test_model(self, test_x, test_y):
        f1 = f1_score(transform_predictions(self.model.predict(test_x, verbose=0)), transform_predictions(test_y),
                      average='macro')

        return f1

    def concat(x, y, new_x, new_y):
        x = np.concatenate((x, new_x))
        y = np.concatenate((y, new_y))

        return x, y

    def transform_data(x, y, max_y, flatten):
        new_y = np.zeros((y.shape[0], max_y))

        #     Reshapes a 1D array to a 2D array containing the 1 at the index of the y value
        #     E.g. [1, 0] -> [[0, 1], [1, 0]]
        for i in range(len(y)):
            new_y[i][y[i]] = 1

        if flatten:
            new_x = x.reshape((x.shape[0], -1))

        else:
            new_x = x

        return new_x, new_y

    def transform_predictions(predictions):
        """
        Reshapes a 2D array to a 1D array containing the index of the highest value.
        E.g. [[0, 1], [1, 0]] -> [1, 0]
        """

        new_pred = np.zeros(predictions.shape[0])

        for i in range(predictions.shape[0]):
            new_pred[i] = np.argmax(predictions[i])

        return new_pred

    def shuffle_train(self):
        """
        Rearranges the order of the training data
        """

        train_data = list(zip(self.data.train))
        random.shuffle(train_data)

        train_x = [train[0] for train in train_data]
        train_y = [train[1] for train in train_data]

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.data.update_train(train_x, train_y)
