import numpy as np
import matplotlib.pyplot as plt
import json


class Results:

    def __init__(self, original_results=None, GMR_results=None, DSMR_results=None, all_MR_results=None):
        """
        Create an object to store and manipulate the results given multiple sets of metamorphic relations (MRs)

        :param original_results: the results using the original data
        :param GMR_results: the results when augmenting the data with generic MRs (GMRs)
        :param DSMR_results: the results when augmenting the data with domain specific MRs (DSMRs)
        :param all_MR_results: the results when augmenting the data with GMRs and DSMRs
        """

        self.original_results = original_results
        self.GMR_results = GMR_results
        self.DSMR_results = DSMR_results
        self.all_MR_results = all_MR_results

    def graph(self, train_f1s: bool = False, test_f1s: bool = True, original_counts: bool = True,
              show_sets: list = [True, True, True, True]):
        """
        Graphs the results of the deep learning with MRs

        :param train_f1s: choose whether to show the train F1 scores
        :param test_f1s: choose whether to show the test F1 scores
        :param original_counts: choose whether to show the F1 scores against the number of original training elements or the actual counts (number of training elements after the MRs)
        :param show_sets: the sets of results to show of ["original_results", "GMR_results", "DSMR_results", "all_MR_results"]. E.g. [True, False, True, False] shows ["original_results", "DSMR_results"]
        """

        legend = ["Unaltered Data", "Data + Generic MRs", "Data + MNIST Specific MRs",
                  "Data + Generic & Domain Specific MRs"]
        legend = np.array(legend)[np.array(show_sets)]

        xs = []
        ys = []

        if train_f1s:
            xs += self.get_forall_sets(show_sets, lambda x: x.train_f1)
        elif test_f1s:
            xs += self.get_forall_sets(show_sets, lambda x: x.train_f1)

        if original_counts:
            ys += self.get_forall_sets(show_sets, lambda x: x.original_count)
        else:
            ys += self.get_forall_sets(show_sets, lambda x: x.actual_count)

        plt.title("Number of given Data Points vs Test F1")
        plt.xlabel("Number of given Data Points")
        plt.ylabel("Macro F1 Score")
        plt.xscale("log", base=2)
        [plt.scatter(xs[i], ys[i]) for i in range(len(xs))]
        plt.legend(legend)
        plt.show()

    def get_forall_sets(self, is_set, get_set_function):
        """
        For all sets of results which are not None call a function

        :param is_set: the sets of results to use of ["original_results", "GMR_results", "DSMR_results", "all_MR_results"]. E.g. [True, False, True, False] uses ["original_results", "DSMR_results"]
        :param get_set_function: the function to be used with the result sets
        :return: the results of the function for each non None set
        """

        results = []

        if is_set[0] and self.original_results is not None:
            results += get_set_function(self.original_results)

        if is_set[1] and self.GMR_results is not None:
            results += get_set_function(self.GMR_results)

        if is_set[2] and self.DSMR_results is not None:
            results += get_set_function(self.DSMR_results)

        if is_set[3] and self.all_MR_results is not None:
            results += get_set_function(self.all_MR_results)

        return results

    def write_to_file(self, filename: str):
        """
        Writes the results to a file
        :param filename: the file (or path) to write the data to
        """

        text = json.dumps(self.__dict__)

        with open(filename, 'w') as file:
            file.write(text)

    @staticmethod
    def read_from_file(filename: str):
        """
        Reads results from a file in the Results class json format

        :param filename: the file (or path) to read the data from
        :return: a Results object
        """

        file = open(filename, "r")

        text = file.read()

        results = Results(json.loads(text))

        return results
