import json


class Info:

    def __init__(self, original_count: list, actual_count: list, train_f1: list, test_f1: list):
        """
        Create an object to store and manipulate the information for a given set of metamorphic relations (MRs)

        :param original_count: the number of training elements before the MRs were used
        :param actual_count: the number of training elements used in training (i.e. after the MRs were used)
        :param train_f1: the average F1 score over the training folds
        :param test_f1: the F1 score calculated on the test set
        """

        if not len(original_count) == len(actual_count) and len(actual_count) == len(train_f1) \
                and len(train_f1) == len(test_f1):
            raise Exception("Must have same number of counts and f1 scores")

        self.original_count = original_count
        self.actual_count = actual_count
        self.train_f1 = train_f1
        self.test_f1 = test_f1

    def toJSON(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def fromJSON(dictionary):
        return Info(dictionary["actual_count"], dictionary["original_count"], dictionary["train_f1"], dictionary["test_f1"])
