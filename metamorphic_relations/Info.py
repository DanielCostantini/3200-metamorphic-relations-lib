import json


class Info:

    def __init__(self, original_count: [int], actual_count: [int], train_f1: [float], test_f1: [float]):
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

    @staticmethod
    def list_to_info(results):

        original_count = [o for (o, a, tr, te) in results]
        actual_count = [a for (o, a, tr, te) in results]
        train_f1 = [tr for (o, a, tr, te) in results]
        test_f1 = [te for (o, a, tr, te) in results]

        return Info(original_count, actual_count, train_f1, test_f1)

