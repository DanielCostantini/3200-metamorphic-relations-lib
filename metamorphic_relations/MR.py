from metamorphic_relations import Data
from metamorphic_relations.Transform import Transform

import numpy as np


class MR:

    def __init__(self, transforms: [Transform], max_composite: int = 1):
        """
        Creates an object to represent the tree of all combinations of the transforms

        :param transforms: a list of transforms
        :param max_composite: the maximum number of transformations that can be performed sequentially on each data element
        """

        self.transforms = transforms
        self.MRs = self.get_composite(max_composite)

    def update_tree(self, max_composite: int):
        """
        Updates the tree based on the current list and max_composite

        :param max_composite: the maximum number of transformations that can be performed sequentially on each data element
        """

        self.MRs = self.get_composite(max_composite)

    def get_composite(self, max_composite: int) -> [(Transform, [])]:
        """
        Gets the tree of composite transforms

        :param max_composite: the maximum number of transformations that can be performed sequentially on each data element
        :return: The tree of composite transforms
        """

        if max_composite <= 0:
            raise Exception("max_composite must be positive")

        max_composite = min(len(self.transforms), max_composite)

        composite_MRs = [(self.transforms[i], self.add_composite(max_composite - 1, [i], self.transforms[i].target)) for
                         i in range(len(self.transforms))]

        return composite_MRs

    def add_composite(self, max_composite: int, used_indices: [int], prev_y: int) -> [(Transform, [])]:
        """
        Adds a branch to the tree of composite transforms

        :param max_composite: the maximum number of remaining transformations that can be performed sequentially given the previous transforms more shallow in the tree
        :param used_indices: the transforms that have already been used in this branch
        :param prev_y: the final y value after the previous transform
        :return: the tree from this point of possible combinations of transforms
        """

        if max_composite == 0:
            return []

        if prev_y == -1:
            return [(self.transforms[i],
                     self.add_composite(max_composite - 1, [i] + used_indices, self.transforms[i].target))
                    for i in range(len(self.transforms)) if i not in used_indices]

        return [
            (self.transforms[i], self.add_composite(max_composite - 1, [i] + used_indices, self.transforms[i].target))
            for i in range(len(self.transforms)) if
            i not in used_indices and (self.transforms[i].current == prev_y or self.transforms[i].current == -1)]

    @staticmethod
    def for_all_labels(transform, label_current_indices: [int] = None, label_target_indices: [int] = None) -> [Transform]:
        """
        Adds transforms for a given set of labels

        :param transform: the transformation function
        :param label_current_indices: the indices of labels to use this transform on (default leads to all labels)
        :param label_target_indices: the indices of labels to give after the transform (default leads to labels remaining the same)
        :return: a list of transforms
        """

        MR_list = []

        if label_current_indices is None:

            MR_list.append(Transform(transform, -1, -1))

        elif label_target_indices is None:

            for i in label_current_indices:
                MR_list.append(Transform(transform, i, i))

        elif len(label_current_indices) == len(label_target_indices):

            for i in range(len(label_current_indices)):
                MR_list.append(Transform(transform, label_current_indices[i], label_target_indices[i]))

        else:

            raise Exception("The current and target indices must have the same length")

        return MR_list

    @staticmethod
    def scale_values_transform(x: np.array, scale_func) -> np.array:
        """
        Scales all the values of the input

        :param x: input data in the form of a numpy array
        :param scale_func: a function to be applied to all int values in the input
        :return: the transformed data
        """

        return np.vectorize(scale_func)(x)

    def perform_MRs_tree(self, xs: np.array, ys: np.array, max_y: int) -> np.array:
        """
        Performs the entire tree of MRs on the given data

        :param xs: x numpy array
        :param ys: y numpy array
        :param max_y: the largest value y can be
        :return: the transformed data and corresponding labels
        """

        if len(self.MRs) == 0:
            return xs, ys

        groups = Data.group_by_label(ys, max_y)

        print(xs.shape)

        return Data.concat_lists([(xs, ys)] + [MR.perform_MRs(t, xs, ys, groups) for t in self.MRs])

    @staticmethod
    def perform_MRs(transform_branch: (Transform, []), xs: np.array, ys: np.array, groups: [[int]]) -> np.array:
        """
        Performs a single branch of the MRs tree

        :param transform_branch: the branch of MRs in the form (current_transform, [following_transforms])
        :param xs: x numpy array
        :param ys: y numpy array
        :param groups: indexed labels of the y data
        :return: the transformed data and corresponding labels
        """

        transform, current_y, target_y = transform_branch[0].func, transform_branch[0].current, transform_branch[
            0].target
        next_transforms = transform_branch[1]

        if current_y == -1:
            xs = MR.perform_GMR(transform, xs)
        else:
            xs, none_count = MR.perform_DSMR(transform, xs, groups[current_y])
            ys = np.full((len(groups[current_y]) - none_count,), target_y)
            groups = Data.group_by_label(ys, len(groups))

        if len(next_transforms) != 0:
            xs, ys = Data.concat_lists([MR.perform_MRs(t, xs, ys, groups) for t in next_transforms])

        return xs, ys

    @staticmethod
    def perform_GMR(transform, xs: np.array) -> np.array:
        """
        Performs the GMR on all the x data

        :param transform: the transformation function
        :param xs: numpy array of x data
        :return: the transformed data (the shape is the same as the input)
        """

        mr_xs = np.zeros(xs.shape)

        for i in range(len(xs)):
            mr_xs[i] = transform(xs[i])

        return mr_xs

    @staticmethod
    def perform_DSMR(transform, xs: np.array, indices: [int]) -> (np.array, int):
        """
        Performs the DSMR on the x data given by indices

        :param transform: the transformation function
        :param xs: numpy array of x data
        :param indices: indexed labels of the y data this MR should be performed on
        :return: the transformed data
        """

        mr_xs = np.zeros(tuple([len(indices)] + list(xs.shape)[1:]))

        j = 0
        none_count = 0

        for i in indices:

            t = transform(xs[i])

            if t is None:
                none_count += 1

            else:
                mr_xs[j] = t
                j += 1

        new_mr_xs = np.zeros(tuple([len(mr_xs) - none_count] + list(mr_xs.shape)[1:]))

        for i in range(len(new_mr_xs)):
            new_mr_xs[i] = mr_xs[i]

        return new_mr_xs, none_count
