from metamorphic_relations.Data import Data
from metamorphic_relations.Transform import Transform

import numpy as np


class MR:
    """
    Creates an object to represent the tree of all combinations of the transforms

    :param transforms: a list of transforms
    :param max_composite: the maximum number of transformations that can be performed sequentially on each data element
    """

    def __init__(self, transforms: list[Transform], max_composite: int = 1):

        self.transforms = transforms
        self.MR_tree = None
        self.MR_list = None
        self.MR_list_names = None
        self.update_composite(max_composite)

    def update_composite(self, max_composite: int):
        """
        Updates the tree based on the current list and max_composite

        :param max_composite: the maximum number of transformations that can be performed sequentially on each data element
        """

        self.MR_tree = self.get_composite_tree(max_composite)
        self.MR_list = self.get_composite_list()
        self.MR_list_names = self.get_composite_list_names()

    def get_composite_tree(self, max_composite: int) -> list[tuple[Transform, list]]:
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

    def add_composite(self, max_composite: int, used_indices: list[int], prev_y: int) -> list[tuple[Transform, list]]:
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

    def get_composite_list(self, MR_tree: list[tuple[Transform, list]] = None) -> list[tuple[Transform, list]]:
        """
        Takes a tree of MRs and converts it to a list of paths through the tree

        :param MR_tree: the tree to convert
        :return: a list of trees each of which has a single path from root to leaf
        """

        if MR_tree is None:
            MR_tree = self.MR_tree

        MR_list = []

        for i in range(len(MR_tree)):

            if len(MR_tree[i][1]) <= 1:

                MR_list.append(MR_tree[i])

            else:

                MR_list += [(MR_tree[i][0], [ts]) for ts in self.get_composite_list(MR_tree[i][1])]

        return MR_list

    def get_composite_list_names(self) -> list[str]:
        """
        Gets the names of each of the composite MRs

        :return: a list of string names
        """

        MR_list_names = []

        for lst in self.MR_list:
            MR_list_names.append(self.get_composite_name(lst))

        return MR_list_names

    @staticmethod
    def get_composite_name(transform_list: tuple[Transform, list]) -> str:
        """
        Gets the name of a composite MR, by concatenating each Transform's individual name

        :param transform_list: the composite MR representation
        :return: a string name
        """

        if len(transform_list[1]) == 0:
            return transform_list[0].name
        else:
            return transform_list[0].name + " -> " + MR.get_composite_name(transform_list[1][0])

    @staticmethod
    def for_all_labels(transform, label_current_indices: list[int] = None, label_target_indices: list[int] = None,
                       name: str = None) -> list[Transform]:
        """
        Adds transforms for a given set of labels

        :param function transform: the transformation function
        :param label_current_indices: the indices of labels to use this transform on (default leads to all labels)
        :param label_target_indices: the indices of labels to give after the transform (default leads to labels remaining the same)
        :param name: the name of the transform by default uses the function name
        :return: a list of transforms
        """

        if name is None:
            name = transform.__str__()

        MR_list = []

        if label_current_indices is None:

            MR_list.append(Transform(transform, -1, -1, name))

        elif label_target_indices is None:

            for i in label_current_indices:
                MR_list.append(Transform(transform, i, i, name + " (" + str(i) + " to " + str(i) + ")"))

        elif len(label_current_indices) == len(label_target_indices):

            for i in range(len(label_current_indices)):
                MR_list.append(Transform(transform, label_current_indices[i], label_target_indices[i], name + " (" + str(label_current_indices[i]) + " to " + str(label_target_indices[i]) + ")"))

        else:

            raise Exception("The current and target indices must have the same length")

        return MR_list

    @staticmethod
    def scale_values_transform(x: np.array, scale_func) -> np.array:
        """
        Scales all the values of the input

        :param x: input data in the form of a numpy array
        :param function scale_func: a function to be applied to all int values in the input
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

        if len(xs) != len(ys):
            raise Exception("xs and ys must have the same length")

        if len(self.MR_tree) == 0:
            return xs, ys

        groups = Data.group_by_label(ys, max_y)

        return Data.concat_lists([(xs, ys)] + [MR.perform_MRs(t, xs, ys, groups) for t in self.MR_tree])

    @staticmethod
    def perform_MRs_list(MR_list: tuple[Transform, list], xs: np.array, ys: np.array, max_y: int) -> np.array:
        """
        Performs the entire tree of MRs on the given data

        :param MR_list: an element of a list of MRs
        :param xs: x numpy array
        :param ys: y numpy array
        :param max_y: the largest value y can be
        :return: the transformed data and corresponding labels
        """

        if len(xs) != len(ys):
            raise Exception("xs and ys must have the same length")

        if len(MR_list) == 0:
            return xs, ys

        groups = Data.group_by_label(ys, max_y)

        return Data.concat_lists([(xs, ys)] + [MR.perform_MRs(MR_list, xs, ys, groups)])

    @staticmethod
    def perform_MRs(transform_branch: tuple[Transform, list], xs: np.array, ys: np.array,
                    groups: list[list[int]]) -> tuple[np.array, np.array]:
        """
        Performs a single branch of the MRs tree

        :param transform_branch: the branch of MRs in the form (current_transform, [following_transforms])
        :param xs: x numpy array
        :param ys: y numpy array
        :param groups: indexed labels of the y data
        :return: the transformed data and corresponding labels
        """

        if len(xs) != len(ys):
            raise Exception("xs and ys must have the same length")

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

        :param function transform: the transformation function
        :param xs: numpy array of x data
        :return: the transformed data (the shape is the same as the input)
        """

        mr_xs = np.zeros(xs.shape)

        for i in range(len(xs)):
            mr_xs[i] = transform(xs[i])

        return mr_xs

    @staticmethod
    def perform_DSMR(transform, xs: np.array, indices: list[int]) -> tuple[np.array, int]:
        """
        Performs the DSMR on the x data given by indices

        :param function transform: the transformation function
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
