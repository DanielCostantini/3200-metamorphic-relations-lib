import math
import numpy as np

import pytest

from metamorphic_relations.Data import Data
from metamorphic_relations.MR import MR
from metamorphic_relations.ImageMR import ImageMR

mr = MR(ImageMR.get_image_GMRs())


def test_MR_update_max_composite_correct():
    org_tree = mr.MR_tree

    mr.update_composite(2)

    assert org_tree != mr.MR_tree


def test_MR_get_composite_tree_correct_max_composite_1():
    assert len(mr.get_composite_tree(1)) == len(ImageMR.get_image_GMRs())


def test_MR_get_composite_tree_correct_max_composite_2():
    assert len(mr.get_composite_tree(2)[0][1]) == 3


def test_MR_get_composite_tree_incorrect_max_composite_0():
    with pytest.raises(Exception):
        mr.get_composite_tree(0)


def test_MR_add_composite_correct():
    mr.update_composite(1)

    assert len(mr.add_composite(1, [0], -1)) == 3


def test_MR_add_composite_correct_max_composite_0():
    mr.update_composite(1)

    assert len(mr.add_composite(0, [0], -1)) == 0


def test_MR_add_composite_correct_prev_y_1():
    mr.update_composite(1)

    assert len(mr.add_composite(0, [0], 1)) == 0


def test_MR_get_composite_list_correct():
    mr.update_composite(1)
    depth = 4
    mr.update_composite(depth)

    assert len(mr.get_composite_list()) == math.factorial(len(ImageMR.get_image_GMRs())) / math.factorial(
        len(ImageMR.get_image_GMRs()) - depth)


def test_MR_get_composite_list_names_correct():
    mr.update_composite(1)
    depth = 4
    mr.update_composite(depth)

    assert mr.MR_list_names[0] == "Rotate by 5 degrees -> Rotate by -5 degrees -> Scale values by +10 -> Blur"


def test_MR_get_composite_name_correct_length_0():
    mr.update_composite(1)
    assert MR.get_composite_name(mr.MR_list[0]) == "Rotate by 5 degrees"


def test_MR_get_composite_name_correct_length_1():
    mr.update_composite(2)
    assert MR.get_composite_name(mr.MR_list[0]) == "Rotate by 5 degrees -> Rotate by -5 degrees"


def test_MR_for_all_labels_correct_all_ys():
    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10))[0].target == -1


def test_MR_for_all_labels_correct_labels_stay_same():
    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5])[0].target == 1


def test_MR_for_all_labels_correct_labels_change():
    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5], [10, 6])[1].current == 5
    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5], [10, 6])[1].target == 6


def test_MR_for_all_labels_incorrect_length():
    with pytest.raises(Exception):
        MR.for_all_labels(lambda x: x, [1, 2], [3])


def test_MR_scale_values_transform_correct_1D():
    assert np.array_equal(MR.scale_values_transform(np.array([1, 2]), lambda x: x + 1), np.array([2, 3]))


def test_MR_scale_values_transform_correct_2D():
    assert np.array_equal(MR.scale_values_transform(np.array([[1, 2], [3, 4]]), lambda x: x * 2),
                          np.array([[2, 4], [6, 8]]))


def test_MR_perform_MRs_tree_correct():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)
    xs, ys = mr.perform_MRs_tree(xs_org, ys_org, 3)

    assert len(xs) == len(ys)
    assert len(xs) == len(xs_org) * (len(ImageMR.get_image_GMRs()) + 1)


def test_MR_perform_MRs_tree_correct_empty_tree():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    mr.MR_tree = []
    xs, ys = mr.perform_MRs_tree(xs_org, ys_org, 3)

    assert np.array_equal(xs_org, xs)
    assert np.array_equal(ys_org, ys)


def test_MR_perform_MRs_tree_incorrect_xs_length():

    xs_org = np.array([[[1, 2]], [[3, 4]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)

    with pytest.raises(Exception):
        mr.perform_MRs_tree(xs_org, ys_org, 3)


def test_MR_perform_MRs_list_correct():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)
    xs, ys = MR.perform_MRs_list(mr.MR_list[0], xs_org, ys_org, 3)

    assert len(xs) == len(ys)
    assert len(xs) == len(xs_org) * 2  # a single transform is done on each xs


def test_MR_perform_MRs_list_correct_empty_list():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    xs, ys = MR.perform_MRs_list([], xs_org, ys_org, 3)

    assert np.array_equal(xs_org, xs)
    assert np.array_equal(ys_org, ys)


def test_MR_perform_MRs_list_incorrect_xs_length():

    xs_org = np.array([[[1, 2]], [[3, 4]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)

    with pytest.raises(Exception):
        MR.perform_MRs_list(mr.MR_list[0], xs_org, ys_org, 3)


def test_MR_perform_MRs_correct():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)
    xs, ys = MR.perform_MRs(mr.MR_list[0], xs_org, ys_org, Data.group_by_label(ys_org, 3))

    assert len(xs) == len(xs_org)


def test_MR_perform_MRs_correct_composite():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(2)
    xs, ys = MR.perform_MRs(mr.MR_tree[0], xs_org, ys_org, Data.group_by_label(ys_org, 3))

    assert len(xs) == len(xs_org) * 3  # does first transform giving 3, and then those 3 have 3 other transforms giving 9 xs


def test_MR_perform_MRs_incorrect_xs_length():

    xs_org = np.array([[[1, 2]], [[3, 4]]])
    ys_org = np.array([0, 1, 2])
    mr.update_composite(1)

    with pytest.raises(Exception):
        MR.perform_MRs(mr.MR_list[0], xs_org, ys_org, Data.group_by_label(ys_org, 3))


def test_MR_perform_GMR_correct():

    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])

    xs = mr.perform_GMR(ImageMR.get_image_GMRs()[0].func, xs_org)

    assert len(xs) == len(xs_org)


def test_MR_perform_DSMR_correct():
    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])

    xs, _ = mr.perform_DSMR(ImageMR.get_image_GMRs()[0].func, xs_org, [0, 2])

    assert len(xs) == 2


def test_MR_perform_DSMR_correct_empty_indices():
    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])

    xs, _ = mr.perform_DSMR(ImageMR.get_image_GMRs()[0].func, xs_org, [])

    assert len(xs) == 0


def test_MR_perform_DSMR_correct_function_returns_None():
    xs_org = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])

    _, none_count = mr.perform_DSMR(lambda x: None, xs_org, [0, 1])

    assert none_count == 2
