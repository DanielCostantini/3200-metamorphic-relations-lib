import math

import pytest

from metamorphic_relations.MR import MR
from metamorphic_relations.ImageMR import ImageMR


def test_get_composite_1():

    assert len(MR(ImageMR.get_image_GMRs(), 1).MR_tree) == 4


def test_get_composite_2():

    assert len(MR(ImageMR.get_image_GMRs(), 2).MR_tree[0][1]) == 3


def test_for_all_labels_generic():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10))[0].target == -1


def test_for_all_labels_stay_same():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5])[0].target == 1


def test_for_all_labels_change():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5], [10, 6])[1].current == 5


def test_get_composite_list():

    mr = MR(ImageMR.get_image_GMRs())

    depth = 4
    mr.update_composite(depth)
    assert len(mr.get_composite_list()) == math.factorial(len(ImageMR.get_image_GMRs())) / math.factorial(len(ImageMR.get_image_GMRs()) - depth)


def test_get_composite_list_names():
    mr = MR(ImageMR.get_image_GMRs())

    depth = 4
    mr.update_composite(depth)

    assert mr.MR_list_names[0] == "Rotate by 5 degrees -> Rotate by -5 degrees -> Scale values by +10 -> Blur"
