import pytest

from metamorphic_relations import MR, Transform, ImageMR


def test_get_composite_1():

    assert len(MR(ImageMR.get_image_GMRs(), 1).MRs) == 4


def test_get_composite_2():

    assert len(MR(ImageMR.get_image_GMRs(), 2).MRs[0][1]) == 3


def test_for_all_labels_generic():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10))[0].target == -1


def test_for_all_labels_stay_same():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5])[0].target == 1


def test_for_all_labels_change():

    assert MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10), [1, 5], [10, 6])[1].current == 5

