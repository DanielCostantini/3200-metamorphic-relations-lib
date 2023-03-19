import pytest

from metamorphic_relations import ImageMR


def test_get_image_GMRs():
    assert len(ImageMR.get_image_GMRs()) == 4