import pytest
import numpy as np

from metamorphic_relations.ImageMR import ImageMR


def test_ImageMR_get_image_GMRs_correct():
    assert len(ImageMR.get_image_GMRs()) == 4


x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_ImageMR_rotate_transform_correct_0():
    assert np.array_equal(ImageMR.rotate_transform(x, 0), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_ImageMR_rotate_transform_correct_90():
    assert np.array_equal(ImageMR.rotate_transform(x, 90), np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]]))


def test_ImageMR_rotate_transform_correct_180():
    assert np.array_equal(ImageMR.rotate_transform(x, 180), np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]))


def test_ImageMR_flip_vertical_transform_correct_empty_x():
    assert np.array_equal(ImageMR.flip_vertical_transform(np.array([])), np.array([]))


def test_ImageMR_flip_vertical_transform_correct_2D():
    assert np.array_equal(ImageMR.flip_vertical_transform(x), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]]))


def test_ImageMR_flip_vertical_transform_correct_1D():
    x = np.array([1, 2, 3])
    assert np.array_equal(ImageMR.flip_vertical_transform(x), np.array([3, 2, 1]))


def test_ImageMR_flip_vertical_transform_correct_3D():
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                  [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    assert np.array_equal(ImageMR.flip_vertical_transform(x),
                          np.array([[[19, 20, 21], [22, 23, 24], [25, 26, 27]],
                                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                                    [[1,  2,  3], [4,  5,  6], [7,  8,  9]]]))


def test_ImageMR_flip_horizontal_transform_correct_2D():
    assert np.array_equal(ImageMR.flip_horizontal_transform(x), np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]]))


def test_ImageMR_flip_horizontal_transform_correct_3D():
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                  [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    assert np.array_equal(ImageMR.flip_horizontal_transform(x),
                          np.array([[[7,  8,  9], [4,  5,  6], [1,  2,  3]],
                                    [[16, 17, 18], [13, 14, 15], [10, 11, 12]],
                                    [[25, 26, 27], [22, 23, 24], [19, 20, 21]]]))


def test_ImageMR_blur_transform_correct_sigma_0():
    assert np.array_equal(ImageMR.blur_transform(x, 0), x)


def test_ImageMR_blur_transform_correct_sigma_neg():
    assert np.array_equal(ImageMR.blur_transform(x, -1), x)


def test_ImageMR_blur_transform_correct_x_0():
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(ImageMR.blur_transform(x, 1), x)
