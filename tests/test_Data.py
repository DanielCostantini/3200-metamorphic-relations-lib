import pytest
import numpy as np

from metamorphic_relations.Data import Data

data = Data(np.array([1, 2, 3]), np.array([0, 1, 0]), np.array([4, 5, 6]), np.array([1, 1, 0]), 1)


def test_data_update_train_correct():
    data.update_train(np.array([7, 8, 9, 10]), np.array([0, 1, 1, 1]))
    assert np.array_equal(data.train_x, [7, 8, 9, 10])
    assert np.array_equal(data.train_y, [0, 1, 1, 1])


def test_data_update_train_incorrect_train_length():

    with pytest.raises(Exception):
        data.update_train(np.array([7, 8, 9]), np.array([0]))


def test_data_shuffle_train_correct():

    org_train_x = data.train_x

    data.shuffle_train()

    assert not np.array_equal(org_train_x, data.train_x)
    assert org_train_x.shape == data.train_x.shape


def test_data_update_test_correct():
    data.update_test(np.array([7, 8, 9]), np.array([0, 1, 1]))
    assert np.array_equal(data.test_x, [7, 8, 9])
    assert np.array_equal(data.test_y, [0, 1, 1])


def test_data_update_test_incorrect_test_length():

    with pytest.raises(Exception):
        data.update_test(np.array([7, 8, 9]), np.array([]))


def test_data_concat_lists_correct():

    tuple1 = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0, 1, 0]))
    tuple2 = (np.array([[4, 4], [5, 5], [6, 6]]), np.array([1, 1, 0]))
    tuple3 = (np.array([[7, 7], [8, 8], [9, 9]]), np.array([0, 1, 1]))

    xs, ys = Data.concat_lists([tuple1, tuple2, tuple3])

    assert np.array_equal(xs, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    assert np.array_equal(ys, [0, 1, 0, 1, 1, 0, 0, 1, 1])


def test_data_concat_lists_correct_short_train_3():

    tuple1 = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0, 1, 0]))
    tuple2 = (np.array([[4, 4], [5, 5], [6, 6]]), np.array([1, 1, 0]))
    tuple3 = (np.array([[7, 7]]), np.array([0]))

    xs, ys = Data.concat_lists([tuple1, tuple2, tuple3])

    assert np.array_equal(xs, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    assert np.array_equal(ys, [0, 1, 0, 1, 1, 0, 0])


def test_data_concat_lists_incorrect_tuple_length():

    tuple1 = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0, 1, 0]), np.array([]))
    tuple2 = (np.array([[4, 4], [5, 5], [6, 6]]), np.array([1, 1, 0]))
    tuple3 = (np.array([[7, 7], [8, 8], [9, 9]]), np.array([0, 1, 1]))

    with pytest.raises(Exception):
        Data.concat_lists([tuple1, tuple2, tuple3])


def test_data_concat_lists_incorrect_pair_length():

    tuple1 = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0, 1, 0, 1, 0]))
    tuple2 = (np.array([[4, 4], [5, 5], [6, 6]]), np.array([1, 1, 0]))
    tuple3 = (np.array([[7, 7], [8, 8], [9, 9]]), np.array([0, 1, 1]))

    with pytest.raises(Exception):
        Data.concat_lists([tuple1, tuple2, tuple3])


y = np.array([1, 2, 0, 1, 1, 0, 2, 1, 1])


def test_data_group_by_label_correct():

    assert Data.group_by_label(y, 3) == [[2, 5], [0, 3, 4, 7, 8], [1, 6]]


def test_data_group_by_label_correct2():

    assert Data.group_by_label(y, 6) == [[2, 5], [0, 3, 4, 7, 8], [1, 6], [], [], []]


def test_data_group_by_label_incorrect_max_y():

    with pytest.raises(Exception):
        Data.group_by_label(y, 1)


def test_data_group_by_label_incorrect_ys():
    with pytest.raises(Exception):
        Data.group_by_label(np.array([0, -2, 2, 1, -1, 0]), 2)


def test_data_get_train_subset_correct():

    data.update_train(np.array([7, 8, 9, 10]), np.array([0, 1, 1, 1]))
    assert np.array_equal(data.get_train_subset()[0], [7, 8, 9, 10])
