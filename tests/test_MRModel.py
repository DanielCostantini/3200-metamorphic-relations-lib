import pytest
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
import keras.layers as layers

from metamorphic_relations import MR
from metamorphic_relations.Data import Data
from metamorphic_relations.ImageMR import ImageMR
from metamorphic_relations.MRModel import MRModel
from metamorphic_relations.Transform import Transform

MNIST = mnist.load_data()
data = Data(train_x=MNIST[0][0][:100], train_y=MNIST[0][1][:100], test_x=MNIST[1][0][:20], test_y=MNIST[1][1][:20],
            max_y=10)

model = Sequential()
model.add(Dense(128, input_shape=MNIST[0][0][0].flatten().shape))
model.add(layers.LeakyReLU())
model.add(Dense(256))
model.add(layers.LeakyReLU())
model.add(Dense(256))
model.add(layers.LeakyReLU())
model.add(Dense(10))
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

mrModel = MRModel(data, model, transform_x=lambda x: x.reshape((x.shape[0], -1)), GMRs=ImageMR.get_image_GMRs(),
                  DSMRs=[Transform(lambda x: ImageMR.flip_horizontal_transform(x), 0, 1)])


def test_MRModel_compare_MR_sets_counts_correct():
    results, models = mrModel.compare_MR_sets_counts()

    assert models[0] != models[1]


def test_MRModel_compare_MR_sets_counts_correct_compare_sets():
    results, models = mrModel.compare_MR_sets_counts(compare_sets=(True, False, False, True))

    assert len(models) == 2
    assert results.original_results is not None
    assert results.GMR_results is None
    assert results.DSMR_results is None
    assert results.all_MR_results is not None


def test_MRModel_compare_MR_sets_counts_incorrect_compare_sets():
    with pytest.raises(Exception):
        mrModel.compare_MR_sets_counts(compare_sets=(True, False))


def test_MRModel_compare_MR_sets_counts_incorrect_min_i():
    with pytest.raises(Exception):
        mrModel.compare_MR_sets_counts(min_i=-1)


def test_MRModel_compare_MR_sets_correct():
    results, models = mrModel.compare_MR_sets()

    assert models[0] != models[1]
    mrModel.model = models[0]
    assert results.original_results.test_f1 == mrModel.test_model()


def test_MRModel_compare_MR_sets_correct_compare_sets():
    results, models = mrModel.compare_MR_sets(compare_sets=(True, False, False, True))

    assert len(models) == 2
    assert results.original_results is not None
    assert results.GMR_results is None
    assert results.DSMR_results is None
    assert results.all_MR_results is not None


def test_MRModel_compare_MR_sets_incorrect_compare_sets():
    with pytest.raises(Exception):
        mrModel.compare_MR_sets(compare_sets=(True, False))


def test_MRModel_compare_MR_counts_correct():
    results, models = mrModel.compare_MR_counts()

    assert models[0] != models[1]


def test_MRModel_compare_MR_counts_incorrect_min_i():
    with pytest.raises(Exception):
        mrModel.compare_MR_counts(min_i=-1)


def test_MRModel_compare_MR_correct():
    results, models = mrModel.compare_MRs()

    assert models[0] != models[1]


def test_MRModel_get_results_correct_MR_obj_empty():

    info, _ = mrModel.get_results(MR_obj=MR([]))

    assert info.original_count[0] == info.actual_count[0]
    assert info.original_count[0] == len(mrModel.data.train_x)


def test_MRModel_get_results_correct_MR_obj_GMRs():

    info, _ = mrModel.get_results(MR_obj=mrModel.GMRs)

    assert info.original_count[0] != info.actual_count[0]


def test_MRModel_get_results_correct_MR_list_empty():
    info, _ = mrModel.get_results(MR_list=[])

    assert info.original_count[0] == info.actual_count[0]
    assert info.original_count[0] == len(mrModel.data.train_x)


def test_MRModel_get_results_correct_MR_list_GMRs():
    info, _ = mrModel.get_results(MR_list=mrModel.GMRs.MR_list[0])

    assert info.original_count[0] != info.actual_count[0]


def test_MRModel_get_results_correct_i_vals_not_None():

    i_vals = [10, 20]
    info, _ = mrModel.get_results(MR_obj=mrModel.GMRs, i_vals=i_vals)

    assert len(info.original_count) == len(i_vals)


def test_MRModel_get_results_incorrect_MR_obj_MR_list():

    with pytest.raises(Exception):
        mrModel.get_results(MR_obj=mrModel.GMRs, MR_list=mrModel.GMRs.MR_list)


def test_MRModel_get_results_incorrect_MR_obj_MR_list_None():

    with pytest.raises(Exception):
        mrModel.get_results(MR_obj=None, MR_list=None)


def test_MRModel_train_model_correct():

    f1 = mrModel.train_model(mrModel.data.train_x, mrModel.data.train_y)

    assert f1 is not None


def test_MRModel_train_model_incorrect_train_len():

    with pytest.raises(Exception):
        mrModel.train_model(mrModel.data.train_x, mrModel.data.test_y)


def test_MRModel_train_model_incorrect_k():

    with pytest.raises(Exception):
        mrModel.train_model(mrModel.data.train_x, mrModel.data.train_y, k=0)


def test_MRModel_test_model_correct():

    f1 = mrModel.train_model(mrModel.data.test_x, mrModel.data.test_y)

    assert f1 is not None


def test_MRModel_test_model_incorrect_test_len():

    with pytest.raises(Exception):
        mrModel.train_model(mrModel.data.test_x, mrModel.data.train_y)


def test_MRModel_concat_correct():

    x, y = MRModel.concat(mrModel.data.train_x, mrModel.data.train_y, mrModel.data.test_x, mrModel.data.test_y)

    assert len(x) == len(mrModel.data.train_x) + len(mrModel.data.test_x)
    assert len(y) == len(mrModel.data.train_y) + len(mrModel.data.test_y)


def test_MRModel_transform_data_correct():

    xs, ys = mrModel.transform_data(mrModel.data.train_x, mrModel.data.train_y)

    assert xs[0] == mrModel.data.train_x[0]
    assert ys[0] == MRModel.y_1D_to_2D(mrModel.data.train_y, mrModel.data.max_y)[0]


def test_MRModel_y_2D_to_1D_correct():

    new_y = MRModel.y_2D_to_1D(np.array([[0, 1], [1, 0]]))

    assert np.array_equal(new_y, np.array([1, 0]))


def test_MRModel_y_2D_to_1D_incorrect_y_shape():

    with pytest.raises(Exception):
        MRModel.y_2D_to_1D(np.array([0, 1]))


def test_MRModel_y_1D_to_2D_correct():

    new_y = MRModel.y_1D_to_2D(np.array([1, 0]), 2)

    assert np.array_equal(new_y, np.array([[0, 1], [1, 0]]))


def test_MRModel_y_1D_to_2D_incorrect_y_shape():

    with pytest.raises(Exception):
        MRModel.y_1D_to_2D(np.array([[0, 1], [1, 0]]), 2)


def test_MRModel_y_1D_to_2D_incorrect_max_y():

    with pytest.raises(Exception):
        MRModel.y_1D_to_2D(np.array([1, 0]), 1)
