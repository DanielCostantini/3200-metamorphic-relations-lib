import pytest
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
import keras.layers as layers

from metamorphic_relations.Data import Data
from metamorphic_relations.ImageMR import ImageMR
from metamorphic_relations.MRModel import MRModel
from metamorphic_relations.Transform import Transform

MNIST = mnist.load_data()
data = Data(train_x=MNIST[0][0][:100], train_y=MNIST[0][1][:100], test_x=MNIST[1][0][:100], test_y=MNIST[1][1][:100], max_y=10)

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
