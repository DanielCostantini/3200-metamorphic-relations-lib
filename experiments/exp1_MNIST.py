from keras import Sequential
from keras.layers import Dense
import keras.layers as layers
from keras.datasets import mnist

from metamorphic_relations import MR
from metamorphic_relations.ImageMR import ImageMR
from metamorphic_relations.Data import Data
from metamorphic_relations.MRModel import MRModel
from metamorphic_relations.Results import Results
from metamorphic_relations.Transform import Transform


def get_MNIST_DSMRs():
    #     The list of domain specific metamorphic relations, each contains a transform function,
    #     a y value to check for, and the  new y value
    #     Each transform function must return the x value of the same shape
    # DSMRs = [Transform(lambda x: ImageMR.rotate_transform(x, angle=180), 0, 0, "Rotate 180 degrees"),
    #          Transform(lambda x: ImageMR.flip_vertical_transform(x), 0, 0, "Flip vertical"),
    #          Transform(lambda x: ImageMR.flip_horizontal_transform(x), 0, 0, "Flip horizontal"),
    #          Transform(lambda x: ImageMR.rotate_transform(x, angle=180), 1, 1, "Rotate 180 degrees"),
    #          Transform(lambda x: ImageMR.flip_vertical_transform(x), 1, 1, "Flip vertical"),
    #          Transform(lambda x: ImageMR.flip_horizontal_transform(x), 1, 1, "Flip horizontal"),
    #          Transform(lambda x: ImageMR.flip_vertical_transform(x), 3, 3, "Flip vertical"),
    #          Transform(lambda x: ImageMR.rotate_transform(x, angle=180), 6, 9, "Rotate 180 degrees"),
    #          Transform(lambda x: ImageMR.rotate_transform(x, angle=180), 8, 8, "Rotate 180 degrees"),
    #          Transform(lambda x: ImageMR.flip_vertical_transform(x), 8, 8, "Flip vertical"),
    #          Transform(lambda x: ImageMR.flip_horizontal_transform(x), 8, 8, "Flip horizontal"),
    #          Transform(lambda x: ImageMR.rotate_transform(x, angle=180), 9, 6, "Rotate 180 degrees")]

    DSMRs = []

    DSMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, angle=180), [0, 1, 6, 8, 9], [0, 1, 9, 8, 6], "Rotate 180 degrees")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_vertical_transform(x), [0, 1, 3, 8], name="Flip vertical")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_horizontal_transform(x), [0, 1, 8], name="Flip horizontal")

    return DSMRs


def get_MNIST_model(input_shape, output_shape):
    #     The model is 4 layers of dense neurons connected via leaky ReLU
    #     Leaky is used to avoid the diminishing ReLU problem
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(Dense(256))
    model.add(layers.LeakyReLU())
    model.add(Dense(256))
    model.add(layers.LeakyReLU())
    model.add(Dense(output_shape))

    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return model


MNIST = mnist.load_data()
data = Data(train_x=MNIST[0][0], train_y=MNIST[0][1], test_x=MNIST[1][0], test_y=MNIST[1][1], max_y=10)
MNIST_model = get_MNIST_model(input_shape=MNIST[0][0][0].flatten().shape, output_shape=data.max_y)

MR_model = MRModel(data=data, model=MNIST_model, transform_x=lambda x: x.reshape((x.shape[0], -1)), GMRs=ImageMR.get_image_GMRs(),
                   DSMRs=get_MNIST_DSMRs())


results = MR_model.compare_MR_sets_counts()
results.write_to_file("Output/MNIST_sets_results.txt")
# results = MR_model.compare_MR_sets()
# results.write_to_file("Output/MNIST_sets_best_results.txt")
# results = MR_model.compare_MR_counts()
# results.write_to_file("Output/MNIST_individual_results.txt")
results = MR_model.compare_MRs()
results.write_to_file("Output/MNIST_individual_best_results.txt")
# results = Results.read_from_file("Output/MNIST_individual_best_results.txt")
# results.graph()
# results.graph(original_counts=False)
