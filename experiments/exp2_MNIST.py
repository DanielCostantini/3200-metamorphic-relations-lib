from keras import Sequential
from keras.layers import Dense
import keras.layers as layers
from keras.datasets import mnist
import matplotlib.pyplot as plt

from metamorphic_relations import MR
from metamorphic_relations.ImageMR import ImageMR
from metamorphic_relations.Data import Data
from metamorphic_relations.MRModel import MRModel
from metamorphic_relations.Results import Results


def get_MNIST_DSMRs():
    #     The list of domain specific metamorphic relations, each contains a transform function,
    #     a y value to check for, and the  new y value
    #     Each transform function must return the x value of the same shape

    DSMRs = []

    DSMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, angle=180), [0, 1, 6, 8, 9], [0, 1, 9, 8, 6], "Rotate 180 degrees")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_vertical_transform(x), [0, 1, 3, 8], name="Flip vertical")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_horizontal_transform(x), [0, 1, 8], name="Flip horizontal")

    return DSMRs


def get_MNIST_model(input_shape, output_shape):
    # The model is 4 layers of dense neurons connected via leaky ReLU
    # Leaky is used to avoid the diminishing ReLU problem
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


def graph_composite():

    c1 = Results.read_from_file("Output/MNIST_sets_results.txt")
    c2 = Results.read_from_file("Output/MNIST_sets_results_2.txt")
    c3 = Results.read_from_file("Output/MNIST_sets_results_3.txt")

    plt.title("MNIST Original Number of Data Points vs Train F1 with Composite MRs")
    plt.xlabel("Number of given Data Points of Original Set")
    plt.ylabel("Train Macro F1 Score")
    plt.xscale("log", base=2)
    plt.scatter(c1.original_results.original_count, c1.original_results.train_f1)
    plt.scatter(c1.all_MR_results.original_count, c1.all_MR_results.train_f1)
    plt.scatter(c2.all_MR_results.original_count, c2.all_MR_results.train_f1)
    plt.scatter(c3.all_MR_results.original_count, c3.all_MR_results.train_f1)
    plt.legend(["Unaltered Data", "Max Composite = 1", "Max Composite = 2", "Max Composite = 3"])
    plt.show()

    plt.title("MNIST Original Number of Data Points vs Test F1 with Composite MRs")
    plt.xlabel("Number of given Data Points of Original Set")
    plt.ylabel("Test Macro F1 Score")
    plt.xscale("log", base=2)
    plt.scatter(c1.original_results.original_count, c1.original_results.test_f1)
    plt.scatter(c1.all_MR_results.original_count, c1.all_MR_results.test_f1)
    plt.scatter(c2.all_MR_results.original_count, c2.all_MR_results.test_f1)
    plt.scatter(c3.all_MR_results.original_count, c3.all_MR_results.test_f1)
    plt.legend(["Unaltered Data", "Max Composite = 1", "Max Composite = 2", "Max Composite = 3"])
    plt.show()

    plt.title("MNIST Actual Number of Data Points vs Train F1 with Composite MRs")
    plt.xlabel("Number of given Data Points after MRs Applied")
    plt.ylabel("Train Macro F1 Score")
    plt.xscale("log", base=2)
    plt.scatter(c1.original_results.actual_count, c1.original_results.train_f1)
    plt.scatter(c1.all_MR_results.actual_count, c1.all_MR_results.train_f1)
    plt.scatter(c2.all_MR_results.actual_count, c2.all_MR_results.train_f1)
    plt.scatter(c3.all_MR_results.actual_count, c3.all_MR_results.train_f1)
    plt.legend(["Unaltered Data", "Max Composite = 1", "Max Composite = 2", "Max Composite = 3"])
    plt.show()

    plt.title("MNIST Actual Number of Data Points vs Test F1 with Composite MRs")
    plt.xlabel("Number of given Data Points after MRs Applied")
    plt.ylabel("Test Macro F1 Score")
    plt.xscale("log", base=2)
    plt.scatter(c1.original_results.actual_count, c1.original_results.test_f1)
    plt.scatter(c1.all_MR_results.actual_count, c1.all_MR_results.test_f1)
    plt.scatter(c2.all_MR_results.actual_count, c2.all_MR_results.test_f1)
    plt.scatter(c3.all_MR_results.actual_count, c3.all_MR_results.test_f1)
    plt.legend(["Unaltered Data", "Max Composite = 1", "Max Composite = 2", "Max Composite = 3"])
    plt.show()


MNIST = mnist.load_data()
data = Data(train_x=MNIST[0][0], train_y=MNIST[0][1], test_x=MNIST[1][0], test_y=MNIST[1][1], max_y=10)
MNIST_model = get_MNIST_model(input_shape=MNIST[0][0][0].flatten().shape, output_shape=data.max_y)

MR_model = MRModel(data=data, model=MNIST_model, transform_x=lambda x: x.reshape((x.shape[0], -1)), GMRs=ImageMR.get_image_GMRs(),
                   DSMRs=get_MNIST_DSMRs())


results, _ = MR_model.compare_MR_sets_counts()
results.write_to_file("Output/MNIST_sets_results.txt")
Results.read_from_file("Output/MNIST_sets_results.txt").graph_all("MNIST")

results, models = MR_model.compare_MR_sets(max_composite=2, compare_sets=(False, False, False, True))
results.write_to_file("Output/MNIST_sets_best_results.txt")

results, _ = MR_model.compare_MRs()
results.write_to_file("Output/MNIST_individual_best_results.txt")
Results.read_from_file("Output/MNIST_individual_best_results.txt").print_individual()

results, _ = MR_model.compare_MR_sets_counts(max_composite=2)
results.write_to_file("Output/MNIST_sets_results_2.txt")
results, _ = MR_model.compare_MR_sets_counts(max_composite=3)
results.write_to_file("Output/MNIST_sets_results_3.txt")
graph_composite()
