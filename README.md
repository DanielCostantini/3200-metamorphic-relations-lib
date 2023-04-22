## Installation
    pip install metamorphic_relations

## Get Started - MNIST Example
For more information on all classes and functions see https://3200-metamorphic-relations-lib.readthedocs.io/

### Create Data Object
Stores the data that the metamorphic relations (MRs) are used on

    from keras.datasets import mnist
    from metamorphic_relations.Data import Data

    MNIST = mnist.load_data()
    data = Data(train_x=MNIST[0][0], train_y=MNIST[0][1], 
            test_x=MNIST[1][0], test_y=MNIST[1][1], max_y=10)

### Create Keras Model
A deep learning keras model to be trained for MNIST image classification

    from keras import Sequential
    from keras.layers import Dense
    import keras.layers as layers

    MNIST_model = Sequential()
    MNIST_model.add(Dense(128, input_shape=MNIST[0][0][0].flatten().shape))
    MNIST_model.add(layers.LeakyReLU())
    MNIST_model.add(Dense(256))
    MNIST_model.add(layers.LeakyReLU())
    MNIST_model.add(Dense(256))
    MNIST_model.add(layers.LeakyReLU())
    MNIST_model.add(Dense(data.max_y))
    MNIST_model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])


### Create Metamorphic Relations (MRs)
A list of domain specific MRs (DSMRs) is created for the given dataset using the transforms provided by ImageMR

    from metamorphic_relations.MR import MR
    from metamorphic_relations.ImageMR import ImageMR

    DSMRs = []
    DSMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, angle=180), [0, 1, 6, 8, 9], [0, 1, 9, 8, 6], "Rotate 180 degrees")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_vertical_transform(x), [0, 1, 3, 8], name="Flip vertical")
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_horizontal_transform(x), [0, 1, 8], name="Flip horizontal")

The list of Generic MRs (GMRs) for images is provided by ImageMR

    GMRs = ImageMR.get_image_GMRs()    

### Create MRModel
The object from which the MRs can be applied to the data and the model can be trained

    from metamorphic_relations.MRModel import MRModel

    MR_model = MRModel(data=data, model=MNIST_model, 
        transform_x=lambda x: x.reshape((x.shape[0], -1)), 
        GMRs=GMRs, DSMRs=DSMRs)
    

### Use MRModel to Create Additional Data
The list of MRs are converted to MR objects which can be used to create new data

    MR_model.DSMRs.perform_MRs_tree(data.train_x, data.train_y, data.max_y)

The maximum number of composite MRs can be increased before perform_MRs_tree() to create more data

    MR_model.DSMRs.update_composite(max_composite=2)


### Use MRModel to Train Keras Model
Trains the given model and returns F1 scores and models for various configurations

    from metamorphic_relations.Results import Results

MR_model.compare_MRs() compares each MR individually using all the training data

    results, models = MR_model.compare_MRs()

MR_model.compare_MR_sets() compares each set of MR (No MRs, GMRs, DSMRs, GMRs + DSMRs) using all the training data

    results, models = MR_model.compare_MR_sets()

MR_model.compare_MR_sets() compares each set of MR (No MRs, GMRs, DSMRs, GMRs + DSMRs) using increasing proportions of the training data

    results, models = MR_model.compare_MR_sets_counts()

### Save and Load Results
Result objects can be written to and read from text files
    
    results, _ = MR_model.compare_MR_sets_counts()
    results.write_to_file("Output/MNIST_sets_results.txt")
    Results.read_from_file("Output/MNIST_sets_results.txt")

### Present Results
Using compare_MRs() gives individual results for each MR, these can be tabulated with initial number of training instances, actual number used, train and test F1 score

    results, _ = MR_model.compare_MRs()
    results.write_to_file("Output/MNIST_individual_best_results.txt")
    Results.read_from_file("Output/MNIST_individual_best_results.txt").print_individual()

The results for each set from compare_MR_sets_counts() can be graphed with graph_all() 

    results, _ = MR_model.compare_MR_sets_counts()
    results.write_to_file("Output/MNIST_sets_results.txt")
    Results.read_from_file("Output/MNIST_sets_results.txt").graph_all("MNIST")
