from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from itertools import chain
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import random
from operator import itemgetter

from metamorphic_relations.Data import Data
from metamorphic_relations.ImageMR import ImageMR
from metamorphic_relations.MR import MR
from metamorphic_relations.MRModel import MRModel
from metamorphic_relations.Results import Results


def get_road_signs_DSMRs():
    DSMRs = []

    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_horizontal_transform(x),
                               [11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_horizontal_transform(x),
                               [19, 20, 33, 34, 36, 37, 38, 39], [20, 19, 34, 33, 37, 36, 39, 38])
    DSMRs += MR.for_all_labels(lambda x: remove_center_circle_transform(x),
                               list(chain(range(6), range(7, 11), [16])), [15] * 11)
    DSMRs += MR.for_all_labels(lambda x: remove_center_triangle_transform(x),
                               list(chain([11], range(19, 32))), [18] * 14)
    DSMRs += MR.for_all_labels(lambda x: change_circle_background(x), list(chain(range(6), range(7, 11), [15, 16])))
    DSMRs += MR.for_all_labels(lambda x: change_triangle_background(x), list(chain([11], range(18, 32))))
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_vertical_transform(x), [12])
    DSMRs += MR.for_all_labels(lambda x: ImageMR.flip_vertical_transform(x), [13, 18], [18, 13])
    DSMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, 120), [13, 15, 18, 30, 40])
    DSMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, 240), [13, 15, 18, 30, 40])

    return DSMRs


def get_road_signs_model(input_shape, output_shape):
    model = Sequential()

    # FROM https://www.kaggle.com/code/berkaylhan/gtsrb-image-classification-with-cnn#Creating-and-Compiling-the-Model
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add((Conv2D(filters=32, kernel_size=(5, 5), activation="relu")))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add((Conv2D(filters=64, kernel_size=(3, 3), activation="relu")))
    model.add((MaxPool2D(pool_size=(2, 2))))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(rate=0.40))
    model.add(Dense(output_shape, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def remove_center_circle_transform(x):
    _, inner_circle = find_circles(x)

    if inner_circle is not None:
        (i_x, i_y, i_r) = inner_circle

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Original", brg_x)
        #         cv2.waitKey(0)

        #         Fills the inner circle with white
        x = cv2.circle(x, (i_x, i_y), i_r, (220, 220, 220), -1)

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Filled inner", brg_x)
        #         cv2.waitKey(0)

        return x

    else:
        return None


def change_circle_background(x):
    outer_circle, _ = find_circles(x)

    if outer_circle is not None:
        (o_x, o_y, o_r) = outer_circle

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Original", brg_x)
        #         cv2.waitKey(0)

        #         Fills outside the outer circle with black
        x = cv2.circle(x, (o_x, o_y), o_r + 20, (0, 0, 0), 40)

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Removed outer", brg_x)
        #         cv2.waitKey(0)

        #         Fills inside the outer circle with black in the background image
        bg = pick_random_background()
        bg = cv2.circle(bg, (o_x, o_y), o_r, (0, 0, 0), -1)

        #         brg_x = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Removed background inner", brg_x)
        #         cv2.waitKey(0)

        #        Combines the foreground and background
        x += bg

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Changed outer", brg_x)
        #         cv2.waitKey(0)

        return x

    else:
        return None


def pick_random_background():
    index = random.randint(0, 1)
    path = "Input/road_sign_backgrounds/rsb" + str(index) + ".jpg"

    image = Image.open(path)
    image = image.resize((64, 64))
    x = np.array(image)
    image.close()

    return x


def find_circles(x):
    #     https://www.tutorialspoint.com/find-circles-in-an-image-using-opencv-in-python#
    dp = 1
    minDist = 1
    canny = 30
    min_acc = 50
    min_r = int(x.shape[0] / 4)

    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist, param1=canny, param2=min_acc, minRadius=min_r,
                               maxRadius=0)

    outer_circle, inner_circle = None, None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        outer_circle = max(circles, key=itemgetter(2))
        (o_x, o_y, o_r) = outer_circle

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist, param1=canny, param2=min_acc, minRadius=min_r,
                                   maxRadius=o_r - 5)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            inner_circle = max(circles, key=itemgetter(2))
            (i_x, i_y, i_r) = inner_circle

    #     Outer - Green, Inner - Blue
    #     cv2.circle(x, (o_x, o_y), o_r, (0, 255, 0), 2)
    #     cv2.circle(x, (i_x, i_y), i_r, (0, 0, 255), 2)
    #     brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("Circles", brg_x)
    #     cv2.waitKey(0)

    return outer_circle, inner_circle


def remove_center_triangle_transform(x):
    _, inner_triangle = find_triangles(x)

    if inner_triangle is not None:

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Original", brg_x)
        #         cv2.waitKey(0)

        #         Fills the inner triangle with white
        img = cv2.drawContours(x, [inner_triangle], -1, (220, 220, 220), -1)

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Filled inner", brg_x)
        #         cv2.waitKey(0)

        return x

    else:
        return None


def change_triangle_background(x):
    outer_triangle, _ = find_triangles(x)

    if outer_triangle is not None:

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Original", brg_x)
        #         cv2.waitKey(0)

        #         Fills outside the outer triangle with black
        mask = np.full(x.shape, 0, dtype="uint8")
        mask = cv2.drawContours(mask, [outer_triangle], -1, (1, 1, 1), -1)

        x *= mask

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Removed outer", brg_x)
        #         cv2.waitKey(0)

        #         Fills inside the outer triangle with black in the background image
        bg = pick_random_background()
        bg = cv2.drawContours(bg, [outer_triangle], -1, (0, 0, 0), -1)

        #         brg_x = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Removed background inner", brg_x)
        #         cv2.waitKey(0)

        #        Combines the foreground and background
        x += bg

        #         brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #         cv2.imshow("Changed outer", brg_x)
        #         cv2.waitKey(0)

        return x

    else:
        return None


def find_triangles(img):
    #     https://www.tutorialspoint.com/how-to-detect-a-triangle-in-an-image-using-opencv-python#:~:text=To%20detect%20a%20triangle%20in%20an%20image%2C%20we%20first%20detect,set%20it%20as%20a%20triangle.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding to convert the grayscale image to a binary image

    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(dilation, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    #     cv2.imshow("Shapes", thresh)
    #     cv2.waitKey(0)

    # find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    triangles = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

        if len(approx) == 3:
            triangles.append(approx)

    if len(triangles) == 0:
        return None, None

    index_big_triangle = np.array(
        [(t[0][0][0] - t[1][0][0]) ** 2 + (t[0][0][1] - t[1][0][1]) ** 2 for t in triangles]).argmax()
    t = triangles[index_big_triangle]

    if cv2.arcLength(t, True) < 100:
        return None, None

    #     outer_tri, inner_tri = get_inner_outer_tri(t, 5)

    #     print("Coords", t)
    #     print("Outer", outer_tri)
    #     # Outer - Green, Inner - Blue
    #     img = cv2.drawContours(img, [outer_tri], -1, (0,255,0), 1)
    #     img = cv2.drawContours(img, [inner_tri], -1, (0,0,255), 1)
    #     brg_x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("Triangle", brg_x)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return get_inner_outer_tri(t, 7)


def get_inner_outer_tri(t, width):
    diag = (width ** 2 / 2.0) ** 0.5

    index_top = np.array([v[0][1] for v in t]).argmin()
    index_left = np.array([v[0][0] for v in t]).argmin()
    index_right = np.array([v[0][0] for v in t]).argmax()

    outer = t.copy()
    outer[index_top][0][1] -= width
    outer[index_left][0][0] -= diag
    outer[index_left][0][1] += diag
    outer[index_right][0][0] += diag
    outer[index_right][0][1] += diag

    inner = t.copy()
    inner[index_top][0][1] += width
    inner[index_left][0][0] += diag
    inner[index_left][0][1] -= diag
    inner[index_right][0][0] -= diag
    inner[index_right][0][1] -= diag

    return outer, inner


def get_images(path, image_paths):
    x = []

    for image_path in image_paths:
        image = Image.open(path + image_path)
        image = image.resize((64, 64))
        x.append(np.array(image))
        image.close()

    return np.array(x)


def load_image_road_signs(path, path_func):
    df = pd.read_excel('GTSRB/' + path + '.xlsx')
    df = df.dropna()

    paths = [path for path in df["Path"] if path_func(path)]

    x = get_images('GTSRB/', paths)

    y = np.array([df["ClassId"][i] for i in range(len(df["ClassId"])) if path_func(df["Path"][i])])

    return (x, y)


def load_road_signs():
    train = load_image_road_signs("Train", lambda x: '00029.png' in x)
    test = load_image_road_signs("Test", lambda x: True)

    np.save("input/road_sign_train_x", train[0])
    np.save("input/road_sign_train_y", train[1])

    np.save("input/road_sign_test_x", test[0])
    np.save("input/road_sign_test_y", test[1])

    return Data(train_x=train[0], train_y=train[1], test_x=test[0], test_y=test[1], max_y=43)


def read_road_sign_data():

    num_test = 1000

    return Data(train_x=np.load("input/road_sign_train_x.npy"),
                train_y=np.load("input/road_sign_train_y.npy"),
                test_x=np.load("input/road_sign_test_x.npy")[:num_test],
                test_y=np.load("input/road_sign_test_y.npy")[:num_test], max_y=43)


# data = load_road_signs()
data = read_road_sign_data()

road_signs_model = get_road_signs_model(input_shape=data.train_x[0].shape, output_shape=data.max_y)

MR_model = MRModel(data=data, model=road_signs_model, GMRs=ImageMR.get_image_GMRs(), DSMRs=get_road_signs_DSMRs())

results = MR_model.compare_MR_sets()
results.write_to_file("Output/GTSRB_results.txt")
results = Results.read_from_file("Output/GTSRB_results.txt")
results.graph()
results.graph(original_counts=False)
