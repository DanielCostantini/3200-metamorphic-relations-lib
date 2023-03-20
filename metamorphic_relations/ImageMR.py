import numpy as np
from scipy import ndimage

from metamorphic_relations import MR
from metamorphic_relations.Transform import Transform


class ImageMR(MR):

    @staticmethod
    def get_image_GMRs() -> list[Transform]:
        """
        Gives a list of generic metamorphic relations (GMRs) for images

        :return: list of Transforms
        """

        GMRs = []

        GMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, 5))
        GMRs += MR.for_all_labels(lambda x: ImageMR.rotate_transform(x, -5))
        GMRs += MR.for_all_labels(lambda x: MR.scale_values_transform(x, lambda y: y + 10))
        GMRs += MR.for_all_labels(lambda x: ImageMR.blur_transform(x, 1))

        return GMRs

    @staticmethod
    def rotate_transform(x: np.array, angle: int) -> np.array:
        """
        Rotates the input by an angle

        :param x: input data in the form of a numpy array
        :param angle: an angle in degrees
        :return: the transformed data
        """

        if angle == 180:
            flat_x = x.flatten()
            rotate_x = np.flip(flat_x)
            x = rotate_x.reshape(x.shape)
        else:
            x = ndimage.rotate(x, angle, reshape=False)

        return x

    @staticmethod
    def flip_vertical_transform(x: np.array) -> np.array:
        """
        Flips the input vertically

        :param x: input data in the form of a numpy array
        :return: the transformed data
        """

        return np.flipud(x)

    @staticmethod
    def flip_horizontal_transform(x: np.array) -> np.array:
        """
        Flips the input horizontally

        :param x: input data in the form of a numpy array
        :return: the transformed data
        """

        return np.fliplr(x)

    @staticmethod
    def blur_transform(x: np.array, sigma: float) -> np.array:
        """
        Blurs the input

        :param x: input data in the form of a numpy array
        :param sigma: a blurring factor (the larger sigma the more blurred the result)
        :return: the transformed data
        """

        return ndimage.gaussian_filter(x, sigma, mode='constant')
