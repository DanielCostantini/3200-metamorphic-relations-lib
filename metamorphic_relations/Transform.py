class Transform:
    """
    Creates an object to be used for transforms

    :param function func: the transformation function, it must take a numpy array to another numpy array of the same shape
    :param current: the label index of that the data originally has
    :param target: the label index of the data after the transform
    """

    def __init__(self, func, current: int, target: int):

        self.func = func
        self.current = current
        self.target = target

