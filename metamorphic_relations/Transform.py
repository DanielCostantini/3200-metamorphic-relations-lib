class Transform:

    def __init__(self, func, current: int, target: int):
        """
        Creates an object to be used for transforms

        :param func: the transformation function, it must take a numpy array to another numpy array of the same shape
        :param current: the label index of that the data originally has
        :param target: the label index of the data after the transform
        """

        self.func = func
        self.current = current
        self.target = target

