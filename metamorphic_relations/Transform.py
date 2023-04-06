class Transform:
    """
    Creates an object to be used for transforms

    :param function func: the transformation function, it must take a numpy array to another numpy array of the same shape
    :param current: the label index of that the data originally has
    :param target: the label index of the data after the transform
    :param name: the name of the transform
    """

    def __init__(self, func, current: int, target: int, name: str = None):

        self.func = func
        self.current = current
        self.target = target

        if name is None:
            self.name = func.__str__()
        else:
            self.name = name

