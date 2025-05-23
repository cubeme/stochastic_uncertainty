"""Module with Flags class to mimic the behavior of absl.flags."""


class Flags(dict):
    """
    Mimic the behavior of absl.flags.
    This class is used to handle command line arguments and configuration settings.
    """

    def __init__(self, *args, **kwargs):
        super(Flags, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Flags, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Flags, self).__delitem__(key)
        del self.__dict__[key]
