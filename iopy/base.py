
def _check_extension(ext):
    """ Returns the extension in the form ".ext"
    Adds the dot if needed

    Parameters
    ----------
    ext: string
        '.ext' or 'ext'

    Returns
    -------
    string, extension, adds the "." if needed

    Example
    -------
    if ext == 'txt' or ext == '.txt', '.txt' is returned in both cases
    """
    if ext[0] == '.':
        return ext
    else:
        return '.'+ext
        return '.'+ext


class Bunch(dict):
    """A dictionary exposing its keys as attributes
    """
    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __setitem__(self, key, value):
        return super(Bunch, self).__setitem__(key, value)

    def __getitem__(self, name):
        item = super(Bunch, self).__getitem__(name)
        return Bunch(item) if type(item) == dict else item

    def __delitem__(self, name):
        return super(Bunch, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self):
        new_bunch = Bunch(self)
        return new_bunch

    def __dir__(self):
        return self.keys()
