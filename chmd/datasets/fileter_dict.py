"""Filter dict dataset."""


class FilterDict(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, in_data):
        return {key: in_data[key] for key in self.keys}