"""Implementation of DeviceStruct."""
import numpy as np
from chainer import Variable
from chainer.device_resident import DeviceResident


class DeviceStruct(DeviceResident):
    """Dictionary that can execute to_device or other methods.

    Example:
        class Batch(DeviceDict):
            positions: Variable
            adjacent1: np.ndarray

    """

    def __init__(self, **kwargs):
        """Initialize dictionary using kwargs."""
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __setattr__(self, name, value):
        """Raise KeyError if name not in __annotations__."""
        if not (name in self.__annotations__ or
                name == '_overridden_to_methods'):
            raise KeyError(name)
        super().__setattr__(name, value)

    def keys(self):
        """Interface like dict."""
        for key in self.__dict__:
            if key in self.__annotations__:
                yield key

    def values(self):
        """Interface like dict."""
        for key, val in self.__dict__.items():
            if key in self.__annotations__:
                yield val

    def items(self):
        """Interface like dict."""
        for key, val in self.__dict__.items():
            if key in self.__annotations__:
                yield key, val

    def device_resident_accept(self, visitor):
        """Neccesarry for to_device, xp, and a lot of other methods."""
        super().device_resident_accept(visitor)
        for key, val in self.__annotations__.items():
            if val is np.ndarray:
                setattr(self, key, visitor.visit_array(getattr(self, key)))
            if val is Variable:
                visitor.visit_variable(getattr(self, key))

    def __repr__(self):
        """Print like dict."""
        return repr(dict(self.items()))
