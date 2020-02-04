# Copied from python 3 documentation
# https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
from collections import OrderedDict


class LRU(OrderedDict):
    """Limit size, evicting the least recently looked-up key when full"""
    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
