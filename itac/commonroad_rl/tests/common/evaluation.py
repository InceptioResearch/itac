__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Helper module for test evaluation
"""
from contextlib import contextmanager


@contextmanager
def does_not_raise():
    """
    Function which can be used for expecting that no exception will be raised
    Usage:
    @pytest.mark.parametrize("expected_exception", does_not_raise())
    def some_function(expected_exception):
        with expected_exception:
            do_the_test()
    """
    yield


def deep_compare(o1: object, o2: object) -> bool:
    """
    Comapres all attributes of two objects

    :param o1: The first object
    :param o2: The second object
    :return: True if all attributes of the objects match
    """
    if o1 is None:
        return o1 == o2

    o1d = getattr(o1, "__dict__", None)
    o2d = getattr(o2, "__dict__", None)

    # if both are objects
    if o1d is not None and o2d is not None:
        # we will compare their dictionaries
        o1, o2 = o1.__dict__, o2.__dict__

    if o1 is not None and o2 is not None:
        # if both are dictionaries, we will compare each key
        if isinstance(o1, dict) and isinstance(o2, dict):
            for k in set().union(o1.keys(), o2.keys()):
                if k in o1 and k in o2:
                    if not deep_compare(o1[k], o2[k]):
                        return False
                else:
                    return False  # some key missing
            return True
    return o1 == o2
