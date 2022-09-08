__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Helper module for the marking of tests
"""
import threading
from enum import Enum
from functools import wraps

import pytest

lock_obj = threading.Lock()


class RunScope(Enum):
    """Enum for the scope of the test methods"""

    UNIT_TEST = "unit"
    MODULE_TEST = "module"
    INTEGRATION_TEST = "integration"


class RunType(Enum):
    """Enum for the type of the test methods"""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"


def unit_test(*args, **kwargs):
    """Unit test marker decorator"""
    return pytest.mark.scope.with_args(RunScope.UNIT_TEST)(*args, **kwargs)


def module_test(*args, **kwargs):
    """Module test marker decorator"""
    return pytest.mark.scope.with_args(RunScope.MODULE_TEST)(*args, **kwargs)


def integration_test(*args, **kwargs):
    """Integration test marker decorator"""
    return pytest.mark.scope.with_args(RunScope.INTEGRATION_TEST)(*args, **kwargs)


def functional(*args, **kwargs):
    """Functional test marker decorator"""
    return pytest.mark.type.with_args(RunType.FUNCTIONAL)(*args, **kwargs)


def nonfunctional(*args, **kwargs):
    """Non-functional test marker decorator"""
    return pytest.mark.type.with_args(RunType.NON_FUNCTIONAL)(*args, **kwargs)


def slow(*args, **kwargs):
    """Slow test marker decorator"""
    return pytest.mark.slow(*args, **kwargs)


def serial(func):
    """Serial test marker decorator"""
    serial_func = pytest.mark.serial(func)

    @wraps(serial_func)
    def inner_func(*args, **kwargs):
        lock_obj.acquire()
        try:
            result = serial_func(*args, **kwargs)
        finally:
            lock_obj.release()
        return result

    return inner_func
