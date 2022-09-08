__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Test configuration script, contains the hooks of the pytest framework
"""

import shutil
from typing import Dict

from pytest import Function

from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import *

# Dictionary contains the items failed because of syntax error
_test_failed_syntax_error: Dict[Function, str] = {}


def pytest_addoption(parser, pluginmanager):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_addoption
    """
    parser.addoption(
        "--scope",
        type=RunScope,
        nargs="+",
        default=list(RunScope),
        help="(tuple) The scope of the tests to be tested",
        choices=list(RunScope),
    )

    parser.addoption(
        "--type",
        type=RunType,
        nargs="+",
        default=list(RunType),
        help="(tuple) The scope of the tests to be tested",
        choices=list(RunType),
    )


def pytest_configure(config):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_configure
    You can modify the configuration values here
    e.g: config.addinivalue_line("markers", "env(name): mark test to run only on named environment")
    """
    pass


def pytest_collection_modifyitems(items, config):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_collection_modifyitems
    """
    # Select items
    selected_items = []
    deselected_items = []

    scopes = config.option.scope
    types = config.option.type
    for item in items:
        # The scope and type must be specified
        scope_marker = item.get_closest_marker("scope")
        type_marker = item.get_closest_marker("type")
        if scope_marker is None:
            _test_failed_syntax_error[
                item
            ] = "The scope of the test has not been defined"
            selected_items.append(item)
            continue
        if type_marker is None:
            _test_failed_syntax_error[
                item
            ] = "The type of the test has not been defined"
            selected_items.append(item)
            continue

        # Select the tests
        if scope_marker.args[0] in scopes and type_marker.args[0] in types:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    # Apply the selection
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


def pytest_runtest_setup(item):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_runtest_setup
    """
    item_output_path = output_root(item.fspath.purebasename)
    if os.path.exists(item_output_path) and os.path.isdir(item_output_path):
        shutil.rmtree(item_output_path)


def pytest_runtest_call(item):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_runtest_call
    """
    print(item)
    if item in _test_failed_syntax_error:
        pytest.xfail(_test_failed_syntax_error[item])


def pytest_runtest_teardown(item):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_runtest_teardown
    """
    pass


def pytest_runtest_makereport(item, call):
    """
    See https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_runtest_makereport
    """
    pass
