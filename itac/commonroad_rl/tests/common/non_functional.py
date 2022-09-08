__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Helper module for non-functional tests
"""
import inspect


def function_to_string(function):
    class FStringReEvaluate:
        def __init__(self, payload):
            self.payload = payload

        def __str__(self):
            vars = inspect.currentframe().f_back.f_back.f_globals.copy()
            vars.update(inspect.currentframe().f_back.f_back.f_locals)
            return self.payload.format(**vars)

    source_code, _ = inspect.getsourcelines(function)
    source_body = source_code[1:]
    source_body_unindented = [line[8:] for line in source_body if len(line) > 8]
    return str(FStringReEvaluate("".join(source_body_unindented)))
