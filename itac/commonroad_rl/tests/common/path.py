"""
Helper module for the path of tests
"""
import os

# Directory contains the output of tests
output_path = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Directory contains the reference files of tests
reference_path = os.path.join(os.path.dirname(__file__), "..", "references")

# Directory contains the resource files of tests
resource_path = os.path.join(os.path.dirname(__file__), "..", "resources")


def resource_root(test_name: str) -> str:
    """
    Returns the root of the resources

    :return: The root of the resources
    """
    return os.path.join(resource_path, test_name)


def reference_root(test_name: str) -> str:
    """
    Returns the root of the references

    :return: The root of the references
    """
    return os.path.join(reference_path, test_name)


def output_root(test_name: str) -> str:
    """
    Returns the root of the outputs of a given test

    :param test_name: The name of the test
    :return: The root of the outputs
    """
    path = os.path.join(output_path, test_name)
    os.makedirs(path, exist_ok=True)
    return path
