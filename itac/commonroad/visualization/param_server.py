import json
import logging
import os
import warnings
from typing import Tuple, Union

# Read defaults only once
default_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'default_draw_params.json')
with open(default_params_path) as fp:
    default_params = json.load(fp)


def write_default_params(filename: str) -> None:
    """
    Write default parameters to the give file as JSON

    :param filename: Destination JSON file
    :return: None
    """
    with open(filename, 'w') as fp:
        json.dump(default_params, fp, indent=4)


class ParamServer:
    """
    Wrapper object for parameters as nested dictionaries. Tries to resolve
    queries with most specialized values. If values are not found, defaults
    are used.
    """

    def __init__(self, params: dict = None, warn_default: bool = False, default=None):
        """
        :param params: Optional parameters to initialize parameter server with
        :param warn_default: Produce a warning when default parameters are used
        :param default: Optional default parameter set. If provided, overrides the defaults in default_draw_params.json.
        """
        self._params = params or {}
        self._warn_default = warn_default
        if isinstance(default, dict):
            self._default = ParamServer(params=default)
        elif isinstance(default, ParamServer):
            self._default = default
        else:
            self._default = default_params

    @staticmethod
    def _resolve_key(param_dict, key):
        if isinstance(param_dict, ParamServer):
            return param_dict.resolve_key(key)
        else:
            if len(key) == 0:
                return None, 0
            tmp_dict = param_dict
            l_key = list(key)
            # Try to find most special version of element
            while len(l_key) > 0:
                k = l_key.pop(0)
                if k in tmp_dict.keys():
                    tmp_dict = tmp_dict[k]
                else:
                    tmp_dict = None
                    break
            if tmp_dict is None and len(key) > 0:
                # If not found, remove first level and try again
                return ParamServer._resolve_key(param_dict, key[1:])
            else:
                return tmp_dict, len(key)

    def resolve_key(self, param_path):
        val, depth = ParamServer._resolve_key(self._params, param_path)
        val_default, depth_default = ParamServer._resolve_key(self._default, param_path)
        if val is None and val_default is None:
            return None, None

        if val_default is not None and val is None:
            if self._warn_default:
                logging.warning('Using default for key {}!'.format(param_path))
            return val_default, depth_default

        if val is not None and val_default is None:
            return val, depth

        if val is not None and val_default is not None:
            if depth >= depth_default:
                return val, depth
            else:
                return val_default, depth_default

    def by_callstack(self, call_stack: Tuple[str, ...], param_path: Union[str, Tuple[str, ...]]):
        """
        Resolves the parameter path using the callstack. If it nothing can be
        found returns None

        :param call_stack: Tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :param param_path: Key or tuple of keys leading to the parameter
        :return: the parameter
        """
        if isinstance(param_path, tuple):
            path = call_stack + param_path
        else:
            path = call_stack + (param_path,)
        return self.__getitem__(path)

    def __getitem__(self, param_path):
        """
        Resolves the parameter by the given key tuple. Parameters are
        resolved recursively. If no parameter can be found under the given
        path, the first element of the tuple is removed and the resolution
        will be retried. This yields the most specialized version of the
        parameter. Default parameters are provided if:

        a) the specified path cannot be resolved in the contained parameters or
        b) the default parameters contain a more specific version than the
        contained parameters

        :param param_path: Key or tuple of keys leading to the parameter
        :return: the parameter
        """
        if not isinstance(param_path, tuple):
            param_path = (param_path,)

        val, _ = self.resolve_key(param_path)
        if val is None:
            warnings.warn(f"Value for key {param_path} not found!")
        return val

    def __setitem__(self, param_path, value):
        """
        Sets the value under the given key

        :param param_path: key or tuple of keys leading to the parameter
        :param value: the value
        :return: None
        """
        if not isinstance(param_path, tuple):
            param_path = (param_path,)
        tmp_dict = self._params
        for key in param_path[:-1]:
            if not isinstance(tmp_dict, dict):
                raise KeyError(
                        'Key "{}" in path "{}" is not subscriptable!'.format(
                            key, param_path))
            if key in tmp_dict.keys():
                tmp_dict = tmp_dict[key]
            else:
                tmp_dict[key] = {}
                tmp_dict = tmp_dict[key]
        if not isinstance(tmp_dict, dict):
            raise KeyError(
                    'Key "{}" in path "{}" is not subscriptable!'.format(key,
                                                                         param_path))
        tmp_dict[param_path[-1]] = value

    def __contains__(self, item):
        return item in self._params

    def update(self, source):
        self._update_recurisve(self._params, source)

    def _update_recurisve(self, dest, source):
        for k, v in source.items():
            if k in dest:
                if isinstance(dest[k], dict):
                    assert isinstance(source[k], dict)
                    self._update_recurisve(dest[k], source[k])
                else:
                    dest[k] = source[k]
            else:
                dest[k] = source[k]

    @staticmethod
    def from_json(fname: str):
        """
        Restores a parameter server from a JSON file

        :param fname: file name and path of the JSON file
        :return: the parameter server
        """
        with open(fname, 'r') as fp:
            data = json.load(fp)
        return ParamServer(data)
