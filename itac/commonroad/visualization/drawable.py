from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.renderer import IRenderer


class IDrawable(ABC):
    """
    Interface for drawable types
    """

    @abstractmethod
    def draw(self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None],
             call_stack: Optional[Tuple[str, ...]]) -> None:
        """
        Draw the object

        :param renderer: Renderer to use for drawing
        :param draw_params: Optional parameters ovrriding the defaults for plotting given by a nested dict that
            recreates the structure of an object or a ParamServer object
        :param call_stack: Optional tuple of string containing the call stack, which allows for differentiation of
            plotting styles depending on the call stack
        :return: None
        """
        pass
