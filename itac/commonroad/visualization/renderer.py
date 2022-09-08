from abc import ABCMeta, abstractmethod


class IRenderer(metaclass=ABCMeta):
    @abstractmethod
    def draw_scenario(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_static_obstacle(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_dynamic_obstacle(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_phantom_obstacle(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_environment_obstacle(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_trajectory(self, obj, draw_params, call_stack):
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_trajectories(self, obj, draw_params, call_stack):
        pass

    @abstractmethod
    def draw_polygon(self, vertices, draw_params, call_stack):
        """
        Draws a polygon shape
        :param vertices: vertices of the polygon
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_rectangle(self, vertices, draw_params, call_stack):
        """
        Draws a rectangle shape
        :param vertices: vertices of the rectangle
        :param draw_params: parameters for plotting given by a nested dict that
        recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_ellipse(self, center, radius_x, radius_yt, draw_params,
                     call_stack):
        """
        Draws a circle shape
        :param ellipse: center position of the ellipse
        :param radius_x: radius of the ellipse along the x-axis
        :param radius_y: radius of the ellipse along the y-axis
        :param draw_params: parameters for plotting given by a nested dict that
        recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_state(self, state, draw_params, call_stack):
        """
        Draws a state as an arrow of its velocity vector
        :param state: state to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_lanelet_network(self, obj, draw_params, call_stack):
        """
        Draws a lanelet network
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict that
        recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_goal_region(self, obj, draw_params, call_stack):
        """
        Draw goal states from goal region
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_planning_problem(self, obj, draw_params, call_stack):
        """
        Draw initial state and goal region of the planning problem
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_planning_problem_set(self, obj, draw_params, call_stack):
        """
        Draws all or selected planning problems from the planning problem
        set. Planning problems can be selected by providing IDs in
        `drawing_params[planning_problem_set][draw_ids]`
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_initital_state(self, obj, draw_params, call_stack):
        """
        Draw initial state with label
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_goal_state(self, obj, draw_params, call_stack):
        """
        Draw goal states
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass

    @abstractmethod
    def draw_traffic_light_sign(self, obj, draw_params, call_stack):
        """
        Draw traffic sings and lights
        :param obj: object to be drawn
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack
        :return: None
        """
        pass
