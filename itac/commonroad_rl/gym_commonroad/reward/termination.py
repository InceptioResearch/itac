"""Module containing the Termination class"""
from commonroad_rl.gym_commonroad.action import Action


class Termination:
    """Class for detecting if the scenario should be terminated"""

    def __init__(self, config: dict):
        """
        :param config: Configuration of the environment
        """

        self.termination_configs = config["termination_configs"]
        self.num_friction_violation = 0

    def reset(self, observation: dict, ego_action: Action):
        """
        For resetting the memory the termination class has for the current scenario

        :param observation: Current observation of the environment
        :param ego_action: Current ego_action of the environment
        """
        self.num_friction_violation = 0

    def is_terminated(self, observation: dict, ego_action: Action) -> (bool, str, dict):
        """
        Detect if the scenario should be terminated

        :param observation: Current observation of the environment
        :param ego_action: Current ego_action of the environment
        :return: Tuple of (terminated: bool, reason: str, termination_info: dict)
        """

        done = False
        termination_info = {
            "is_goal_reached": 0,
            "is_collision": 0,
            "is_off_road": 0,
            "is_time_out": 0,
            "is_friction_violation": 0,
            "num_friction_violation": self.num_friction_violation
        }
        termination_reason = None

        if observation["is_off_road"][0]:  # Ego vehicle is off-road
            termination_info["is_off_road"] = 1
            if self.termination_configs["terminate_on_off_road"]:
                done = True
                termination_reason = "is_off_road"

        elif observation["is_collision"][0]:  # Collision with others
            termination_info["is_collision"] = 1
            if self.termination_configs["terminate_on_collision"]:
                done = True
                termination_reason = "is_collision"

        elif observation["is_goal_reached"][0]:  # Goal region is reached
            termination_info["is_goal_reached"] = 1
            if self.termination_configs["terminate_on_goal_reached"]:
                done = True
                termination_reason = "is_goal_reached"

        elif observation["is_time_out"][0]:  # Max simulation time step is reached
            termination_info["is_time_out"] = 1
            if self.termination_configs["terminate_on_time_out"]:
                done = True
                termination_reason = "is_time_out"

        elif observation["is_friction_violation"][0]:  # Friction limitation is violated
            self.num_friction_violation += 1
            termination_info["is_friction_violation"] = 1
            termination_info["num_friction_violation"] = self.num_friction_violation
            if self.termination_configs["terminate_on_friction_violation"]:
                done = True
                termination_reason = "is_friction_violation"

        return done, termination_reason, termination_info
