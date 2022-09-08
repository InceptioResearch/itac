"""Unit tests for termination class"""

from commonroad_rl.gym_commonroad.reward.termination import Termination
from commonroad_rl.tests.common.marker import unit_test, functional

configs = {
    "termination_configs": {
        "terminate_on_off_road": True
    }
}


@unit_test
@functional
def test_termination():
    termination = Termination(configs)
    observation = {
        "is_goal_reached": [False],
        "is_collision": [False],
        "is_off_road": [False],
        "is_time_out": [False],
        "is_friction_violation": [False],
    }

    done, _, _ = termination.is_terminated(observation, None)
    assert not done

    observation = {
        "is_goal_reached": [False],
        "is_collision": [False],
        "is_off_road": [True],
        "is_time_out": [False],
        "is_friction_violation": [False],
    }

    done, reason, _ = termination.is_terminated(observation, None)
    assert done
    assert reason == "is_off_road"
