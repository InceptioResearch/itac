"""Unit tests for sparse reward class"""
from commonroad_rl.gym_commonroad.reward.sparse_reward import SparseReward
from commonroad_rl.tests.common.marker import unit_test, functional

configs = {
    "reward_type": "sparse_reward",
    "reward_configs_sparse": {
        "reward_goal_reached": 50.,
        "reward_collision": -50.,
        "reward_off_road": -20.,
        "reward_friction_violation": -30.,
        "reward_time_out": -10.
    }
}


@unit_test
@functional
def test_sparse_reward():
    reward = SparseReward(configs)
    observations = {
        "is_goal_reached": [True],
        "is_collision": [False],
        "is_off_road": [False],
        "is_friction_violation": [False],
        "is_time_out": [False]
    }

    assert reward.calc_reward(observations, None) == 50

    observations = {
        "is_goal_reached": [False],
        "is_collision": [False],
        "is_off_road": [True],
        "is_friction_violation": [False],
        "is_time_out": [False]
    }

    assert reward.calc_reward(observations, None) == -20
