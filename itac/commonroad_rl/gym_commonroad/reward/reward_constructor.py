"""File containing a reward function generator (Can't be incorporated into other modules because of cyclic import
problems """
from commonroad_rl.gym_commonroad.reward.dense_reward import DenseReward
from commonroad_rl.gym_commonroad.reward.hybrid_reward import HybridReward
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.reward.sparse_reward import SparseReward


def make_reward(configs: dict) -> Reward:
    """
    Initializes the reward class according to the env_configurations

    :param configs: The configuration of the environment
    :return: Reward class, either hybrid, sparse or dense
    """

    reward_type = configs["reward_type"]

    if reward_type == "sparse_reward":
        return SparseReward(configs)
    elif reward_type == "hybrid_reward":
        return HybridReward(configs)
    elif reward_type == "dense_reward":
        return DenseReward(configs)
    else:
        raise ValueError(f"Illegal reward type: {reward_type}!")
