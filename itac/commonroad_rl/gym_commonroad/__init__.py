"""
CommonRoad Gym environment
"""
import gym

# Notice: this code is run everytime the gym_commonroad module is imported
# this might be pretty shady but seems to be common practice so let's at least catch the errors occurring here
try:
    gym.envs.register(
        id="commonroad-v1",
        entry_point="commonroad_rl.gym_commonroad.commonroad_env:CommonroadEnv",
        kwargs=None,
    )
except gym.error.Error:
    # print("[gym_commonroad/__init__.py] Error occurs while registering commonroad-v1")
    pass
