import os
import gym
import numpy as np
from commonroad_rl.tests.common.marker import *
from commonroad_rl.utils_run.wrappers import IncreaseTimeStepWrapper
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

resource_path = resource_root("test_wrapper")
pickle_xml_scenarios(
    input_dir=os.path.join(resource_path),
    output_dir=os.path.join(resource_path, "pickles")
)

meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

@pytest.mark.parametrize(
    "planning_horizon",
    [1.,
     2.,
     0.4]
)

@module_test
@functional
def test_step(planning_horizon):
    env = gym.make("commonroad-v1",
                   meta_scenario_path=meta_scenario_path,
                   train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path,
                   action_configs={"planning_horizon": planning_horizon})

    wrapper = IncreaseTimeStepWrapper
    env.seed(1)
    env = wrapper(env)
    env.reset()
    done = False
    step = 0
    while not done:
        action = np.array([0., 0.])
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
    expected_step = env.current_step // int(planning_horizon / env.scenario.dt) + 1
    assert step == expected_step
