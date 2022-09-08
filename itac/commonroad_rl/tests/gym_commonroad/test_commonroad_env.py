"""
Module tests of the module gym_commonroad
"""
import copy
import os
import glob
import random
import timeit
import numpy as np
from commonroad.scenario.scenario import ScenarioID
from stable_baselines.common.env_checker import check_env
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_rl.gym_commonroad import *
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.non_functional import function_to_string
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

resource_path = resource_root("test_gym_commonroad")
pickle_xml_scenarios(
    input_dir=os.path.join(resource_path),
    output_dir=os.path.join(resource_path, "pickles")
)

meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

output_path = output_root("test_gym_commonroad")
visualization_path = os.path.join(output_path, "visualization")


@pytest.mark.parametrize(("num_of_checks", "test_env", "play"),
                         [(15, False, False),
                          (15, False, True),
                          (15, True, False),
                          (15, True, True)])
@module_test
@functional
def test_check_env(num_of_checks, test_env, play):
    # Run more circles of checking to search for sporadic issues
    for idx in range(num_of_checks):
        print(f"Checking progress: {idx + 1}/{num_of_checks}")
        env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                       test_reset_config_path=problem_path, visualization_path=visualization_path, test_env=False,
                       play=False, )
        check_env(env)


@pytest.mark.parametrize(("reward_type"),
                         [("hybrid_reward"),
                          ("sparse_reward"),
                          ("dense_reward")])
@module_test
@functional
def test_step(reward_type):
    env = gym.make("commonroad-v1",
                   meta_scenario_path=meta_scenario_path,
                   train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path,
                   visualization_path=visualization_path,
                   reward_type=reward_type)
    env.reset()
    done = False
    while not done:
        # for i in range(50):
        action = env.action_space.sample()
        # action = np.array([0., 0.1])
        obs, reward, done, info = env.step(action)

        # TODO: define reference format and assert
        # print(f"step {i}, reward {reward:2f}")


@module_test
@functional
def test_reset_env_with_scenario():
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.obstacle import DynamicObstacle
    from commonroad.planning.planning_problem import PlanningProblem
    from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker

    env_kwargs = {"vehicle_params": {"vehicle_model": 2},
                  "cache_navigators": True,
                  "action_configs": {"continuous_collision_checking": False}}
    env_reset = gym.make("commonroad-v1",
                         meta_scenario_path=meta_scenario_path,
                         train_reset_config_path=problem_path,
                         test_reset_config_path=problem_path, **env_kwargs)
    env_step = copy.deepcopy(env_reset)

    xml_fn = sorted(glob.glob(os.path.join(resource_path, "*.xml")))[0]
    scenario, planning_problem_set = CommonRoadFileReader(xml_fn).open(lanelet_assignment=True)
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    observations_reset = env_reset.reset(scenario=scenario, planning_problem=planning_problem)
    observations_step = env_step.reset(scenario=scenario, planning_problem=planning_problem)
    assert np.allclose(observations_reset, observations_step)

    collision_checker_step = create_collision_checker(scenario)
    observation_keys = []
    for key, value in env_reset.observation_dict.items():
        observation_keys.extend([key]*value.size)
    obs_distance_goal_long_advance_idx = observation_keys.index("distance_goal_long_advance")
    obs_distance_goal_lat_advance_idx = observation_keys.index("distance_goal_lat_advance")
    obs_distance_goal_long_idx = observation_keys.index("distance_goal_long")
    obs_distance_goal_lat_idx = observation_keys.index("distance_goal_lat")
    obs_distance_goal_time = observation_keys.index("distance_goal_time")
    obs_remaining_steps = observation_keys.index("remaining_steps")

    for i in range(1, 50):
        action = np.array([0.2, 1.0])
        _ = env_reset.step(action)
        observations_step, _, done, info = env_step.step(action)
        # env_step.ego_action.vehicle.update_collision_object(create_convex_hull=False)
        collision_step = collision_checker_step.collide(env_step.ego_action.vehicle.collision_object)

        obs_distance_goal_long = observations_reset[obs_distance_goal_long_idx]
        obs_distance_goal_lat = observations_reset[obs_distance_goal_lat_idx]

        # create new scenario with next step
        next_scenario = Scenario(dt=scenario.dt, scenario_id=ScenarioID("tmp"))
        next_scenario.add_objects(scenario.lanelet_network)
        for obstacle in scenario.dynamic_obstacles:
            obstacle_state = copy.deepcopy(obstacle.state_at_time(i))
            if obstacle_state is None:
                continue
            obstacle_state.time_step = 0
            next_obstacle = DynamicObstacle(obstacle.obstacle_id,
                                            obstacle.obstacle_type,
                                            obstacle.obstacle_shape,
                                            obstacle_state,
                                            prediction=None)
            next_scenario.add_objects(next_obstacle)
        next_scenario.assign_obstacles_to_lanelets(time_steps=[0])
        collision_checker_reset = create_collision_checker(next_scenario)
        ego_next_state = env_reset.ego_action.vehicle.state
        ego_next_state.time_step = 0
        ego_next_state.slip_angle = 0.

        observations_reset = env_reset.reset(scenario=next_scenario, planning_problem=PlanningProblem(
            planning_problem.planning_problem_id,
            ego_next_state,
            planning_problem.goal
        ))

        collision_reset = collision_checker_reset.collide(env_reset.ego_action.vehicle.collision_object)

        if collision_reset != collision_step:
            # find obstacle that causes collision
            for dynamic_obstacle in scenario.dynamic_obstacles:
                co = create_collision_object(dynamic_obstacle)
                if co.collide(env_step.ego_action.vehicle.collision_object):
                    obstacle_id = dynamic_obstacle.obstacle_id
                    break
            import commonroad_dc.pycrcc as pycrcc
            obstacle_reset = next_scenario.obstacle_by_id(obstacle_id)
            obstacle_state_reset = obstacle_reset.state_at_time(0)
            co_reset_obstacle = pycrcc.RectOBB(obstacle_reset.obstacle_shape.length / 2,
                                               obstacle_reset.obstacle_shape.width / 2,
                                               obstacle_state_reset.orientation,
                                               obstacle_state_reset.position[0],
                                               obstacle_state_reset.position[1])
            print(co_reset_obstacle.collide(env_reset.ego_action.vehicle.collision_object))

            obstacle_step = scenario.obstacle_by_id(obstacle_id)
            obstacle_state_step = obstacle_step.state_at_time(i)
            co_step_obstacle = pycrcc.RectOBB(obstacle_step.obstacle_shape.length / 2,
                                               obstacle_step.obstacle_shape.width / 2,
                                               obstacle_state_step.orientation,
                                               obstacle_state_step.position[0],
                                               obstacle_state_step.position[1])
            # print(co_step_obstacle.collide(env_step.ego_action.vehicle.collision_object))
            # print(co_step_obstacle.collide(env_reset.ego_action.vehicle.collision_object))

            from commonroad.visualization.mp_renderer import MPRenderer
            renderer = MPRenderer()
            co_step_obstacle.draw(renderer, draw_params={"facecolor": "blue"})
            co_reset_obstacle.draw(renderer, draw_params={"facecolor": "black"})
            env_reset.ego_action.vehicle.collision_object.draw(renderer, draw_params={"facecolor": "red"})
            env_step.ego_action.vehicle.collision_object.draw(renderer, draw_params={"facecolor": "green"})

            renderer.render(show=True)

        assert collision_reset == collision_step == \
               env_step.observation_dict['is_collision'] == env_reset.observation_dict['is_collision'], \
            f"step={i} " \
            f"ego_state_step={env_step.ego_action.vehicle.state}, ego_state_reset={env_reset.ego_action.vehicle.state}"

        # fix goal distance advance
        observations_reset[obs_distance_goal_long_advance_idx] = \
            abs(obs_distance_goal_long) - abs(observations_reset[obs_distance_goal_long_idx])
        observations_reset[obs_distance_goal_lat_advance_idx] = \
            abs(obs_distance_goal_lat) - abs(observations_reset[obs_distance_goal_lat_idx])
        observations_reset[obs_remaining_steps] -= i
        observations_reset[obs_distance_goal_time] += i

        # a = np.where(np.abs(observations_step-observations_reset) > 1e-8)
        assert np.allclose(observations_reset, observations_step), f"{env_reset.observation_dict['is_collision']}, " \
                                                                   f"{env_step.observation_dict['is_collision']}, " \
                                                                   f"collision_reset={collision_reset}, " \
                                                                   f"collision_step={collision_step}"

        if not np.allclose(observations_reset, observations_step):
            print(f"ego_state: env_reset={env_reset.ego_action.vehicle.state}, env_step={env_step.ego_action.vehicle.state}")
            keys = []
            idxs = np.where(np.abs(observations_step-observations_reset) > 1e-8)
            for idx in idxs[0]:
                keys.append(observation_keys[idx])
                print(f"{observation_keys[idx]}: observation_reset={observations_reset[idx]}, observation_step={observations_step[idx]}")
            raise ValueError


@pytest.mark.parametrize(("reward_type"),
                         [("dense_reward"),
                          ("sparse_reward"),
                          ("hybrid_reward")])
@module_test
@functional
def test_observation_order(reward_type):
    env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path, flatten_observation=False)

    # set random seed to make the env choose the same planning problem
    random.seed(0)
    obs_dict = env.reset()

    # collect observation in other format
    env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path, flatten_observation=True)

    # seed needs to be reset before function call
    random.seed(0)
    obs_flatten = env.reset()
    obs_flatten_exp = np.zeros(env.observation_space.shape)

    # flatten the dictionary observation
    index = 0
    for obs_dict_value in obs_dict.values():
        size = np.prod(obs_dict_value.shape)
        obs_flatten_exp[index: index + size] = obs_dict_value.flat
        index += size

    # compare 2 observation
    assert np.allclose(obs_flatten_exp, obs_flatten), "Two observations don't have the same order"


@pytest.mark.parametrize(("reward_type"),
                         [("dense_reward"),
                          ("sparse_reward"),
                          ("hybrid_reward")])
@module_test
@nonfunctional
def test_step_time(reward_type):
    # Define reference time
    reference_time = 15.0

    def measurement_setup():
        import gym
        import numpy as np

        env = gym.make("commonroad-v1", meta_scenario_path="{meta_scenario_path}",
                       train_reset_config_path="{problem_path}",
                       test_reset_config_path="{problem_path}", visualization_path="{visualization_path}",
                       reward_type="{reward_type}", )
        env.reset()
        action = np.array([0.0, 0.0])

    def measurement_code(env, action):
        env.step((action))

    setup_str = function_to_string(measurement_setup)
    code_str = function_to_string(measurement_code)

    times = timeit.repeat(setup=setup_str, stmt=code_str, repeat=1, number=1000)
    min_time = np.amin(times)

    # TODO: Set exclusive CPU usage for this thread, because other processes influence the result  # assert
    #  average_time < reference_time, f"The step is too slow, average time was {average_time}"
