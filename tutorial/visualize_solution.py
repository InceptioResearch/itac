from itac import *
import matplotlib.pyplot as plt

vehicle4 = parameters_semi_trailer.parameters_semi_trailer()


def visualize_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, trajectory: Trajectory) -> None:
    from IPython import display

    num_time_steps = len(trajectory.state_list)

    # create the ego vehicle prediction using the trajectory and the shape of the obstacle
    dynamic_obstacle_initial_state = trajectory.state_list[0]
    dynamic_obstacle_shape = Rectangle(width=1.8, length=4.3)
    dynamic_obstacle_prediction = TrajectoryPrediction(
        trajectory, dynamic_obstacle_shape)

    # generate the dynamic obstacle according to the specification
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = ObstacleType.CAR
    dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                       dynamic_obstacle_type,
                                       dynamic_obstacle_shape,
                                       dynamic_obstacle_initial_state,
                                       dynamic_obstacle_prediction)

    # visualize scenario
    plt.ion()
    plt.figure(figsize=(15, 15))
    for i in range(0, num_time_steps):
        display.clear_output(wait=True)
        # plt.figure(figsize=(10, 10))
        center_x = trajectory.state_list[i].position[0]
        center_y = trajectory.state_list[i].position[1]
        renderer = MPRenderer(
            plot_limits=[center_x-70, center_x+70, center_y-70, center_y+70])
        scenario.draw(renderer, draw_params={'time_begin': i})
        planning_problem_set.draw(renderer)
        velocity = trajectory.state_list[i].velocity
        dynamic_obstacle.draw(renderer, draw_params={'time_begin': i,
                                                     'dynamic_obstacle': {'shape': {'facecolor': 'green'},
                                                                          'trajectory': {'draw_trajectory': True,
                                                                                         'facecolor': '#ff00ff',
                                                                                         'draw_continuous': True,
                                                                                         'z_order': 60,
                                                                                         'line_width': 5}
                                                                          }
                                                     })

        plt.gca().set_aspect('equal')
        renderer.render()
        plt.text(center_x+20, center_y+20,
                 '{:.3f} m/s, time_step:{}'.format(velocity, i), fontsize=14)
        # plt.show()
        plt.pause(scenario.dt)


class EgoParameters:

    tractor_l = vehicle4.l
    tractor_w = vehicle4.w
    # axes distances
    tractor_wb = vehicle4.a + vehicle4.b

    # trailer parameters
    trailer_l = vehicle4.trailer.l = 13.6  # trailer length
    trailer_w = vehicle4.trailer.w = 2.55  # trailer width
    trailer_l_hitch = vehicle4.trailer.l_hitch = 12.00  # hitch length
    trailer_l_total = vehicle4.trailer.l_total = 16.5  # total system length
    trailer_l_wb = vehicle4.trailer.l_wb = 8.1  # trailer wheelbase


def truck_visualize_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, trajectory: Trajectory) -> None:
    from IPython import display

    num_time_steps = len(trajectory.state_list)
    vehicle_params = EgoParameters()

    # create the ego vehicle prediction using the trajectory and the shape of the obstacle
    truck_tractor_initial_state = trajectory.state_list[0]
    truck_tractor_shape = Rectangle(
        width=vehicle_params.tractor_w, length=vehicle_params.tractor_l)
    truck_tractor_prediction = TrajectoryPrediction(
        trajectory, truck_tractor_shape)

    # generate the dynamic obstacle according to the specification
    truck_tractor_id = scenario.generate_object_id()
    truck_tractor = DynamicObstacle(obstacle_id=truck_tractor_id,
                                    obstacle_type=ObstacleType.TRUCK,
                                    obstacle_shape=truck_tractor_shape,
                                    initial_state=truck_tractor_initial_state,
                                    prediction=truck_tractor_prediction)

    trailer_state_list = []
    for i in range(0, num_time_steps):
        # add new state to state_list
        trailer_state_list.append(State(**{'position': trajectory.state_list[i].position_trailer,
                                           'orientation': trajectory.state_list[i].yaw_angle_trailer,
                                           'time_step': i}))
    # create the planned trajectory starting at time step 0

    trailer_trajectory = Trajectory(
        initial_time_step=0, state_list=trailer_state_list[0:])
    # create the prediction using the planned trajectory and the shape of the trailer
    trailer_shape = Rectangle(length=vehicle_params.trailer_l-(vehicle_params.tractor_l / 2 - vehicle_params.tractor_wb / 2),
                              width=vehicle4.trailer.w)
    trailer_prediction = TrajectoryPrediction(trajectory=trailer_trajectory,
                                              shape=trailer_shape)
    # the trailer can be visualized by converting it into a DynamicObstacle
    trailer = DynamicObstacle(obstacle_id=100, obstacle_type=ObstacleType.TRUCK,
                              obstacle_shape=trailer_shape, initial_state=trailer_state_list[0],
                              prediction=trailer_prediction)
    # visualize scenario
    plt.ion()
    plt.figure(figsize=(10, 10))
    for i in range(0, num_time_steps):
        display.clear_output(wait=True)
        # plt.figure(figsize=(10, 10))
        center_x = trajectory.state_list[i].position[0]
        center_y = trajectory.state_list[i].position[1]
        renderer = MPRenderer(
            plot_limits=[center_x-100, center_x+100, center_y-100, center_y+100])
        scenario.draw(renderer, draw_params={'time_begin': i})
        planning_problem_set.draw(renderer)
        velocity = trajectory.state_list[i].velocity
        truck_tractor.draw(renderer, draw_params={'time_begin': i,
                                                  'dynamic_obstacle': {'shape': {'facecolor': 'green'},
                                                                       'trajectory': {'draw_trajectory': True,
                                                                                      'facecolor': '#ff00ff',
                                                                                      'draw_continuous': True,
                                                                                      'z_order': 60,
                                                                                      'line_width': 5}
                                                                       }
                                                  })

        trailer.draw(renderer, draw_params={'time_begin': i, 'dynamic_obstacle': {
            'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
                'facecolor': 'darkviolet'}}}}}})
        plt.gca().set_aspect('equal')
        renderer.render()
        plt.text(center_x+20, center_y+20,
                 '{:.3f} m/s, time_step:{}'.format(velocity, i), fontsize=14)
        plt.show()
        plt.pause(scenario.dt)
