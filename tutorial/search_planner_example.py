import time
import argparse
import matplotlib.pyplot as plt
from visualize_solution import visualize_solution
from itac import *
import sys

sys.path.append('/home/liwei/code/itac/itac/')
from itac import *

from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType  # NOQA
from SMP.motion_planner.utility import plot_primitives  # NOQA
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton  # NOQA

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='input commonroad xml and choose search motion planner(ASTAR, GBFS, UCS, DFS, BFS)')
    parser.add_argument("--input",
                        help="commonroad xml",
                        default="scenarios/highway/CHN_AEB-1_1_T-1.xml")
    parser.add_argument("--search_method",
                        help="search motion planner",
                        default="ASTAR")
    parser.add_argument("--vehicle_model",
                        help='vehicle model',
                        default='itac/vehiclemodels/primitives/vehicle_model_primitives_V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml')

    args = parser.parse_args()
    file = args.input
    planner = args.search_method
    vehicle = args.vehicle_model
    if planner == 'ASTAR':
        # A* Search
        type_motion_planner = MotionPlannerType.ASTAR
    elif planner == 'GBFS':
        # Greedy Best First Search
        type_motion_planner = MotionPlannerType.GBFS
    elif planner == 'UCS':
        # Uniform Cost Search (aka Dijkstra's algorithm)
        type_motion_planner = MotionPlannerType.UCS
    elif planner == 'BFS':
        # Breadth First Search
        type_motion_planner = MotionPlannerType.BFS
    elif planner == 'DFS':
        # Depth First Search
        type_motion_planner = MotionPlannerType.DFS
    elif planner == 'STUDENT':
        # you own motion planner
        type_motion_planner = MotionPlannerType.STUDENT

    print('Input commonroad xml file: {}'.format(file))
    print('Choose search motion planner: {}'.format(planner))
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(file).open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(
        planning_problem_set.planning_problem_dict.values())[0]

    # load the xml with stores the motion primitives
    name_file_motion_primitives = vehicle
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(
        name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # construct motion planner
    print('Construct motion planner ...')
    motion_planner = MotionPlanner.create(scenario=scenario,
                                          planning_problem=planning_problem,
                                          automaton=automaton,
                                          motion_planner_type=type_motion_planner)

    # solve for solution
    print('Solve for solution ...')
    list_paths_primitives, _, _ = motion_planner.execute_search(timeout_sec=30, vehicle_type = 'SEDAN')
    if list_paths_primitives == None:
        print('Cannot Find solution ...')
        exit(0)
    print('Find solution ...')

    print('Visualize solution ...')
    trajectory_solution = create_trajectory_from_list_states(
        list_paths_primitives)

    # visualize solution
    visualize_solution(scenario, planning_problem_set, trajectory_solution)

    time.sleep((len(trajectory_solution.state_list)) / 10 + 1)

    # create PlanningProblemSolution object
    kwarg = {'planning_problem_id': planning_problem.planning_problem_id,
             # used vehicle model, change if needed
             'vehicle_model': VehicleModel.SEDAN,
             # used vehicle type, change if needed
             'vehicle_type': VehicleType.SEDAN,
             # cost funtion, DO NOT use JB1
             'cost_function': CostFunction.ST,
             'trajectory': trajectory_solution}

    planning_problem_solution = PlanningProblemSolution(**kwarg)

    # create Solution object
    kwarg = {'scenario_id': scenario.scenario_id,
             'planning_problem_solutions': [planning_problem_solution]}

    solution = Solution(**kwarg)

    """ # valid solution
    from commonroad_dc.feasibility.solution_checker import valid_solution
    valid_solution(scenario, planning_problem_set, solution) """

    dir_output = "./outputs/solutions/"
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    # write solution to a CommonRoad XML file
    csw = CommonRoadSolutionWriter(solution)
    csw.write_to_file(output_path=dir_output, overwrite=True)
    print('Solution file saved.')
