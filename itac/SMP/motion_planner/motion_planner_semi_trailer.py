from typing import Union
from enum import Enum, unique

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.maneuver_automaton.maneuver_automaton_semi_trailer import TruckManeuverAutomaton
# iterative search algorithms
from SMP.motion_planner.search_algorithms.sequential_search import BreadthFirstSearch
from SMP.motion_planner.search_algorithms.sequential_search import DepthFirstSearch
# best first search algorithms
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
from SMP.motion_planner.search_algorithms.best_first_search import UniformCostSearch
# depth limited search algorithms
from SMP.motion_planner.search_algorithms.depth_limited_search import DepthLimitedSearch
# motion planner from student
from SMP.motion_planner.search_algorithms.student import StudentTruckMotionPlanner
from SMP.motion_planner.search_algorithms.student_example import StudentTruckMotionPlannerExample


@unique
class TruckMotionPlannerType(Enum):
    """
    Enumeration definition of different algorithms.
    """
    BFS = "bfs"
    DFS = "dfs"
    UCS = "ucs"
    GBFS = "gbfs"
    ASTAR = "astar"
    DLS = "dls"
    STUDENT = "student"
    STUDENT_EXAMPLE = "student_example"


class TruckMotionPlanner:
    """
    Class to load and execute the specified motion planner.
    """

    class NoSuchTruckMotionPlanner(KeyError):
        """
        Error message when the specified motion planner does not exist.
        """

        def __init__(self, message):
            self.message = message

    dict_motion_planners = dict()
    dict_motion_planners[TruckMotionPlannerType.BFS] = BreadthFirstSearch
    dict_motion_planners[TruckMotionPlannerType.DFS] = DepthFirstSearch
    dict_motion_planners[TruckMotionPlannerType.UCS] = UniformCostSearch
    dict_motion_planners[TruckMotionPlannerType.GBFS] = GreedyBestFirstSearch
    dict_motion_planners[TruckMotionPlannerType.ASTAR] = AStarSearch
    dict_motion_planners[TruckMotionPlannerType.DLS] = DepthLimitedSearch
    # add your own custom motion planner here
    dict_motion_planners[TruckMotionPlannerType.STUDENT] = StudentTruckMotionPlanner
    dict_motion_planners[TruckMotionPlannerType.STUDENT_EXAMPLE] = StudentTruckMotionPlannerExample

    @classmethod
    def create(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
               plot_config=DefaultPlotConfig,
               motion_planner_type: TruckMotionPlannerType = TruckMotionPlannerType.GBFS) -> Union[UniformCostSearch,
                                                                                                   DepthLimitedSearch,
                                                                                                   GreedyBestFirstSearch,
                                                                                                   BreadthFirstSearch,
                                                                                                   DepthFirstSearch,
                                                                                                   AStarSearch,
                                                                                                   StudentTruckMotionPlanner,
                                                                                                   StudentTruckMotionPlannerExample]:
        """
        Method to instantiate the specified motion planner.
        """
        try:
            return cls.dict_motion_planners[motion_planner_type](scenario, planning_problem, automaton,
                                                                 plot_config=plot_config)
        except KeyError:
            raise cls.NoSuchTruckMotionPlanner(
                f"TruckMotionPlanner with type <{motion_planner_type}> does not exist.")

    @classmethod
    def BreadthFirstSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                           plot_config=DefaultPlotConfig) -> BreadthFirstSearch:
        """
        Method to instantiate a Breadth-First-Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.BFS)

    @classmethod
    def DepthFirstSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                         plot_config=DefaultPlotConfig) -> DepthFirstSearch:
        """
        Method to instantiate a Depth-First-Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.DFS)

    @classmethod
    def DepthLimitedSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                           plot_config=DefaultPlotConfig) -> DepthLimitedSearch:
        """
        Method to instantiate a Depth-Limited-Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.DLS)

    @classmethod
    def UniformCostSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                          plot_config=DefaultPlotConfig) -> UniformCostSearch:
        """
        Method to instantiate a Uniform-Cost-Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.UCS)

    @classmethod
    def GreedyBestFirstSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                              plot_config=DefaultPlotConfig) -> GreedyBestFirstSearch:
        """
        Method to instantiate a Greedy-Best-First-Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.GBFS)

    @classmethod
    def AStarSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                    plot_config=DefaultPlotConfig) -> AStarSearch:
        """
        Method to instantiate an A* Search motion planner.
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.ASTAR)

    @classmethod
    def StudentTruckMotionPlanner(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: TruckManeuverAutomaton,
                                  plot_config=DefaultPlotConfig) -> StudentTruckMotionPlanner:
        """
        Method to instantiate your motion planner..
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config, TruckMotionPlannerType.STUDENT)

    @classmethod
    def StudentTruckMotionPlannerExample(cls, scenario: Scenario, planning_problem: PlanningProblem,
                                         automaton: TruckManeuverAutomaton,
                                         plot_config=DefaultPlotConfig) -> StudentTruckMotionPlannerExample:
        """
        Method to instantiate your motion planner..
        """
        return TruckMotionPlanner.create(scenario, planning_problem, automaton, plot_config,
                                         TruckMotionPlannerType.STUDENT_EXAMPLE)
