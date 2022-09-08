import itertools
from enum import Enum
from typing import Dict, List, Tuple

from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import LaneletRouteMatcher, SolutionProperties, merge_trajectories

from commonroad.common.solution import Solution, CostFunction, VehicleType

import commonroad_dc.costs.partial_cost_functions as cost_functions


class PartialCostFunction(Enum):
    """
    See https://gitlab.lrz.de/tum-cps/commonroad-cost-functions/-/blob/master/costFunctions_commonroad.pdf for more
    details.

    A: Acceleration,
    J: Jerk,
    Jlat: Lateral Jerk,
    Jlon: Longitudinal Jerk,
    SA: Steering Angle,
    SR: Steering Rate,
    Y: Yaw Rate,
    LC: Lane Center Offset,
    V: Velocity Offset,
    Vlon: Longitudinal Velocity Offset,
    O: Orientation Offset,
    D: Distance to Obstacles,
    L: Path Length,
    T: Time,
    ID: Inverse Duration,
    """
    A = "A"
    J = "J"
    Jlat = "Jlat"
    Jlon = "Jlon"
    SA = "SA"
    SR = "SR"
    Y = "Y"
    LC = "LC"
    V = "V"
    Vlon = "Vlon"
    O = "O"
    D = "D"
    L = "L"
    T = "T"
    ID = "ID"


PartialCostFunctionMapping = {
    PartialCostFunction.A:  cost_functions.acceleration_cost,
    PartialCostFunction.J:  cost_functions.jerk_cost,
    PartialCostFunction.Jlat:  cost_functions.jerk_lat_cost,
    PartialCostFunction.Jlon:  cost_functions.jerk_lon_cost,
    PartialCostFunction.SA:  cost_functions.steering_angle_cost,
    PartialCostFunction.SR:  cost_functions.steering_rate_cost,
    PartialCostFunction.Y:  cost_functions.yaw_cost,
    PartialCostFunction.LC:  cost_functions.lane_center_offset_cost,
    PartialCostFunction.V:  cost_functions.velocity_offset_cost,
    PartialCostFunction.Vlon:  cost_functions.longitudinal_velocity_offset_cost,
    PartialCostFunction.O:  cost_functions.orientation_offset_cost,
    PartialCostFunction.D:  cost_functions.distance_to_obstacle_cost,
    PartialCostFunction.L:  cost_functions.lane_center_offset_cost,
    PartialCostFunction.T:  cost_functions.time_cost,
    PartialCostFunction.ID:  cost_functions.inverse_duration_cost,
}


cost_function_mapping =\
    {
        CostFunction.JB1: [
            (PartialCostFunction.T, 1.0)
        ],
        CostFunction.MW1: [
            (PartialCostFunction.Jlat, 5.0),
            (PartialCostFunction.Jlon, 0.5),
            (PartialCostFunction.Vlon, 0.2),
            (PartialCostFunction.ID, 1.0)
        ],
        CostFunction.SA1: [
            (PartialCostFunction.SA, 0.1),
            (PartialCostFunction.SR, 0.1),
            (PartialCostFunction.D, 100000.0),
        ],
        CostFunction.SM1: [
            (PartialCostFunction.A, 50.0),
            (PartialCostFunction.SA, 50.0),
            (PartialCostFunction.SR, 50.0),
            (PartialCostFunction.L, 1.0),
            (PartialCostFunction.V, 20.0),
            (PartialCostFunction.O, 50.0),
        ],
        CostFunction.SM2: [
            (PartialCostFunction.A, 50.0),
            (PartialCostFunction.SA, 50.0),
            (PartialCostFunction.SR, 50.0),
            (PartialCostFunction.L, 1.0),
            (PartialCostFunction.O, 50.0),
        ],
        CostFunction.SM3: [
            (PartialCostFunction.A, 50.0),
            (PartialCostFunction.SA, 50.0),
            (PartialCostFunction.SR, 50.0),
            (PartialCostFunction.V, 20.0),
            (PartialCostFunction.O, 50.0),
        ],
        CostFunction.WX1: [
            (PartialCostFunction.T, 10.0),
            (PartialCostFunction.V, 1.0),
            (PartialCostFunction.A, 0.1),
            (PartialCostFunction.J, 0.1),
            (PartialCostFunction.D, 0.1),
            (PartialCostFunction.L, 10.0),
        ],
        CostFunction.TR1: [
            (PartialCostFunction.Jlon, 0.01),
            (PartialCostFunction.SR, 22),
            (PartialCostFunction.D, 8),
            (PartialCostFunction.LC, 0.5),
        ],
        CostFunction.ST: [
            (PartialCostFunction.A, 1),
            (PartialCostFunction.J, 1),
            (PartialCostFunction.SA, 1),
            (PartialCostFunction.SR, 1),
            (PartialCostFunction.Y, 1),
            (PartialCostFunction.V, 10),
            (PartialCostFunction.D, 1),
            (PartialCostFunction.T, 100),
        ]
    }


# additional attributes that need to be computed before evaluation
required_properties = {
    PartialCostFunction.A: [],
    PartialCostFunction.J: [],
    PartialCostFunction.Jlat: [SolutionProperties.LatJerk],
    PartialCostFunction.Jlon: [SolutionProperties.LonJerk],
    PartialCostFunction.SA: [],
    PartialCostFunction.SR: [],
    PartialCostFunction.Y: [],
    PartialCostFunction.LC: [],
    PartialCostFunction.V: [],
    PartialCostFunction.Vlon: [SolutionProperties.LonVelocity, SolutionProperties.AllowedVelocityInterval],
    PartialCostFunction.O: [],
    PartialCostFunction.D: [SolutionProperties.LonDistanceObstacles],
    PartialCostFunction.L: [],
    PartialCostFunction.T: [],
    PartialCostFunction.ID: [], }


class CostFunctionEvaluator:
    def __init__(self, cost_function_id: CostFunction, vehicle_type: VehicleType):
        self.cost_function_id: CostFunction = cost_function_id
        self.vehicle_type = vehicle_type
        self.partial_cost_funcs: List[Tuple[PartialCostFunction,float]] = cost_function_mapping[self.cost_function_id]

    @classmethod
    def init_from_solution(cls, solution: Solution):
        return cls(cost_function_id=CostFunction[solution.cost_ids[0]],
                   vehicle_type=VehicleType(int(solution.vehicle_ids[0][-1])))

    @property
    def required_properties(self):
        return list(itertools.chain.from_iterable(required_properties[p] for p, _ in self.partial_cost_funcs))

    def evaluate_pp_solution(self, cr_scenario: Scenario, cr_pproblem: PlanningProblem, trajectory: Trajectory,
            draw_lanelet_path=False, debug_plot=False):
        """
        Computes costs of one solution for cr_pproblem.

        :param cr_scenario: scenario
        :param cr_pproblem: planning problem that is solved by trajectory
        :param trajectory: solution trajectory
        :param draw_lanelet_path: optionally visualize the detected lanelet path with respect to whose some parameters for the cost computation are determined (only useful for development).
        :param debug_plot: show plot in case a trajectory cannot be transformed to curvilinear coordinates.
        :return: result of evaluation
        """
        evaluation_result = PlanningProblemCostResult(cost_function_id=self.cost_function_id,
                                                      solution_id=cr_pproblem.planning_problem_id)
        lm = LaneletRouteMatcher(cr_scenario, self.vehicle_type)
        trajectory, _, properties = lm.compute_curvilinear_coordinates(trajectory,
                                                                       required_properties=self.required_properties,
                                                                       draw_lanelet_path=draw_lanelet_path,
                                                                       debug_plot=debug_plot)
        for pcf, weight in self.partial_cost_funcs:
            pcf_func = PartialCostFunctionMapping[pcf]
            evaluation_result.add_partial_costs(pcf, pcf_func(cr_scenario, cr_pproblem, trajectory, properties), weight)

        return evaluation_result

    def evaluate_solution(self, scenario: Scenario, cr_pproblems: PlanningProblemSet, solution: Solution)\
            -> "SolutionResult":
        """
        Computes costs for all solutions of a planning problem set.

        :param scenario: scenario that was solved
        :param cr_pproblems: planning problem set that was solved
        :param solution: Solution object that contains trajectories
        :return: SolutionResult object that contains partial and total costs
        """
        results = SolutionResult(benchmark_id=solution.benchmark_id)
        for pps in solution.planning_problem_solutions:
            results.add_results(
                    self.evaluate_pp_solution(scenario, cr_pproblems.planning_problem_dict[pps.planning_problem_id],
                                              pps.trajectory, False))

        return results


class PlanningProblemCostResult:
    def __init__(self, cost_function_id: CostFunction, solution_id: int):
        """
        Contains results of a single solution of a planning problem.

        """
        self.cost_function_id = cost_function_id
        self.partial_costs: Dict[PartialCostFunction, float] = {}
        self.weights: Dict[PartialCostFunction, float] = {}
        self.solution_id: int = solution_id

    @property
    def total_costs(self) -> float:
        c = 0.0
        for pcf, cost in self.partial_costs.items():
            c += cost * self.weights[pcf]

        return c

    def add_partial_costs(self, pcf: PartialCostFunction, cost: float, weight):
        self.partial_costs[pcf] = cost
        self.weights[pcf] = weight

    def __str__(self):
        nl = "\n"
        t = "\t"
        return f"Partial costs for solution of planning problem {self.solution_id}:\n" \
               f"{nl.join([p.name + ':' + t + str(self.weights[p] * c) for p, c in self.partial_costs.items()])}"


class SolutionResult:
    def __init__(self, benchmark_id: str, pp_results: List[PlanningProblemCostResult] = ()):
        """
        Contains results of all solutions of a planning problem set.

        """
        self.benchmark_id: str = benchmark_id
        self.total_costs: float = 0.0
        self.pp_results: Dict[int, PlanningProblemCostResult] = {}
        for r in pp_results:
            self.add_results(r)

    def add_results(self, pp_result: PlanningProblemCostResult):
        self.pp_results[pp_result.solution_id] = pp_result
        self.total_costs += pp_result.total_costs

    def __str__(self):
        nl = "\n\t"
        return f"Total costs for benchmark {self.benchmark_id}:\n" \
               f"{self.total_costs}\n" \
               f"{nl.join(str(pr) for pr in self.pp_results.values())}"
