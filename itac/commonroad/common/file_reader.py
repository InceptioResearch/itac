from collections import defaultdict
from typing import Dict
from xml.etree import ElementTree
from abc import ABC

from commonroad import SUPPORTED_COMMONROAD_VERSIONS
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle, Circle, Polygon, ShapeGroup, Shape
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.prediction.prediction import Occupancy, SetBasedPrediction, TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LineMarking, LaneletType, RoadUser, StopLine
from commonroad.scenario.obstacle import ObstacleType, StaticObstacle, DynamicObstacle, Obstacle, EnvironmentObstacle, \
    SignalState, PhantomObstacle
from commonroad.scenario.scenario import Scenario, Tag, GeoTransformation, Location, Environment, Time, \
    TimeOfDay, Weather, Underground, ScenarioID
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.traffic_sign import *
from commonroad.scenario.intersection import Intersection, IntersectionIncomingElement


__author__ = "Stefanie Manzinger, Sebastian Maierhofer"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles", "CAR@TUM"]
__version__ = "2022.1"
__maintainer__ = "Stefanie Manzinger, Sebastian Maierhofer"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


def read_value_exact_or_interval(xml_node: ElementTree.Element)\
        -> Union[float, Interval]:
    if xml_node.find('exact') is not None:
        value = float(xml_node.find('exact').text)
    elif xml_node.find('intervalStart') is not None \
            and xml_node.find('intervalEnd') is not None:
        value = Interval(
            float(xml_node.find('intervalStart').text),
            float(xml_node.find('intervalEnd').text))
    else:
        raise Exception()
    return value


def read_time(xml_node: ElementTree.Element) -> Union[int, Interval]:
    if xml_node.find('exact') is not None:
        value = int(xml_node.find('exact').text)
    elif xml_node.find('intervalStart') is not None \
            and xml_node.find('intervalEnd') is not None:
        value = Interval(
            int(xml_node.find('intervalStart').text),
            int(xml_node.find('intervalEnd').text))
    else:
        raise Exception()
    return value


class CommonRoadFileReader:
    """ Class which reads CommonRoad XML-files. The XML-files are composed of
    (1) a formal representation of the road network,
    (2) static and dynamic obstacles,
    (3) the planning problem of the ego vehicle(s). """
    def __init__(self, filename: str):
        """
        :param filename: full path + filename of the CommonRoad XML-file,
        """
        self._filename = filename
        self._tree = None
        self._dt = None
        self._benchmark_id = None
        self._meta_data = None

    def open(self, lanelet_assignment: bool = False) -> Tuple[Scenario, PlanningProblemSet]:
        """
        Reads a CommonRoad XML-file.

        :param lanelet_assignment: activates calculation of lanelets occupied by obstacles
        :return: the scenario containing the road network and the obstacles and the planning problem set \
        containing the planning problems---initial states and goal regions--for all ego vehicles.
        """
        self._read_header()
        scenario = self._open_scenario(lanelet_assignment)
        planning_problem_set = self._open_planning_problem_set(scenario.lanelet_network)
        return scenario, planning_problem_set

    def open_lanelet_network(self) -> LaneletNetwork:
        """
        Reads the lanelet network of a CommonRoad XML-file.

        :return: object of class LaneletNetwork
        """
        self._read_header()
        return LaneletNetworkFactory.create_from_xml_node(self._tree)

    def _open_scenario(self, lanelet_assignment: bool) -> Scenario:
        """
        Reads the lanelet network and obstacles from the CommonRoad XML-file.

        :param lanelet_assignment: activates calculation of lanelets occupied by obstacles
        :return: object of class scenario containing the road network and the obstacles
        """
        scenario = ScenarioFactory.create_from_xml_node(self._tree, self._dt, self._benchmark_id,
                                                        self._commonroad_version, self._meta_data, lanelet_assignment)
        return scenario

    def _open_planning_problem_set(self, lanelet_network: LaneletNetwork) \
            -> PlanningProblemSet:
        """
        Reads all planning problems from the CommonRoad XML-file.

        :return: object of class PlanningProblemSet containing the planning problems for all ego vehicles.
        """
        planning_problem_set = PlanningProblemSetFactory.create_from_xml_node(
            self._tree, lanelet_network)
        return planning_problem_set

    def _read_header(self):
        """ Parses the CommonRoad XML-file into element tree; reads the global time step size of the time-discrete
        scenario and the CommonRoad benchmark ID."""
        self._parse_file()
        commonroad_version = self._get_commonroad_version()
        assert commonroad_version in SUPPORTED_COMMONROAD_VERSIONS, '<CommonRoadFileReader/_read_header>: ' \
                                                                    'CommonRoad version of XML-file {} is not ' \
                                                                    'supported. Supported versions: {}. Got ' \
                                                                    'version: {}.'.format(self._filename,
                                                                                          SUPPORTED_COMMONROAD_VERSIONS,
                                                                                          commonroad_version)
        self._dt = self._get_dt()
        self._benchmark_id = self._get_benchmark_id()
        self._commonroad_version = commonroad_version
        if commonroad_version == '2018b':
            self._meta_data = {'author': self._get_author(),
                               'affiliation': self._get_affiliation(),
                               'source': self._get_source(),
                               'tags': self._get_tags(),
                               'location': Location()}
        else:
            self._meta_data = {'author': self._get_author(),
                               'affiliation': self._get_affiliation(),
                               'source': self._get_source()}

    def _parse_file(self):
        """ Parses the CommonRoad XML-file into element tree."""
        self._tree = ElementTree.parse(self._filename)

    def _get_dt(self) -> float:
        """ Reads the time step size of the time-discrete scenario."""
        return float(self._tree.getroot().get('timeStepSize'))

    def _get_benchmark_id(self) -> str:
        """ Reads the unique CommonRoad benchmark ID of the scenario."""
        return self._tree.getroot().get('benchmarkID')

    def _get_commonroad_version(self) -> str:
        """ Reads the CommonRoad version of the XML-file."""
        return self._tree.getroot().get('commonRoadVersion')

    def _get_author(self) -> str:
        """ Reads the author of the scenario."""
        return self._tree.getroot().get('author')

    def _get_affiliation(self) -> str:
        """ Reads the affiliation of the author of the scenario."""
        return self._tree.getroot().get('affiliation')

    def _get_source(self) -> str:
        """ Reads the source of the scenario."""
        return self._tree.getroot().get('source')

    def _get_tags(self) -> Set[Tag]:
        """ Reads the tags of the scenario."""
        tags_string = self._tree.getroot().get('tags')
        splits = tags_string.split()
        tags = set()
        for tag in splits:
            try:
                tags.add(Tag(tag))
            except ValueError:
                warnings.warn('Scenario tag \'{}\' not valid.'.format(tag), stacklevel=2)

        return tags


class ScenarioFactory:
    """ Class to create an object of class Scenario from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, dt: float, benchmark_id: str, commonroad_version: str,
                             meta_data: dict, lanelet_assignment: bool):
        """
        :param xml_node: XML element
        :param dt: time step size of the scenario
        :param benchmark_id: unique CommonRoad benchmark ID
        :param commonroad_version: CommonRoad version of the file
        :param lanelet_assignment: activates calculation of lanelets occupied by obstacles
        :return: CommonRoad scenario
        """
        if commonroad_version != '2018b':
            meta_data["tags"] = TagsFactory.create_from_xml_node(xml_node)
            meta_data["location"] = LocationFactory.create_from_xml_node(xml_node)
        else:
            LaneletFactory._speed_limits = {}

        scenario_id = ScenarioID.from_benchmark_id(benchmark_id, commonroad_version)
        scenario = Scenario(dt, scenario_id, **meta_data)

        scenario.add_objects(LaneletNetworkFactory.create_from_xml_node(xml_node))
        if commonroad_version == '2018b':
            large_num = 10000
            scenario.add_objects(cls._obstacles_2018b(xml_node, scenario.lanelet_network, lanelet_assignment))
            for key, value in LaneletFactory._speed_limits.items():
                for lanelet in value:
                    if SupportedTrafficSignCountry.GERMANY.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, [str(key)])
                    elif SupportedTrafficSignCountry.USA.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDUsa.MAX_SPEED, [str(key)])
                    elif SupportedTrafficSignCountry.CHINA.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDChina.MAX_SPEED, [str(key)])
                    elif SupportedTrafficSignCountry.SPAIN.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDSpain.MAX_SPEED, [str(key)])
                    elif SupportedTrafficSignCountry.RUSSIA.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDRussia.MAX_SPEED, [str(key)])
                    elif SupportedTrafficSignCountry.ZAMUNDA.value == scenario_id.country_id:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDZamunda.MAX_SPEED, [str(key)])
                    else:
                        traffic_sign_element = TrafficSignElement(TrafficSignIDZamunda.MAX_SPEED, [str(key)])
                        warnings.warn("Unknown country: Default traffic sign IDs are used.")
                    traffic_sign = TrafficSign(scenario.generate_object_id() + large_num, [traffic_sign_element],
                                               {lanelet},
                                               scenario.lanelet_network.find_lanelet_by_id(lanelet).right_vertices[0])
                    scenario.add_objects(traffic_sign, {lanelet})
            LaneletFactory._speed_limits = {}
        else:
            scenario.add_objects(cls._obstacles(xml_node, scenario.lanelet_network, lanelet_assignment))

        return scenario

    @classmethod
    def _obstacles_2018b(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork,
                         lanelet_assignment: bool) -> List[Obstacle]:
        """
        Reads all obstacles specified in a CommonRoad XML-file.
        :param xml_node: XML element
        :param dt: time step size of the scenario
        :param lanelet_assignment: activates calculation of lanelets occupied by obstacles
        :return: list of static and dynamic obstacles specified in the CommonRoad XML-file
        """
        obstacles = list()
        for o in xml_node.findall('obstacle'):
            if o.find('role').text == 'static':
                obstacles.append(StaticObstacleFactory.create_from_xml_node(o, lanelet_network, lanelet_assignment))
            elif o.find('role').text == 'dynamic':
                obstacles.append(DynamicObstacleFactory.create_from_xml_node(o, lanelet_network, lanelet_assignment))
            else:
                raise ValueError('Role of obstacle is unknown. Got role: {}'.format(xml_node.find('role').text))
        return obstacles

    @classmethod
    def _obstacles(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork,
                   lanelet_assignment: bool) -> List[Obstacle]:
        """
        Reads all obstacles specified in a CommonRoad XML-file.
        :param xml_node: XML element
        :param dt: time step size of the scenario
        :param lanelet_assignment: activates calculation of lanelets occupied by obstacles
        :return: list of static and dynamic obstacles specified in the CommonRoad XML-file
        """
        obstacles = []
        for o in xml_node.findall('staticObstacle'):
            obstacles.append(StaticObstacleFactory.create_from_xml_node(o, lanelet_network, lanelet_assignment))
        for o in xml_node.findall('dynamicObstacle'):
            obstacles.append(DynamicObstacleFactory.create_from_xml_node(o, lanelet_network, lanelet_assignment))
        for o in xml_node.findall('environmentObstacle'):
            obstacles.append(EnvironmentObstacleFactory.create_from_xml_node(o))
        for o in xml_node.findall('phantomObstacle'):
            obstacles.append(PhantomObstacleFactory.create_from_xml_node(o))
        return obstacles


class TagsFactory:
    """ Class to create a tag set from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Set[Tag]:
        """
        :param xml_node: XML element
        :return: set of tags
        """
        tags = set()
        tag_element = xml_node.find('scenarioTags')
        for elem in Tag:
            if tag_element.find(elem.value) is not None:
                tags.add(elem)
        return tags


class LocationFactory:
    """ Class to create a location from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Union[Location, None]:
        """
        :param xml_node: XML element
        :return: location object
        """
        if xml_node.find('location') is not None:
            location_element = xml_node.find('location')
            geo_name_id = int(location_element.find('geoNameId').text)
            gps_latitude = float(location_element.find('gpsLatitude').text)
            gps_longitude = float(location_element.find('gpsLongitude').text)
            if location_element.find('geoTransformation') is not None:
                geo_transformation = GeoTransformationFactory.create_from_xml_node(
                    location_element.find('geoTransformation'))
            else:
                geo_transformation = None
            if location_element.find('environment') is not None:
                environment = EnvironmentFactory.create_from_xml_node(
                    location_element.find('environment'))
            else:
                environment = None

            return Location(geo_name_id, gps_latitude, gps_longitude, geo_transformation, environment)
        else:
            return None


class GeoTransformationFactory:
    """ Class to create a geotransformation object of an XML element according to the CommonRoad specification."""

    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> GeoTransformation:
        """
        :param xml_node: XML element
        :return: GeoTransformation object
        """
        geo_reference = xml_node.find('geoReference').text
        if xml_node.find('additionalTransformation') is not None:
            add_trans_node = xml_node.find('additionalTransformation')
            x_translation = float(add_trans_node.find('xTranslation').text)
            y_translation = float(add_trans_node.find('yTranslation').text)
            z_rotation = float(add_trans_node.find('zRotation').text)
            scaling = float(add_trans_node.find('scaling').text)
            return GeoTransformation(geo_reference, x_translation, y_translation, z_rotation, scaling)
        else:
            return GeoTransformation(geo_reference)


class EnvironmentFactory:
    """ Class to create a environment object of an XML element according to the CommonRoad specification."""

    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Environment:
        """
        :param xml_node: XML element
        :return: Environment object
        """
        time = TimeFactory.create_from_xml_node(xml_node.find('time').text)
        weather = Weather(xml_node.find('weather').text)
        underground = Underground(xml_node.find('underground').text)
        time_of_day = TimeOfDay(xml_node.find('timeOfDay').text)

        return Environment(time, time_of_day, weather, underground)


class TimeFactory:
    """ Class to create a time object of an XML element."""

    @classmethod
    def create_from_xml_node(cls, time_text: str) -> Time:
        """
        :param time_text: time as string
        :return: time object
        """
        hours = int(time_text[0:2])
        minutes = int(time_text[3:5])

        return Time(hours, minutes)


class LaneletNetworkFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> LaneletNetwork:
        """
        Reads all lanelets specified in a CommonRoad XML-file.
        :param xml_node: XML element
        :return: list of lanelets
        """
        lanelets = []
        for lanelet_node in xml_node.findall('lanelet'):
            lanelets.append(LaneletFactory.create_from_xml_node(lanelet_node))
        lanelet_network = LaneletNetwork.create_from_lanelet_list(lanelets)

        country = cls._find_country( xml_node)
        first_traffic_sign_occurence = cls._find_first_traffic_sign_occurence(lanelet_network)
        for traffic_sign_node in xml_node.findall('trafficSign'):
            lanelet_network.add_traffic_sign(TrafficSignFactory.create_from_xml_node(traffic_sign_node, country,
                                                                                     first_traffic_sign_occurence,
                                                                                     lanelet_network), [])

        for traffic_light_node in xml_node.findall('trafficLight'):
            lanelet_network.add_traffic_light(TrafficLightFactory.create_from_xml_node(traffic_light_node, country,
                                                                                       lanelet_network), [],)

        for intersection_node in xml_node.findall('intersection'):
            lanelet_network.add_intersection(IntersectionFactory.create_from_xml_node(intersection_node))

        return lanelet_network

    @staticmethod
    def _find_first_traffic_sign_occurence(lanelet_network: LaneletNetwork) -> Dict[int, Set[int]]:
        """
        Evaluates all lanelets if a traffic sign occurs first within it
        :param lanelet_network: CommonRoad lanelet network
        :return: list of tuples with traffic sign ID and corresponding lanelet ID
        """
        occurences = {}
        for lanelet in lanelet_network.lanelets:
            for traffic_sign in lanelet.traffic_signs:
                # create set object if none exist
                if occurences.get(traffic_sign) is None:
                    occurences[traffic_sign] = set()
                # if there exists no predecessor, current lanelet is first occurence
                if len(lanelet.predecessor) == 0:
                    occurences[traffic_sign].add(lanelet.lanelet_id)
                # if no predecessor references the traffic sign, this is the first occurence
                elif all(traffic_sign not in
                         lanelet_network.find_lanelet_by_id(pre).traffic_signs for pre in lanelet.predecessor):
                    occurences[traffic_sign].add(lanelet.lanelet_id)

        return occurences

    @staticmethod
    def _find_country(xml_node: ElementTree.Element) -> SupportedTrafficSignCountry:
        """
        Extracts country from location element
        :param xml_node: CommonRoad root xml node
        :return: supported traffic sign country enum
        """
        if xml_node._root.attrib["benchmarkID"][:2] == "C-":
            country = xml_node._root.attrib["benchmarkID"][2:5]
        else:
            country = xml_node._root.attrib["benchmarkID"][:3]

        if SupportedTrafficSignCountry.GERMANY.value == country:
            return SupportedTrafficSignCountry.GERMANY
        elif SupportedTrafficSignCountry.USA.value == country:
            return SupportedTrafficSignCountry.USA
        elif SupportedTrafficSignCountry.CHINA.value == country:
            return SupportedTrafficSignCountry.CHINA
        elif SupportedTrafficSignCountry.SPAIN.value == country:
            return SupportedTrafficSignCountry.SPAIN
        elif SupportedTrafficSignCountry.RUSSIA.value == country:
            return SupportedTrafficSignCountry.RUSSIA
        elif SupportedTrafficSignCountry.ARGENTINA.value == country:
            return SupportedTrafficSignCountry.ARGENTINA
        elif SupportedTrafficSignCountry.ITALY.value == country:
            return SupportedTrafficSignCountry.ITALY
        elif SupportedTrafficSignCountry.FRANCE.value == country:
            return SupportedTrafficSignCountry.FRANCE
        elif SupportedTrafficSignCountry.PUERTO_RICO.value == country:
            return SupportedTrafficSignCountry.PUERTO_RICO
        elif SupportedTrafficSignCountry.CROATIA.value == country:
            return SupportedTrafficSignCountry.CROATIA
        elif SupportedTrafficSignCountry.GREECE.value == country:
            return SupportedTrafficSignCountry.GREECE
        elif SupportedTrafficSignCountry.BELGIUM.value == country:
            return SupportedTrafficSignCountry.BELGIUM
        elif SupportedTrafficSignCountry.ZAMUNDA.value == country:
            return SupportedTrafficSignCountry.ZAMUNDA
        else:
            warnings.warn("Unknown country: Default traffic sign IDs are used. Specified country: " + country)
            return SupportedTrafficSignCountry.ZAMUNDA


class LaneletFactory:
    """ Class to create an object of class Lanelet from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Lanelet:
        """
        :param xml_node: XML element
        :return: object of class Lanelet according to the CommonRoad specification.
        """
        lanelet_id = int(xml_node.get('id'))

        left_vertices = cls._vertices(xml_node.find('leftBound'))
        right_vertices = cls._vertices(xml_node.find('rightBound'))
        center_vertices = 0.5 * (left_vertices + right_vertices)

        line_marking_left_vertices = cls._line_marking(xml_node.find('leftBound'))
        line_marking_right_vertices = cls._line_marking(xml_node.find('rightBound'))

        predecessors = cls._predecessors(xml_node)
        successors = cls._successors(xml_node)

        adjacent_left, adjacent_left_same_direction = cls._adjacent_left(xml_node)
        adjacent_right, adjacent_right_same_direction = cls._adjacent_right(xml_node)

        stop_line = cls._stop_line(xml_node, left_vertices[-1], right_vertices[-1])
        lanelet_type = cls._lanelet_type(xml_node)
        user_one_way = cls._user_one_way(xml_node)
        user_bidirectional = cls._user_bidirectional(xml_node)
        traffic_signs = cls._traffic_signs(xml_node)
        traffic_lights = cls._traffic_lights(xml_node)

        if cls._speed_limit_exists(xml_node) is not None:
            speed_limit = cls._speed_limit_exists(xml_node)
            if not hasattr(cls, '_speed_limits'):
                cls._speed_limits = {speed_limit: {lanelet_id}}
            elif cls._speed_limits.get(speed_limit) is not None:
                cls._speed_limits[speed_limit].add(lanelet_id)
            else:
                cls._speed_limits[speed_limit] = {lanelet_id}

        return Lanelet(
            left_vertices=left_vertices, center_vertices=center_vertices, right_vertices=right_vertices,
            lanelet_id=lanelet_id,
            predecessor=predecessors, successor=successors,
            adjacent_left=adjacent_left, adjacent_left_same_direction=adjacent_left_same_direction,
            adjacent_right=adjacent_right, adjacent_right_same_direction=adjacent_right_same_direction,
            line_marking_left_vertices=line_marking_left_vertices,
            line_marking_right_vertices=line_marking_right_vertices, stop_line=stop_line,
            lanelet_type=lanelet_type, user_one_way=user_one_way, user_bidirectional=user_bidirectional,
            traffic_signs=traffic_signs, traffic_lights=traffic_lights)

    @classmethod
    def _vertices(cls, xml_node: ElementTree.Element) -> np.ndarray:
        """
        Reads the vertices of the lanelet boundary.
        :param xml_node: XML element
        :return: The vertices of the boundary of the Lanelet described as a polyline
        """
        return PointListFactory.create_from_xml_node(xml_node)

    @classmethod
    def _predecessors(cls, xml_node: ElementTree.Element) -> List[int]:
        """
        Reads all predecessor lanelets.
        :param xml_node: XML element
        :return: list of IDs of all predecessor lanelets
        """
        predecessors = list()
        for l in xml_node.findall('predecessor'):
            predecessors.append(int(l.get('ref')))
        return predecessors

    @classmethod
    def _successors(cls, xml_node: ElementTree.Element) -> List[int]:
        """
        Reads all successor lanelets.
        :param xml_node: XML element
        :return: list of IDs of all successor lanelets
        """
        successors = list()
        for l in xml_node.findall('successor'):
            successors.append(int(l.get('ref')))
        return successors

    @classmethod
    def _adjacent_left(cls, xml_node: ElementTree.Element) -> Tuple[Union[int, None], Union[bool, None]]:
        """
        Reads the ID and the driving direction of a neighboring lanelet which is adjacent to the left.
        :param xml_node: XML element
        :return: the ID of the lanelet which is adjacent to the left (None if not existing);
                 the driving direction of the neighboring lanelet (None if not existing)
        """
        adjacent_left = None
        adjacent_left_same_direction = None
        if xml_node.find('adjacentLeft') is not None:
            adjacent_left = int(xml_node.find('adjacentLeft').get('ref'))
            if xml_node.find('adjacentLeft').get('drivingDir') == 'same':
                adjacent_left_same_direction = True
            else:
                adjacent_left_same_direction = False
        return adjacent_left, adjacent_left_same_direction

    @classmethod
    def _adjacent_right(cls, xml_node: ElementTree.Element) -> Tuple[Union[int, None], Union[bool, None]]:
        """
        Reads the ID and the driving direction of a neighboring lanelet which is adjacent to the right.
        :param xml_node: XML element
        :return: the ID of the lanelet which is adjacent to the right (None if not existing);
                 the driving direction of the neighboring lanelet (None if not existing)
        """
        adjacent_right = None
        adjacent_right_same_direction = None
        if xml_node.find('adjacentRight') is not None:
            adjacent_right = int(xml_node.find('adjacentRight').get('ref'))
            if xml_node.find('adjacentRight').get('drivingDir') == 'same':
                adjacent_right_same_direction = True
            else:
                adjacent_right_same_direction = False
        return adjacent_right, adjacent_right_same_direction

    @classmethod
    def _speed_limit_exists(cls, xml_node: ElementTree.Element) -> Union[float, None]:
        """
        Evaluates if a speed limit element from a previous CommonRoad version exists
        :param xml_node: XML element
        :return: boolean indicating if speed limit element exists
        """
        speed_limit = None
        if xml_node.find('speedLimit') is not None:
            speed_limit = float(xml_node.find('speedLimit').text)
            return speed_limit
        return speed_limit

    @classmethod
    def _lanelet_type(cls, xml_node: ElementTree.Element) -> Union[Set[LaneletType], None]:
        """
        Reads the lanelet types of the lanelet.

        :param xml_node: XML element
        :return: set of lanelet types for a lanelet
        """
        lanelet_types = set()
        for l_type in xml_node.findall('laneletType'):
            if LaneletType(l_type.text) is not None:
                lanelet_types.add(LaneletType(l_type.text))
            else:
                raise ValueError('<LaneletFactory/_lanelet_type>: Unkown type of lanelet: %s.' % l_type.text)
        return lanelet_types

    @classmethod
    def _user_one_way(cls, xml_node: ElementTree.Element) -> Union[Set[RoadUser], None]:
        """
        Reads the one way users of the lanelet.

        :param xml_node: XML element
        :return: set of allowed road users driving along the driving direction of a lanelet
        """
        users_one_way = set()
        for user in xml_node.findall('userOneWay'):
            if RoadUser(user.text) is not None:
                users_one_way.add(RoadUser(user.text))
            else:
                raise ValueError('<LaneletFactory/_user_one_way>: Unkown type of road user on lanelet: %s.' % user.text)

        return users_one_way

    @classmethod
    def _user_bidirectional(cls, xml_node: ElementTree.Element) -> Union[Set[RoadUser], None]:
        """
        Reads bidirectional users of the lanelet.

        :param xml_node: XML element
        :return: set of allowed road users driving in both driving directions of a lanelet
        """
        users_bidirectional = set()
        for user in xml_node.findall('userBidirectional'):
            if RoadUser(user.text) is not None:
                users_bidirectional.add(RoadUser(user.text))
            else:
                raise ValueError('<LaneletFactory/_user_bidirectional>: Unkown type of road user on lanelet: %s.'
                                 % user.text)
        return users_bidirectional

    @classmethod
    def _stop_line(cls, xml_node: ElementTree.Element, left_lanelet_end, right_lanelet_end) -> Union[StopLine, None]:
        """
        Reads the stop line of the lanelet.

        :param xml_node: XML element
        :return: stop line element of lanelet (None if not specified)
        """
        stop_line = None
        traffic_sign_ref = None
        traffic_light_ref = None
        if xml_node.find('stopLine') is not None:
            points = PointListFactory.create_from_xml_node(xml_node.find('stopLine'))
            line_marking = LineMarking(xml_node.find('stopLine').find('lineMarking').text)
            if xml_node.find('stopLine').find('trafficSignRef') is not None:
                traffic_sign_ref = set()
                for sign in xml_node.find('stopLine').findall('trafficSignRef'):
                    traffic_sign_ref.add(int(sign.get('ref')))
            if xml_node.find('stopLine').find('trafficLightRef') is not None:
                traffic_light_ref = set()
                for light in xml_node.find('stopLine').findall('trafficLightRef'):
                    traffic_light_ref.add(int(light.get('ref')))
            if len(points) > 0:
                stop_line = StopLine(start=points[0], end=points[1], line_marking=line_marking,
                                     traffic_sign_ref=traffic_sign_ref, traffic_light_ref=traffic_light_ref)
            else:
                stop_line = StopLine(start=left_lanelet_end, end=right_lanelet_end, line_marking=line_marking,
                                     traffic_sign_ref=traffic_sign_ref, traffic_light_ref=traffic_light_ref)

        return stop_line

    @classmethod
    def _line_marking(cls, xml_node: ElementTree.Element) -> Union[None, LineMarking]:
        """
        Reads the line marking of the left or right lanelet boundary.
        :param xml_node: XML element
        :return: the type of the line marking of the lanelet boundary (None if not specified).
        """
        line_marking = LineMarking.UNKNOWN
        if xml_node.find('lineMarking') is not None:
            if LineMarking(xml_node.find('lineMarking').text) is not None:
                line_marking = LineMarking(xml_node.find('lineMarking').text)
            else:
                raise ValueError('<LaneletFactory/_line_marking>: Unkown type of line marking: %s.'
                                 % line_marking)
        return line_marking

    @classmethod
    def _traffic_signs(cls, xml_node: ElementTree.Element) -> Union[Set[int], None]:
        """
        Reads the traffic sign references of the lanelet.

        :param xml_node: XML element
        :return: set of traffic sign IDs (None if not specified).
        """
        traffic_signs = set()
        for traffic_sign_ref in xml_node.findall('trafficSignRef'):
            if traffic_sign_ref.get("ref") is not None:
                traffic_signs.add(int(traffic_sign_ref.get("ref")))
            else:
                raise ValueError('<LaneletFactory/_traffic_signs>: Unknown type of traffic sign reference: %s.'
                                 % traffic_sign_ref.get("ref"))
        return traffic_signs

    @classmethod
    def _traffic_lights(cls, xml_node: ElementTree.Element) -> Union[Set[int], None]:
        """
        Reads the traffic sign references of the lanelet.

        :param xml_node: XML element
        :return: set of traffic light IDs (None if not specified).
        """
        traffic_lights = set()
        for traffic_light_ref in xml_node.findall('trafficLightRef'):
            if traffic_light_ref.get("ref") is not None:
                traffic_lights.add(int(traffic_light_ref.get("ref")))
            else:
                raise ValueError('<LaneletFactory/_traffic_signs>: Unkown type of traffic light reference: %s.'
                                 % traffic_light_ref.get("ref"))
        return traffic_lights


class TrafficSignFactory:
    """ Class to create an object of class TrafficSign from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, country: SupportedTrafficSignCountry,
                             first_traffic_sign_occurence: Dict[int, Set[int]],
                             lanelet_network: LaneletNetwork) -> TrafficSign:
        """
        :param xml_node: XML element
        :param country: country where traffic sign stands
        :param first_traffic_sign_occurence: set of first occurences of traffic sign
        :param lanelet_network: CommonRoad lanelet network
        :return: object of class TrafficSign according to the CommonRoad specification.
        """
        traffic_sign_id = int(xml_node.get('id'))
        assert traffic_sign_id in first_traffic_sign_occurence.keys(), \
            '<CommonRoadFileReader/TrafficSignFactory.create_from_xml_node>: ' \
            'CommonRoad file is invalid! Traffic sign {} is not referenced by a lanelet!'.format(traffic_sign_id)

        traffic_sign_elements = []
        for element in xml_node.findall('trafficSignElement'):
            traffic_sign_elements.append(TrafficSignElementFactory.create_from_xml_node(element, country))

        if xml_node.find('position') is not None:
            position = PointFactory.create_from_xml_node(xml_node.find('position').find('point'))
        else:
            # traffic signs are always placed on right side of road if no position is given (for right-hand traffic)
            position = None
            for lanelet_id in first_traffic_sign_occurence.get(traffic_sign_id):
                if country.value in LEFT_HAND_TRAFFIC:
                    if lanelet_network.find_lanelet_by_id(lanelet_id).adj_left_same_direction is None or \
                            lanelet_network.find_lanelet_by_id(lanelet_id).adj_left_same_direction is False:
                        if any(element.traffic_sign_element_id.name in TRAFFIC_SIGN_VALIDITY_START
                               for element in traffic_sign_elements):
                            position = lanelet_network.find_lanelet_by_id(lanelet_id).left_vertices[0]
                        else:
                            position = lanelet_network.find_lanelet_by_id(lanelet_id).left_vertices[-1]
                else:
                    if lanelet_network.find_lanelet_by_id(lanelet_id).adj_right_same_direction is None \
                            or lanelet_network.find_lanelet_by_id(lanelet_id).adj_right_same_direction is False:
                        if any(element.traffic_sign_element_id.name in TRAFFIC_SIGN_VALIDITY_START
                               for element in traffic_sign_elements):
                            position = lanelet_network.find_lanelet_by_id(lanelet_id).right_vertices[0]
                        else:
                            position = lanelet_network.find_lanelet_by_id(lanelet_id).right_vertices[-1]
            if position is None:
                current_lanelet =\
                    lanelet_network.find_lanelet_by_id(list(first_traffic_sign_occurence.get(traffic_sign_id))[0])
                if country.value in LEFT_HAND_TRAFFIC:
                    while current_lanelet.adj_left_same_direction is not None \
                            and current_lanelet.adj_left_same_direction is not False:
                        current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
                    if any(element.traffic_sign_element_id.name in TRAFFIC_SIGN_VALIDITY_START for element in
                           traffic_sign_elements):
                        position = current_lanelet.left_vertices[0]
                    else:
                        position = current_lanelet.left_vertices[-1]
                else:
                    while current_lanelet.adj_right_same_direction is not None \
                            and current_lanelet.adj_right_same_direction is not False:
                        current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_right)
                    if any(element.traffic_sign_element_id.name in TRAFFIC_SIGN_VALIDITY_START for element in
                           traffic_sign_elements):
                        position = current_lanelet.right_vertices[0]
                    else:
                        position = current_lanelet.right_vertices[-1]

        if xml_node.get('virtual') is not None:
            if xml_node.get('virtual').text == "true":
                virtual = True
            elif xml_node.get('virtual').text == "false":
                virtual = False
            else:
                raise ValueError()
        else:
            virtual = False

        return TrafficSign(traffic_sign_id=traffic_sign_id, position=position,
                           first_occurrence=first_traffic_sign_occurence[traffic_sign_id],
                           traffic_sign_elements=traffic_sign_elements, virtual=virtual)


class TrafficSignElementFactory:
    """ Class to create an object of class TrafficSignElement from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element,
                             country: SupportedTrafficSignCountry) -> TrafficSignElement:
        """
        :param xml_node: XML element
        :param country: country where traffic sign stands
        :return: object of class TrafficSignElement according to the CommonRoad specification.
        """
        try:
            if country is SupportedTrafficSignCountry.GERMANY:
                traffic_sign_element_id = TrafficSignIDGermany(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.ZAMUNDA:
                traffic_sign_element_id = TrafficSignIDZamunda(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.USA:
                traffic_sign_element_id = TrafficSignIDUsa(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.CHINA:
                traffic_sign_element_id = TrafficSignIDChina(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.SPAIN:
                traffic_sign_element_id = TrafficSignIDSpain(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.RUSSIA:
                traffic_sign_element_id = TrafficSignIDRussia(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.ARGENTINA:
                traffic_sign_element_id = TrafficSignIDArgentina(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.ITALY:
                traffic_sign_element_id = TrafficSignIDItaly(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.FRANCE:
                traffic_sign_element_id = TrafficSignIDFrance(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.PUERTO_RICO:
                traffic_sign_element_id = TrafficSignIDPuertoRico(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.CROATIA:
                traffic_sign_element_id = TrafficSignIDCroatia(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.GREECE:
                traffic_sign_element_id = TrafficSignIDGreece(xml_node.find('trafficSignID').text)
            elif country is SupportedTrafficSignCountry.BELGIUM:
                traffic_sign_element_id = TrafficSignIDBelgium(xml_node.find('trafficSignID').text)
            else:
                warnings.warn("Unknown country: Default traffic sign ID is used. Specified country: " + country.value)
                traffic_sign_element_id = TrafficSignIDZamunda(xml_node.find('trafficSignID').text)
        except ValueError:
            warnings.warn("<FileReader>: Unknown TrafficElementID! Default traffic sign ID is used. Specified country: "
                          + country.value + " / Specified traffic sign ID: " + xml_node.find('trafficSignID').text)
            traffic_sign_element_id = TrafficSignIDZamunda.UNKNOWN

        additional_values = []
        for additional_value in xml_node.findall('additionalValue'):
            additional_values.append(additional_value.text)

        return TrafficSignElement(traffic_sign_element_id=traffic_sign_element_id,
                                  additional_values=additional_values)


class TrafficLightFactory:
    """ Class to create an object of class TrafficLight from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, country: SupportedTrafficSignCountry,
                             lanelet_network: LaneletNetwork) -> TrafficLight:
        """
        :param xml_node: XML element
        :param country: country where traffic sign stands
        :param lanelet_network: CommonRoad lanelet network
        :return: object of class TrafficLight according to the CommonRoad specification.
        """
        traffic_light_id = int(xml_node.get('id'))

        if xml_node.find('position') is not None:
            position = PointFactory.create_from_xml_node(xml_node.find('position').find('point'))
        else:
            # traffic lights are always placed on right side of road if no position is given (for right-hand traffic)
            current_lanelet = None
            for lanelet in lanelet_network.lanelets:
                if traffic_light_id in lanelet.traffic_lights:
                    current_lanelet = lanelet
                    break
            if current_lanelet is None:
                raise ValueError("Error in xml-file: Traffic Light not referenced by a lanelet")
            if country.value in LEFT_HAND_TRAFFIC:
                while current_lanelet.adj_left_same_direction is not None \
                        and current_lanelet.adj_left_same_direction is not False:
                    current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
                position = current_lanelet.left_vertices[-1]
            else:
                while current_lanelet.adj_right_same_direction is not None \
                        and current_lanelet.adj_right_same_direction is not False:
                    current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_right)
                position = current_lanelet.right_vertices[-1]

        if xml_node.find('active') is not None:
            if xml_node.find('active').text == "true":
                active = True
            elif xml_node.find('active').text == "false":
                active = False
            else:
                active = True
        else:
            active = True

        if xml_node.find('direction') is not None:
            if xml_node.find('direction').text == "right":
                direction = TrafficLightDirection.RIGHT
            elif xml_node.find('direction').text == "straight":
                direction = TrafficLightDirection.STRAIGHT
            elif xml_node.find('direction').text == "left":
                direction = TrafficLightDirection.LEFT
            elif xml_node.find('direction').text == "leftStraight":
                direction = TrafficLightDirection.LEFT_STRAIGHT
            elif xml_node.find('direction').text == "straightRight":
                direction = TrafficLightDirection.STRAIGHT_RIGHT
            elif xml_node.find('direction').text == "leftRight":
                direction = TrafficLightDirection.LEFT_RIGHT
            elif xml_node.find('direction').text == "all":
                direction = TrafficLightDirection.ALL
            else:
                direction = TrafficLightDirection.ALL
        else:
            direction = TrafficLightDirection.ALL

        traffic_ligth_cycle, time_offset = TrafficLightCycleFactory.create_from_xml_node(xml_node.find('cycle'))

        return TrafficLight(traffic_light_id=traffic_light_id, cycle=traffic_ligth_cycle, position=position,
                            direction=direction, active=active, time_offset=time_offset)


class TrafficLightCycleFactory:
    """ Class to create an object of class TrafficLightCycleElement from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Tuple[List[TrafficLightCycleElement], int]:
        """
        :param xml_node: XML element
        :return: list of objects of class TrafficLightCycleElement according to the CommonRoad specification.
        """
        traffic_ligth_cycle_elements = []
        for cycleElement in xml_node.findall('cycleElement'):
            state = cycleElement.find('color').text
            duration = int(cycleElement.find('duration').text)
            traffic_ligth_cycle_elements.append(TrafficLightCycleElement(state=TrafficLightState(state),
                                                                         duration=duration))

        if xml_node.find('timeOffset') is not None:
            time_offset = int(xml_node.find('timeOffset').text)
        else:
            time_offset = 0

        return traffic_ligth_cycle_elements, time_offset


class IntersectionFactory:
    """ Class to create an object of class Intersection from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Intersection:
        """
        :param xml_node: XML element
        :return: object of class Intersection according to the CommonRoad specification.
        """
        intersection_id = int(xml_node.get('id'))
        incomings = []
        for incoming_node in xml_node.findall('incoming'):
            incomings.append(IntersectionIncomingFactory.create_from_xml_node(incoming_node))

        if xml_node.find('crossing') is not None:
            crossings = set()
            for crossing_ref in xml_node.find('crossing').findall('crossingLanelet'):
                crossings.add(int(crossing_ref.get("ref")))
        else:
            crossings = None

        return Intersection(intersection_id=intersection_id, incomings=incomings, crossings=crossings)


class IntersectionIncomingFactory:
    """ Class to create an object of class IntersectionIncomingElement from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> IntersectionIncomingElement:
        """
        :param xml_node: XML element
        :return: object of class IntersectionIncomingElement according to the CommonRoad specification.
        """
        incoming_id = int(xml_node.get('id'))
        incoming_lanelets = set()
        successors_right = set()
        successors_straight = set()
        successors_left = set()
        left_of = None
        for incoming_lanelet_ref in xml_node.findall('incomingLanelet'):
            incoming_lanelets.add(int(incoming_lanelet_ref.get('ref')))
        for successor_right_ref in xml_node.findall('successorsRight'):
            successors_right.add(int(successor_right_ref.get('ref')))
        for successor_straight_ref in xml_node.findall('successorsStraight'):
            successors_straight.add(int(successor_straight_ref.get('ref')))
        for successor_left_ref in xml_node.findall('successorsLeft'):
            successors_left.add(int(successor_left_ref.get('ref')))
        for left_of_ref in xml_node.findall('isLeftOf'):
            left_of = int(left_of_ref.get('ref'))

        return IntersectionIncomingElement(incoming_id=incoming_id, incoming_lanelets=incoming_lanelets,
                                           successors_right=successors_right, successors_straight=successors_straight,
                                           successors_left=successors_left, left_of=left_of)


class ObstacleFactory(ABC):
    @classmethod
    def read_type(cls, xml_node: ElementTree.Element) -> ObstacleType:
        obstacle_type = None
        if xml_node.find('type') is not None:
            if ObstacleType(xml_node.find('type').text) is not None:
                obstacle_type = ObstacleType(xml_node.find('type').text)
            else:
                raise ValueError('Type of obstacle is unknown. Got type: {}'.format(xml_node.find('type').text))

        return obstacle_type

    @classmethod
    def read_id(cls, xml_node: ElementTree.Element) -> int:
        obstacle_id = int(xml_node.get('id'))
        return obstacle_id

    @classmethod
    def read_initial_state(cls, xml_node: ElementTree.Element) -> State:
        initial_state = StateFactory.create_from_xml_node(xml_node)
        return initial_state

    @classmethod
    def read_shape(cls, xml_node: ElementTree.Element) -> Shape:
        shape = ShapeFactory.create_from_xml_node(xml_node)
        return shape

    @classmethod
    def read_initial_signal_state(cls, xml_node: ElementTree.Element) -> SignalState:
        initial_signal_state = SignalStateFactory.create_from_xml_node(xml_node)
        return initial_signal_state


class StaticObstacleFactory(ObstacleFactory):
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork,
                             lanelet_assignment: bool) -> StaticObstacle:
        obstacle_type = StaticObstacleFactory.read_type(xml_node)
        obstacle_id = StaticObstacleFactory.read_id(xml_node)
        initial_state = StaticObstacleFactory.read_initial_state(xml_node.find('initialState'))
        initial_signal_state = StaticObstacleFactory.read_initial_signal_state(xml_node.find('initialSignalState'))
        signal_series = SignalSeriesFactory.create_from_xml_node((xml_node.find('signalSeries')))
        shape = StaticObstacleFactory.read_shape(xml_node.find('shape'))

        if lanelet_assignment is True:
            rotated_shape = shape.rotate_translate_local(initial_state.position, initial_state.orientation)
            initial_shape_lanelet_ids = set(lanelet_network.find_lanelet_by_shape(rotated_shape))
            initial_center_lanelet_ids = set(lanelet_network.find_lanelet_by_position([initial_state.position])[0])
            for l_id in initial_shape_lanelet_ids:
                 lanelet_network.find_lanelet_by_id(l_id).add_static_obstacle_to_lanelet(obstacle_id=obstacle_id)
        else:
            initial_center_lanelet_ids = None
            initial_shape_lanelet_ids = None

        return StaticObstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type,
                              obstacle_shape=shape, initial_state=initial_state,
                              initial_center_lanelet_ids=initial_center_lanelet_ids,
                              initial_shape_lanelet_ids=initial_shape_lanelet_ids,
                              initial_signal_state=initial_signal_state,
                              signal_series=signal_series)


class DynamicObstacleFactory(ObstacleFactory):
    @staticmethod
    def find_obstacle_shape_lanelets(initial_state: State, state_list: List[State], lanelet_network: LaneletNetwork,
                                     obstacle_id: int, shape: Shape) -> Dict[int, Set[int]]:
        """
        Extracts for each shape the corresponding lanelets it is on

        :param initial_state: initial CommonRoad state
        :param state_list: trajectory state list
        :param lanelet_network: CommonRoad lanelet network
        :param obstacle_id: ID of obstacle
        :param shape: shape of obstacle
        :return: list of IDs of all predecessor lanelets
        """
        compl_state_list = [initial_state] + state_list
        lanelet_ids_per_state = {}

        for state in compl_state_list:
            rotated_shape = shape.rotate_translate_local(state.position, state.orientation)
            lanelet_ids = lanelet_network.find_lanelet_by_shape(rotated_shape)
            for l_id in lanelet_ids:
                lanelet_network.find_lanelet_by_id(l_id).add_dynamic_obstacle_to_lanelet(obstacle_id=obstacle_id,
                                                                                         time_step=state.time_step)
            lanelet_ids_per_state[state.time_step] = set(lanelet_ids)

        return lanelet_ids_per_state

    @staticmethod
    def find_obstacle_center_lanelets(initial_state: State, state_list: List[State],
                                      lanelet_network: LaneletNetwork) -> Dict[int, Set[int]]:
        """
        Extracts for each shape the corresponding lanelets it is on

        :param initial_state: initial CommonRoad state
        :param state_list: trajectory state list
        :param lanelet_network: CommonRoad lanelet network
        :return: list of IDs of all predecessor lanelets
        """
        compl_state_list = [initial_state] + state_list
        lanelet_ids_per_state = {}

        for state in compl_state_list:
            lanelet_ids = lanelet_network.find_lanelet_by_position([state.position])[0]
            lanelet_ids_per_state[state.time_step] = set(lanelet_ids)

        return lanelet_ids_per_state

    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork,
                             lanelet_assignment: bool) -> DynamicObstacle:
        obstacle_type = DynamicObstacleFactory.read_type(xml_node)
        obstacle_id = DynamicObstacleFactory.read_id(xml_node)
        shape = DynamicObstacleFactory.read_shape(xml_node.find('shape'))
        initial_state = DynamicObstacleFactory.read_initial_state(xml_node.find('initialState'))
        initial_signal_state = DynamicObstacleFactory.read_initial_signal_state(xml_node.find('initialSignalState'))
        signal_series = SignalSeriesFactory.create_from_xml_node((xml_node.find('signalSeries')))
        initial_center_lanelet_ids = set()
        initial_shape_lanelet_ids = set()

        if xml_node.find('trajectory') is not None:
            if lanelet_assignment is True:
                rotated_shape = shape.rotate_translate_local(initial_state.position, initial_state.orientation)
                initial_shape_lanelet_ids = set(lanelet_network.find_lanelet_by_shape(rotated_shape))
                initial_center_lanelet_ids = set(lanelet_network.find_lanelet_by_position([initial_state.position])[0])
                for l_id in initial_shape_lanelet_ids:
                    lanelet_network.find_lanelet_by_id(l_id).\
                        add_dynamic_obstacle_to_lanelet(obstacle_id=obstacle_id, time_step=initial_state.time_step)
            else:
                initial_shape_lanelet_ids = None
                initial_center_lanelet_ids = None
            trajectory = TrajectoryFactory.create_from_xml_node(xml_node.find('trajectory'))
            if lanelet_assignment is True:
                shape_lanelet_assignment = cls.find_obstacle_shape_lanelets(initial_state, trajectory.state_list,
                                                                            lanelet_network, obstacle_id, shape)
                center_lanelet_assignment = cls.find_obstacle_center_lanelets(initial_state, trajectory.state_list,
                                                                              lanelet_network)
            else:
                shape_lanelet_assignment = None
                center_lanelet_assignment = None
            prediction = TrajectoryPrediction(trajectory, shape, center_lanelet_assignment, shape_lanelet_assignment)
        elif xml_node.find('occupancySet') is not None:
            prediction = SetBasedPredictionFactory.create_from_xml_node(xml_node.find('occupancySet'))
        else:
            prediction = None
        return DynamicObstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type,
                               obstacle_shape=shape, initial_state=initial_state, prediction=prediction,
                               initial_center_lanelet_ids=initial_center_lanelet_ids,
                               initial_shape_lanelet_ids=initial_shape_lanelet_ids,
                               initial_signal_state=initial_signal_state,
                               signal_series=signal_series)


class TrajectoryFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) \
            -> Trajectory:
        state_list = list()
        for state_node in xml_node.findall('state'):
            state_list.append(StateFactory.create_from_xml_node(state_node))
        if isinstance(state_list[0].time_step, Interval):
            t0 = min(state_list[0].time_step)
        else:
            t0 = state_list[0].time_step
        return Trajectory(t0, state_list)


class SetBasedPredictionFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> SetBasedPrediction:
        occupancies = list()
        for occupancy in xml_node.findall('occupancy'):
            occupancies.append(OccupancyFactory.create_from_xml_node(occupancy))
        if isinstance(occupancies[0].time_step, Interval):
            t0 = min(occupancies[0].time_step)
        else:
            t0 = occupancies[0].time_step
        return SetBasedPrediction(t0, occupancies)


class OccupancyFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Occupancy:
        shape = ShapeFactory.create_from_xml_node(xml_node.find('shape'))
        time = read_time(xml_node.find('time'))
        return Occupancy(time, shape)


class ShapeFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Shape:
        shape_list = list()
        for c in list(xml_node):
            shape_list.append(cls._read_single_shape(c))
        shape = cls._create_shape_group_if_needed(shape_list)
        return shape

    @classmethod
    def _read_single_shape(cls, xml_node: ElementTree.Element) \
            -> Shape:
        tag_string = xml_node.tag
        if tag_string == 'rectangle':
            return RectangleFactory.create_from_xml_node(xml_node)
        elif tag_string == 'circle':
            return CircleFactory.create_from_xml_node(xml_node)
        elif tag_string == 'polygon':
            return PolygonFactory.create_from_xml_node(xml_node)

    @classmethod
    def _create_shape_group_if_needed(cls, shape_list: List[Shape]) -> Shape:
        if len(shape_list) > 1:
            sg = ShapeGroup(shape_list)
            return sg
        else:
            return shape_list[0]


class RectangleFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Rectangle:
        length = float(xml_node.find('length').text)
        width = float(xml_node.find('width').text)
        if xml_node.find('orientation') is not None:
            orientation = float(xml_node.find('orientation').text)
        else:
            orientation = 0.0
        if xml_node.find('center') is not None:
            center = PointFactory.create_from_xml_node(
                xml_node.find('center'))
        else:
            center = np.array([0.0, 0.0])
        return Rectangle(length, width, center, orientation)


class CircleFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Circle:
        radius = float(xml_node.find('radius').text)
        if xml_node.find('center') is not None:
            center = PointFactory.create_from_xml_node(
                xml_node.find('center'))
        else:
            center = np.array([0.0, 0.0])
        return Circle(radius, center)


class PolygonFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Polygon:
        vertices = PointListFactory.create_from_xml_node(xml_node)
        return Polygon(vertices)


class PlanningProblemSetFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork) \
            -> PlanningProblemSet:
        planning_problem_set = PlanningProblemSet()
        for p in xml_node.findall('planningProblem'):
            planning_problem_set.add_planning_problem(
                PlanningProblemFactory.create_from_xml_node(p, lanelet_network))
        return planning_problem_set


class PlanningProblemFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork) \
            -> PlanningProblem:
        planning_problem_id = int(xml_node.get('id'))
        initial_state = cls._add_initial_state(xml_node)
        goal_region = GoalRegionFactory.create_from_xml_node(xml_node, lanelet_network)
        return PlanningProblem(planning_problem_id, initial_state, goal_region)

    @classmethod
    def _add_initial_state(cls, xml_node: ElementTree.Element) \
            -> State:
        initial_state = StateFactory.create_from_xml_node(xml_node.find('initialState'))
        return initial_state


class GoalRegionFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: LaneletNetwork)\
            -> GoalRegion:
        state_list = list()
        lanelets_of_goal_position = defaultdict(list)
        for idx, goal_state_node in enumerate(xml_node.findall('goalState')):
            state_list.append(StateFactory.create_from_xml_node(goal_state_node, lanelet_network))
            if goal_state_node.find('position') is not None\
                    and goal_state_node.find('position').find('lanelet') is not None:
                for l in goal_state_node.find('position').findall('lanelet'):
                    lanelets_of_goal_position[idx].append(int(l.get('ref')))
        if not lanelets_of_goal_position:
            lanelets_of_goal_position = None
        return GoalRegion(state_list, lanelets_of_goal_position)


class StateFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element, lanelet_network: Union[LaneletNetwork, None] = None)\
            -> State:
        state_args = dict()
        if xml_node.find('position') is not None:
            position = cls._read_position(xml_node.find('position'), lanelet_network)
            state_args['position'] = position
        if xml_node.find('time') is not None:
            state_args['time_step'] = read_time(xml_node.find('time'))
        if xml_node.find('orientation') is not None:
            orientation = cls._read_orientation(xml_node.find('orientation'))
            state_args['orientation'] = orientation
        if xml_node.find('velocity') is not None:
            speed = read_value_exact_or_interval(xml_node.find('velocity'))
            state_args['velocity'] = speed
        if xml_node.find('acceleration') is not None:
            acceleration = read_value_exact_or_interval(xml_node.find('acceleration'))
            state_args['acceleration'] = acceleration
        if xml_node.find('yawRate') is not None:
            yaw_rate = read_value_exact_or_interval(xml_node.find('yawRate'))
            state_args['yaw_rate'] = yaw_rate
        if xml_node.find('slipAngle') is not None:
            slip_angle = read_value_exact_or_interval(xml_node.find('slipAngle'))
            state_args['slip_angle'] = slip_angle
        if xml_node.find('steeringAngle') is not None:
            slip_angle = read_value_exact_or_interval(xml_node.find('steeringAngle'))
            state_args['steering_angle'] = slip_angle
        if xml_node.find('hitchAngle') is not None:
            slip_angle = read_value_exact_or_interval(xml_node.find('hitchAngle'))
            state_args['hitch_angle'] = slip_angle
        return State(**state_args)

    @classmethod
    def _read_position(cls, xml_node: ElementTree.Element,
                       lanelet_network: Union[LaneletNetwork, None] = None) \
            -> Union[np.ndarray, Shape]:
        if xml_node.find('point') is not None:
            position = PointFactory.create_from_xml_node(xml_node.find('point'))
        elif (xml_node.find('rectangle') is not None
              or xml_node.find('circle') is not None
              or xml_node.find('polygon') is not None):
            position = ShapeFactory.create_from_xml_node(xml_node)
        elif lanelet_network is not None and xml_node.find('lanelet') is not None:
            position_list = list()
            for l in xml_node.findall('lanelet'):
                lanelet = lanelet_network.find_lanelet_by_id(int(l.get('ref')))
                polygon = lanelet.convert_to_polygon()
                position_list.append(polygon)
            position = ShapeGroup(position_list)
        else:
            raise Exception()
        return position

    @classmethod
    def _read_orientation(cls, xml_node: ElementTree.Element) -> Union[float, AngleInterval]:
        if xml_node.find('exact') is not None:
            value = float(xml_node.find('exact').text)
        elif xml_node.find('intervalStart') is not None \
                and xml_node.find('intervalEnd') is not None:
            value = AngleInterval(
                float(xml_node.find('intervalStart').text),
                float(xml_node.find('intervalEnd').text))
        else:
            raise Exception()
        return value


class SignalStateFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> Union[SignalState, None]:
        if xml_node is None:
            return None
        state_args = dict()
        if xml_node.find('time') is not None:
            state_args['time_step'] = read_time(xml_node.find('time'))
        if xml_node.find('horn') is not None:
            horn = cls._read_boolean(xml_node.find('horn'))
            state_args['horn'] = horn
        if xml_node.find('indicatorLeft') is not None:
            indicatorLeft = cls._read_boolean(xml_node.find('indicatorLeft'))
            state_args['indicator_left'] = indicatorLeft
        if xml_node.find('indicatorRight') is not None:
            indicatorRight = cls._read_boolean(xml_node.find('indicatorRight'))
            state_args['indicator_right'] = indicatorRight
        if xml_node.find('brakingLights') is not None:
            brakingLights = cls._read_boolean(xml_node.find('brakingLights'))
            state_args['braking_lights'] = brakingLights
        if xml_node.find('hazardWarningLights') is not None:
            hazardWarningLights = cls._read_boolean(xml_node.find('hazardWarningLights'))
            state_args['hazard_warning_lights'] = hazardWarningLights
        if xml_node.find('flashingBlueLights') is not None:
            flashingBlueLights = cls._read_boolean(xml_node.find('flashingBlueLights'))
            state_args['flashing_blue_lights'] = flashingBlueLights
        return SignalState(**state_args)

    @classmethod
    def _read_boolean(cls, xml_node: ElementTree.Element) -> bool:
        if xml_node.text == "true":
            return True
        elif xml_node.text == "false":
            return False
        else:
            raise ValueError()


class SignalSeriesFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> List[SignalState]:
        signal_state_list = []
        if xml_node is None:
            return signal_state_list
        for signal_state_node in xml_node.findall('signalState'):
            signal_state_list.append(SignalStateFactory.create_from_xml_node(signal_state_node))

        return signal_state_list


class PointListFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> np.ndarray:
        point_list = []
        for point_node in xml_node.findall("point"):
            point_list.append(
                PointFactory.create_from_xml_node(point_node))
        return np.array(point_list)


class PointFactory:
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> np.ndarray:
        x = float(xml_node.find('x').text)
        y = float(xml_node.find('y').text)
        if xml_node.find('z') is None:
            return np.array([x, y])
        else:
            z = float(xml_node.find('z').text)
            return np.array([x, y, z])


class EnvironmentObstacleFactory(ObstacleFactory):
    """ Class to create a list of objects of type EnvironmentObstacle from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> EnvironmentObstacle:
        obstacle_type = EnvironmentObstacleFactory.read_type(xml_node)
        obstacle_id = EnvironmentObstacleFactory.read_id(xml_node)
        shape = EnvironmentObstacleFactory.read_shape(xml_node.find('shape'))

        return EnvironmentObstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type, obstacle_shape=shape)


class PhantomObstacleFactory(ObstacleFactory):
    """ Class to create a list of objects of class PhantomObstacle from an XML element."""
    @classmethod
    def create_from_xml_node(cls, xml_node: ElementTree.Element) -> PhantomObstacle:
        obstacle_id = PhantomObstacleFactory.read_id(xml_node)
        if xml_node.find('occupancySet') is not None:
            prediction = SetBasedPredictionFactory.create_from_xml_node(xml_node.find('occupancySet'))
        else:
            prediction = None
        return PhantomObstacle(obstacle_id=obstacle_id, prediction=prediction)
