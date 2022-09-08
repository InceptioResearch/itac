__author__ = "Sebastian Maierhofer"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["CAR@TUM"]
__version__ = "2022.1"
__maintainer__ = "Sebastian Maierhofer"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Release"

from typing import List, Set, Dict

from commonroad.common.validity import *


class IntersectionIncomingElement:
    """
    This class represents an incoming element of an intersection.
    An incoming can consist of several adjacent lanelets which lead to an intersection, right, straight,
    and left successor lanelets, and a reference to the incoming located on the left side.
    The right/straight/left successors are used to infer for which lanelets the traffic light
    is valid for and to facilitate the calculation of priorities at intersections.
    The left incoming is used to infer the right-before-left-rule.
    """

    def __init__(self, incoming_id: int, incoming_lanelets: Set[int] = None, successors_right: Set[int] = None,
                 successors_straight: Set[int] = None, successors_left: Set[int] = None, left_of: int = None):
        """
        :param incoming_id: incoming element ID
        :param incoming_lanelets: set of IDs of incoming lanelets
        :param successors_right: set of IDs of incoming lanelets which turn right
        :param successors_straight: set of IDs of incoming lanelets which go straight
        :param successors_left: set of IDs of incoming lanelets which turn left
        :param left_of: incoming element ID of incoming element located left of this incoming element
        """
        self._incoming_id = None
        self._incoming_lanelets = None
        self.incoming_id = incoming_id
        self.incoming_lanelets = incoming_lanelets
        self.successors_right = successors_right
        self.successors_straight = successors_straight
        self.successors_left = successors_left
        self._left_of = left_of

    def __eq__(self, other):
        if not isinstance(other, IntersectionIncomingElement):
            warnings.warn(f"Inequality between IntersectionIncomingElement "
                          f"{repr(self)} and different type {type(other)}")
            return False

        if self._incoming_id == other.incoming_id and self._incoming_lanelets == other.incoming_lanelets \
                and self._successors_right == other.successors_right \
                and self._successors_straight == other.successors_straight \
                and self._successors_left == other.successors_left and self._left_of == other.left_of:
            return True

        warnings.warn(f"Inequality of IntersectionIncomingElement {repr(self)} and the other one {repr(other)}")
        return False

    def __hash__(self):
        return hash((self._incoming_id, frozenset(self._incoming_lanelets), frozenset(self._successors_right),
                     frozenset(self._successors_straight), frozenset(self._successors_left), self._left_of))

    def __str__(self):
        return f"IntersectionIncomingElement with id {self._incoming_id} represents the incoming " \
               f"lanelets {self._incoming_lanelets} and has right successors {self._successors_right}, " \
               f"straight successors {self._successors_straight}, and left successors {self._successors_left}"

    def __repr__(self):
        return f"IntersectionIncomingElement(incoming_id={self._incoming_id}, " \
               f"incoming_lanelets={self._incoming_lanelets}, successors_right={self._successors_right}, " \
               f"successors_straight={self._successors_straight}, successors_left={self._successors_left}, " \
               f"left_of={self._left_of})"

    @property
    def incoming_id(self) -> int:
        """
        :returns ID of incoming
        """
        return self._incoming_id

    @incoming_id.setter
    def incoming_id(self, i_id: int):
        """
        :param i_id ID of incoming
        """
        if self._incoming_id is None:
            assert is_natural_number(i_id), '<IntersectionIncomingElement/incoming_id>: Provided incoming_id is not ' \
                                            'valid! id={}'.format(i_id)
            self._incoming_id = i_id
        else:
            warnings.warn('<IntersectionIncomingElement/incoming_id>: incoming_id of incoming is immutable')

    @property
    def incoming_lanelets(self) -> Set[int]:
        """
        :returns set of IDs of incoming lanelets
        """
        return self._incoming_lanelets

    @incoming_lanelets.setter
    def incoming_lanelets(self, incoming_lanelets: Set[int]):
        """
        :param incoming_lanelets: set of IDs of incoming lanelets
        """
        if self._incoming_lanelets is None:
            assert len(incoming_lanelets) > 0, '<IntersectionIncomingElement/incoming_id>: Incoming ' \
                                               'must consist of at least one lanelet '
            self._incoming_lanelets = incoming_lanelets
        else:
            warnings.warn('<IntersectionIncomingElement/incoming_lanelets>: incoming_lanelets of '
                          'incoming are immutable')

    @property
    def successors_right(self) -> Set[int]:
        """
        :returns set of IDs of incoming lanelets which turn right
        """
        return self._successors_right

    @successors_right.setter
    def successors_right(self, successors_right: Set[int]):
        """
        :param successors_right: set of IDs of incoming lanelets which turn right
        """
        if successors_right is None:
            self._successors_right = set()
        else:
            self._successors_right = successors_right

    @property
    def successors_straight(self) -> Set[int]:
        """
        :returns set of IDs of incoming lanelets which go straight
        """
        return self._successors_straight

    @successors_straight.setter
    def successors_straight(self, successors_straight: Set[int]):
        """
        :param successors_straight: set of IDs of incoming lanelets which go straight
        """
        if successors_straight is None:
            self._successors_straight = set()
        else:
            self._successors_straight = successors_straight

    @property
    def successors_left(self) -> Set[int]:
        """
        :returns set of IDs of incoming lanelets which turn left
        """
        return self._successors_left

    @successors_left.setter
    def successors_left(self, successors_left: Set[int]):
        """
        :param successors_left: set of IDs of incoming lanelets which turn left
        """
        if successors_left is None:
            self._successors_left = set()
        else:
            self._successors_left = successors_left

    @property
    def left_of(self) -> int:
        """
        :returns incoming element ID of incoming element located left of this incoming element
        """
        return self._left_of

    @left_of.setter
    def left_of(self, left_of: int):
        """
        :param left_of: incoming element ID of incoming element located left of this incoming element
        """
        self._left_of = left_of


class Intersection:
    """
    This class represent an intersection.
    An intersection element is defined by at least one incoming and an optional crossing element.
    The crossing element models lanelets which cross other lanelets, e.g., these are usually lanelets of type crosswalk.
    """

    def __init__(self, intersection_id: int, incomings: List[IntersectionIncomingElement], crossings: Set[int] = None):
        """
        :param intersection_id: ID of intersection element
        :param incomings: set of incoming elements in intersection
        :param crossings: set of crossing elements in intersection
        """
        self._intersection_id = None
        self._incomings = None
        self._crossings = None

        self.intersection_id = intersection_id
        self.incomings = incomings
        self.crossings = crossings

    def __eq__(self, other):
        if not isinstance(other, Intersection):
            warnings.warn(f"Inequality between Intersection {repr(self)} and different type {type(other)}")
            return False

        list_elements_eq = True
        incomings = {incoming.incoming_id: incoming for incoming in self._incomings}
        incomings_other = {incoming.incoming_id: incoming for incoming in other.incomings}
        intersection_eq = len(incomings) == len(incomings_other)
        for k in incomings.keys():
            if k not in incomings_other:
                intersection_eq = False
                continue
            if incomings.get(k) != incomings_other.get(k):
                list_elements_eq = False

        if intersection_eq and self._intersection_id == other.intersection_id and self._crossings == other.crossings:
            return list_elements_eq

        warnings.warn(f"Inequality of Intersection {repr(self)} and the other one {repr(other)}")
        return False

    def __hash__(self):
        return hash((self._intersection_id, frozenset(self._incomings), frozenset(self._crossings)))

    def __str__(self):
        return f"Intersection with id {self._intersection_id} consisting of {len(self._incomings)} incoming elements " \
               f"and crossings {self._crossings}"

    def __repr__(self):
        return f"Intersection(intersection_id={self._intersection_id}, incomings={repr(self._incomings)}, " \
               f"crossings={self._crossings})"

    @property
    def intersection_id(self) -> int:
        """
        :returns ID of intersection element
        """
        return self._intersection_id

    @intersection_id.setter
    def intersection_id(self, i_id: int):
        """
        :param i_id ID of intersection element
        """
        if self._intersection_id is None:
            assert is_natural_number(i_id), '<Intersection/intersection_id>: Provided intersection_id is not ' \
                                            'valid! id={}'.format(i_id)
            self._intersection_id = i_id
        else:
            warnings.warn('<Intersection/intersection_id>: intersection_id of intersection is immutable')

    @property
    def incomings(self) -> List[IntersectionIncomingElement]:
        """
        :returns set of incoming elements in intersection
        """
        return self._incomings

    @incomings.setter
    def incomings(self, incomings: List[IntersectionIncomingElement]):
        """
        :param incomings: i_id ID of intersection element
        """
        if self._incomings is None:
            assert len(incomings) > 0, '<Intersection/incomings>: Intersection ' \
                                       'must consist of at least two incomings '
            self._incomings = incomings
        else:
            warnings.warn('<Intersection/incomings>: incomings of intersection are immutable')

    @property
    def crossings(self) -> Set[int]:
        """
        :returns set of crossing elements in intersection
        """
        return self._crossings

    @crossings.setter
    def crossings(self, crossings: Set[int]):
        """
        :param crossings: set of crossing elements in intersection
        """
        if self._crossings is None:
            if crossings is None:
                self._crossings = set()
            else:
                self._crossings = crossings
        else:
            warnings.warn('<Intersection/crossings>: crossings of intersection are immutable')

    @property
    def map_incoming_lanelets(self) -> Dict[int, IntersectionIncomingElement]:
        """
        Maps all incoming lanelet ids to IntersectionIncomingElement
        :returns dictionary mapping lanelet IDs to incomings
        """
        return {l_id: inc for inc in self.incomings for l_id in inc.incoming_lanelets}
