import abc
from typing import List, Tuple

import zlib
import numpy as np

from .Cascade import ParetoCascade

class Base_Vector(object):
    __vector: list      # list with arbitrary internal form representing the change between two solution parameter sets
    __length: np.double # normalised distance between the two points represented by the vector

    def __init__(self, solution_vector: list, normalised_length: np.double) -> None:
        self.__vector = solution_vector
        self.__length = normalised_length

    @property
    def vector(self) -> list:
        return self.__vector

    @property
    def length(self) -> np.double:
        return self.__length

class Base_Space(abc.ABC):
    _rng: np.random.Generator

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def rng_gauss(self, Ndim: int) -> Tuple[np.double, np.double]: # N-dimensional Gaussian distribution, i.e., Normal(mu=0,sigma=1)
        norm = 0
        while norm < 1E-8:
            gauss = np.random.normal(0, 1, Ndim)
            norm  = np.linalg.norm(gauss)

        return gauss, norm

    @abc.abstractmethod
    def random_coordinate(self) -> list:
        return None

    @abc.abstractmethod
    def random_nearby_coordinate(self, origin: list, unit_sigma: np.double, direction: Base_Vector = None) -> list:
        return None

    @abc.abstractmethod
    def delta(self, from_solution_coordinate: list, to_solution_coordinate: list) -> Base_Vector:
        return None

    @abc.abstractmethod
    def coordinate_to_json_string(self, solution_coordinate: list) -> str:
        pass

    @abc.abstractmethod
    def coordinate_from_json_string(self, json_string: str) -> list:
        pass

class Base_Solution(object):
    _space: Base_Space    # the space this coordinate sits within

    __coordinate: list    # list with arbitrary internal form representing the parameter set for this current solution
    __cost: np.ndarray    # array of one or more associated costs
    __checksum: np.uint32 # checksum of the JSON string representation

    def __init__(self, solution_space: Base_Space, solution_coordinate: list = None) -> None:
        self._space = solution_space

        if solution_coordinate is None:
            self.__coordinate = solution_space.random_coordinate()
        else:
            self.__coordinate = solution_coordinate

        self.__cost = None
        self.__checksum = None

    @property
    def coordinate(self) -> list:
        return self.__coordinate

    @property
    def cost(self) -> np.ndarray:
        return self.__cost

    @cost.setter
    def cost(self, solution_cost: np.ndarray) -> None:
        self.__cost = solution_cost

    @property
    def jstr(self) -> str:
        return self._space.coordinate_to_json_string(self.__coordinate)

    @property
    def checksum(self) -> np.uint32:
        if self.__checksum is None:
            self.__checksum = zlib.adler32(bytearray(self.jstr, 'utf-8'))
        return self.__checksum

class Base_Problem(abc.ABC):
    __space: Base_Space
    __evaluations: int

    def __init__(self, solution_space: Base_Space) -> None:
        self.__space = solution_space
        self.__evaluations = 0

    @property
    def space(self) -> Base_Space:
        return self.__space

    @property
    def evaluations(self) -> int:
        return self.__evaluations

    @abc.abstractmethod
    def _evaluate(self, X: Base_Solution) -> None:
        pass

    def evaluate(self, X: Base_Solution) -> None:
        self.__evaluations = self.__evaluations + 1
        self._evaluate(X)

class Base_Optimiser(abc.ABC):
    __problem: Base_Problem
    __space: Base_Space
    __it: int
    __count: np.uint32  # how many records so far
    __index: np.ndarray # cross-reference checksum with index and ranking
    __costs: np.ndarray # costs, cross-ref and ranking
    __history: list     # complete history of solutions
    __cascade: ParetoCascade
    __noisy: bool
    __scout_count: int
    __scout_list: List[Tuple['Base_Scout',int]]

    def __init__(self, the_problem: Base_Problem, scout_list: List[Tuple['Base_Scout',int]]) -> None:
        self.__problem = the_problem
        self.__space = the_problem.space
        self.__it = 0
        self.__count = 0
        self.__index = None
        self.__costs = None
        self.__history = []
        self.__cascade = None
        self.__noisy = False
        self.__scout_count = 0
        self.__scout_list = scout_list

        for s in scout_list:
            scout, Nscout = s
            scout.optimiser = self
            scout_count = scout.scouts
            self.__scout_count = self.__scout_count + scout_count * Nscout

    @property
    def problem(self) -> Base_Problem:
        return self.__problem

    @property
    def space(self) -> Base_Space:
        return self.__space

    @property
    def iteration(self) -> int:
        return self.__it

    @property
    def cascade(self) -> ParetoCascade:
        return self.__cascade

    @property
    def noisy(self) -> None:
        return self.__noisy

    @noisy.setter
    def noisy(self, print_info: bool) -> None:
        self.__noisy = print_info

    @property
    def scout_count(self) -> int:
        return self.__scout_count

    @property
    def scout_list(self) -> List[Tuple['Base_Scout',int]]:
        return self.__scout_list

    @abc.abstractmethod
    def _iterate(self, sigma: np.double) -> None:
        pass

    def iterate(self, sigma: np.double) -> None:
        self.__it = self.__it + 1
        self._iterate(sigma)

    def print(self, print_full_solution: bool) -> None:
        for i in range(0, self.__count):
            if self.__index[i,2] == 0:
                if print_full_solution:
                    S = self.__history[self.__index[i,0]]
                    print('{index} -> {cost} @ {soln}'.format(index=self.__index[i,:], cost=self.__costs[i,:], soln=S.jstr))
                else:
                    print('           {cost}'.format(cost=self.__costs[i,:]))

    def _record(self, solution: Base_Solution) -> np.uint32:
        rank: np.uint32 = 0xFFFFFFFF

        cost = solution.cost
        csum = solution.checksum

        if self.__count == 0 and cost.shape[0] > 1:
            rng = self.__space.rng
            C = ParetoCascade(rng)
            for c in range(0, 10):
                C = ParetoCascade(rng, C)
            self.__cascade = C

        if self.__count == 0:
            self.__costs = np.empty((500,cost.shape[0]), dtype=np.double)
            self.__index = np.empty((500,3), dtype=np.uint32)
        elif self.__count == self.__costs.shape[0]:
            new_size = self.__costs.shape[0] + 500
            self.__costs = np.resize(self.__costs, (new_size,cost.shape[0]))
            self.__index = np.resize(self.__index, (new_size,3))

        self.__costs[self.__count,:] = cost
        self.__index[self.__count,:] = [self.__count, csum, rank]

        self.__history.append(solution)
        self.__count = self.__count + 1

        if self.__cascade is not None:
            rank = self.__cascade.add(solution)
            self.__index[self.__count-1,2] = rank
        else:
            # This method of ranking only really works for non-multiobjective problems
            self.__index[0:self.__count,2] = rank
            for col in range(0, self.__costs.shape[1]):
                rank = 0
                last = None
                ivec = np.argsort(self.__costs[0:self.__count,col])
                for iv in ivec:
                    if last is not None:
                        if self.__costs[iv,col] > last:
                            rank = rank + 1
                    if self.__index[iv,2] > rank:
                        self.__index[iv,2] = rank
                    last = self.__costs[iv,col]
            rank = self.__index[self.__count-1,2]

        return rank

    def trim_history(self, new_count: np.uint32) -> None:
        if self.__count <= new_count:
            return
        ivec = np.argsort(self.__index[0:self.__count,2])
        self.__index = self.__index[ivec,:]
        self.__costs = self.__costs[ivec,:]
        self.__count = new_count
        hist = []
        for i in range(0, self.__count):
            hist.append(self.__history[self.__index[i,0]])
            self.__index[i,0] = i
        self.__history = hist

    def select_solution(self, max_rank: np.uint32) -> Base_Solution:
        if self.__count == 0:
            return None
        if self.__cascade is not None:
            return self.__cascade.select_solution()

        ivectuple = np.where(self.__index[0:self.__count,2] <= max_rank)
        indices = ivectuple[0]
        if indices.size == 0:
            return None
        selection_index = indices[self.__space.rng.choice(indices.shape[0])]
        return self.__history[self.__index[selection_index,0]]

    def lookup(self, solution: Base_Solution) -> Base_Solution:
        if self.__count == 0:
            return None

        ivectuple = np.where(self.__index[0:self.__count,1] == solution.checksum)
        if ivectuple[0].size == 0:
            return None

        match = None

        jstr = solution.jstr
        for iv in ivectuple[0]:
            S = self.__history[self.__index[iv,0]]
            if S.jstr == jstr:
                match = S
                break

        return match

    def evaluate_and_record(self, new_solution: Base_Solution) -> np.uint32:
        self.__problem.evaluate(new_solution)
        return self._record(new_solution)
