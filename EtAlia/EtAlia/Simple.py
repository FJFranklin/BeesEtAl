import abc
from typing import Tuple

import json
import numpy as np

from .Base import Base_Vector, Base_Space, Base_Solution, Base_Problem, Base_Optimiser
from .Scout import Base_Scout, FrontierScout

class SimpleTestFunction(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def extents(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def within_constraints(self, x: np.ndarray) -> bool:
        return True

class SimpleSpace(Base_Space):
    __bbox: np.ndarray
    __range: np.ndarray
    __Ndim: int
    __granularity: int

    def __init__(self, bounding_box: list) -> None:
        Base_Space.__init__(self)
        self.__bbox = np.array(bounding_box)
        assert (self.__bbox.ndim == 2) and (self.__bbox.shape[0] == 2), "SimpleSpace: bounding_box should be a two-row matrix of minima and maxima"
        self.__Ndim = self.__bbox.shape[1]
        self.__range = self.__bbox[1,:] - self.__bbox[0,:]
        self.__granularity = None

    @property
    def granularity(self) -> None:
        return self.__granularity

    @granularity.setter
    def granularity(self, number_of_decimals: int) -> None:
        self.__granularity = number_of_decimals

    @granularity.deleter
    def granularity(self) -> None:
        self.__granularity = None

    def random_coordinate(self) -> list:
        new_coordinate = self._rng.uniform(self.__bbox[0,:], self.__bbox[1,:], self.__Ndim)
        if self.__granularity is not None:
            new_coordinate = 0 + np.around(new_coordinate, self.__granularity) # add zero to prevent numpy retaining -ve sign in -0.0
        return [new_coordinate]

    def random_nearby_coordinate(self, origin: list, unit_sigma: np.double) -> list:
        origin_coord = origin[0]
        new_coordinate = None
        for n in range(0, 100):
            gauss, norm = self.rng_gauss(self.__Ndim)
            new_coordinate = origin_coord + self.__range * unit_sigma * gauss
            if self.__granularity is not None:
                new_coordinate = 0 + np.around(new_coordinate, self.__granularity) # add zero to prevent numpy retaining -ve sign in -0.0
            if (new_coordinate < self.__bbox[0,:]).any():
                continue
            if (new_coordinate > self.__bbox[1,:]).any():
                continue
            break
        if new_coordinate is not None:
            return [new_coordinate]
        return None

    def delta(self, from_solution_coordinate: list, to_solution_coordinate: list) -> Base_Vector:
        v = to_solution_coordinate[0] - from_solution_coordinate[0]
        l = np.linalg.norm(v)
        return Base_Vector(v, l)

    def coordinate_to_json_string(self, solution_coordinate: list) -> str:
        return json.dumps(solution_coordinate[0].tolist())

    def coordinate_from_json_string(self, json_string: str) -> list:
        return [np.array(json.loads(json_string))]

class SimpleProblem(Base_Problem):
    __function: SimpleTestFunction

    def __init__(self, space: SimpleSpace, test_function: SimpleTestFunction) -> None:
        Base_Problem.__init__(self, space)
        self.__function = test_function

    def evaluate(self, X: Base_Solution) -> None:
        X.cost = self.__function.evaluate(X.coordinate[0])

class SimpleOptimiser(Base_Optimiser):
    __sigma: np.double

    def __init__(self, the_problem: SimpleProblem) -> None:
        Base_Optimiser.__init__(self, the_problem)
        self.__sigma = 0.5

    def _iterate(self, noisy: bool = False) -> None:
        it = self.iteration
        problem = self.problem
        space = problem.space

        if noisy:
            print("Iteration {i}: ".format(i=it))

        if it == 1:
            B = Base_Scout(self)
            for s in range(0, 10):
                B.scout()
        else:
            F = FrontierScout(self, self.__sigma)
            for s in range(0, 10):
                F.scout()
            self.__sigma = self.__sigma * 0.99

        if noisy:
            if self.cascade is not None:
                self.cascade.rank_print()
            else:
                self.print(False)

    def pareto_solutions(self) -> list:
        if self.cascade is None:
            return None
        sols = []
        cost = []
        for sol in self.cascade.sols:
            sols.append(sol.coordinate[0])
            cost.append(sol.cost)
        return np.asarray(sols), np.asarray(cost)