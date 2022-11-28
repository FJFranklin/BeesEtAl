import abc
from typing import Tuple

import json
import numpy as np

from .Base import Base_Vector, Base_Space, Base_Solution, Base_Problem, Base_Optimiser

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
        self.__sigma = 0.1

    def _iterate(self, noisy: bool = False) -> None:
        it = self.iteration
        problem = self.problem
        space = problem.space

        if noisy:
            print("Iteration {i}: ".format(i=it))

        if it == 1:
            for scout in range(0, 10):
                self._scout_evaluate_record()
        else:
            for scout in range(0, 10):
                S = self.select_solution(it)
                if S is not None:
                    S = Base_Solution(space, space.random_nearby_coordinate(S.coordinate, self.__sigma))
                    if S is not None:
                        if self._lookup(S) is not None:
                            S = None
                if S is not None:
                    problem.evaluate(S)
                    self._record(S)
                else:
                    self._scout_evaluate_record()
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

class YueQuLiang(SimpleTestFunction):
    '''
    A Multiobjective Particle Swarm Optimizer Using Ring Topology for Solving Multimodal Multiobjective Problems
    Caitong Yue; Boyang Qu; Jing Liang
    DOI: 10.1109/TEVC.2017.2754271

    1: MMF1
    '''
    __no: int   # 1-

    def __init__(self, test_fn_number: int) -> None:
        SimpleTestFunction.__init__(self)
        self.__no = test_fn_number
        assert (0 < test_fn_number) and (test_fn_number <= 1), "Yue, Qu & Liang test function number is 1-"

    def extents(self) -> np.ndarray:
        if self.__no == 1:
            ext = np.asarray([[1,-1], [3,1]])
        return ext

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.__no == 1:
            f1 = x[0]
            f2 = 1.0 - np.sqrt(x[0]) + 2.0 * (x[1] - np.sin(6 * np.pi * x[0] + np.pi))**2
            cost = np.asarray([f1,f2])
        return cost

#    def within_constraints(self, x: np.ndarray) -> bool:
#        return True

class Viennet(SimpleTestFunction):
    """
    Viennet multi-objective optimisation test function
    """

    def __init__(self) -> None:
        SimpleTestFunction.__init__(self)

    def extents(self) -> np.ndarray:
        return np.asarray([-3 * np.ones(2), 3 * np.ones(2)])

    def evaluate(self, X):
        x = X[0]
        y = X[1]

        r2 = x**2 + y**2
        f1 = 0.5 * r2 + np.sin(r2)
        f2 = (3 * x - 2 * y + 4)**2 / 8 + (x - y + 1)**2 / 27 + 15
        f3 = 1 / (r2 + 1) - 1.1 * np.exp(-r2)

        return np.asarray([f1,f2,f3])

class Gholami(SimpleTestFunction):
    '''
    Gholami & Mohammadi, A Novel Combination of Bees and Firefly Algorithm to Optimize Continuous Problems

    Türker Tuncer, LDW-SCSA: Logistic Dynamic Weight based Sine Cosine Search Algorithm for Numerical Functions Optimization 
    https://arxiv.org/ftp/arxiv/papers/1809/1809.03055.pdf

    Hartmut Pohlheim, Examples of Objective Functions
    http://www.geatbx.com/download/GEATbx_ObjFunExpl_v38.pdf

    Wikipedia, Test functions for optimization
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    F1:  De Jong / Sphere (ND) cost function; optimum @ (0,...
    F2:  Schwefel 2.22 (ND) cost function; optimum @ (0,...
    F3:  Schwefel 1.2 - Rotated hyper-ellipsoid (ND) cost function; optimum @ (0,...
    F4:  Schwefel 2.21 (ND) cost function; optimum @ (0,...
    F5:  Rosenbrock (ND) cost function; optimum @ (0,...
    F6:  Step (ND) cost function; optimum @ (-0.5,...
    F7:  Noise (ND) cost function; optimum @ (0,...
    F8:  Schwefel (ND) cost function
    F9:  Rastrigin (ND) cost function; optimum @ (0,...
    F10: Ackley (ND) cost function; optimum @ (0,...
    F11: Griewangk (ND) cost function; optimum @ (0,...
    F12: Generalised Penalised 1 (ND) cost function; optimum @ (0,...
    '''
    __Ndim: int # number of dimensions
    __no: int   # 1-12

    def __init__(self, Ndim: int, test_fn_number: int) -> None:
        SimpleTestFunction.__init__(self)
        self.__Ndim = Ndim
        self.__no = test_fn_number
        assert Ndim > 0, "Number of dimensions is a non-zero positive integer"
        assert (0 < test_fn_number) and (test_fn_number <= 12), "Gholami test function number is 1-12"

    def extents(self) -> np.ndarray:
        if self.__no == 1:
            emin, emax = -5.12, 5.12
        elif self.__no == 2:
            emin, emax = -10, 10
        elif self.__no == 3:
            emin, emax = -65.536, 65.536
        elif self.__no == 4:
            emin, emax = -100, 100
        elif self.__no == 5:
            emin, emax = -2.048, 2.048
        elif self.__no == 6:
            emin, emax = -100, 100
        elif self.__no == 7:
            emin, emax = -1.28, 1.28
        elif self.__no == 8:
            emin, emax = -500, 500
        elif self.__no == 9:
            emin, emax = -5.12, 5.12
        elif self.__no == 10:
            emin, emax = -32.768, 32.768
        elif self.__no == 11:
            emin, emax = -600, 600
        elif self.__no == 12:
            emin, emax = -50, 50

        return np.asarray([emin * np.ones(self.__Ndim), emax * np.ones(self.__Ndim)])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.__no == 1:
            cost = sum(np.power(x, 2))
        elif self.__no == 2:
            cost = sum(np.abs(x)) + np.prod(np.abs(x))
        elif self.__no == 3:
            cost = 0
            for i in range(0, len(x)):
                cost = cost + (sum(x[0:(i+1)]))**2
        elif self.__no == 4:
            cost = max(np.abs(x))
        elif self.__no == 5:
            cost = sum(100 * np.power(x[1:] - np.power(x[0:-1], 2), 2) + np.power(1 - x[0:-1], 2))
        elif self.__no == 6:
            cost = sum(np.floor(np.power(x + 0.5, 2)))
        elif self.__no == 7:
            cost = sum(np.power(x, 4) * np.asarray(range(1, 1 + len(x)))) + np.random.rand(1)
        elif self.__no == 8:
            cost = -sum(x * np.sin(np.sqrt(abs(x))))
        elif self.__no == 9:
            cost = sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x) + 10)
        elif self.__no == 10:
            rms_x = np.sqrt(x.dot(x) / len(x))
            cost = np.exp(1) + 20 * (1 - np.exp(-rms_x / 5)) - np.exp(sum(np.cos(2 * np.pi * x)) / len(x))
        elif self.__no == 11:
            cost = sum(np.power(x, 2)) / 4000 - np.prod(np.cos(np.power(x, 2) / np.power(range(1, 1+len(x)), 0.5))) + 1
        elif self.__no == 12:
            a = 10
            k = 100
            m = 4
            c = 0
            for i in range(0, len(x)):
                xi = x[i]
                if xi > a:
                    c = c + k * (xi - a)**m
                elif xi < -a:
                    c = c + k * (-xi - a)**m

            y = 1 + (x + 1) / 4
            cost = sum(np.power(y[0:(len(x)-1)] - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * y[1:len(x)]), 2)))
            cost = (self.cost + 10 * np.sin(np.pi * y[0]) + (y[len(x)-1] - 1)**2) * np.pi / len(x) + c

        return np.asarray([cost])
