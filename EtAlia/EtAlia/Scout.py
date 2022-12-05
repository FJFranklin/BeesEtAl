from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull

from .Base import Base_Space, Base_Solution, Base_Optimiser

class Base_Scout(object):
    __opt: Base_Optimiser
    __space: Base_Space
    __uniqueness: int   # set to zero to disable testing for uniqueness; otherwise, number of attempts
    __sigma: np.double

    def __init__(self) -> None:
        self.__opt = None
        self.__space = None
        self.__uniqueness: int = 100
        self.__sigma = 1.0

    @property
    def scouts(self) -> int:
        return 1

    @property
    def optimiser(self) -> Base_Optimiser:
        return self.__opt

    @optimiser.setter
    def optimiser(self, opt: Base_Optimiser) -> None:
        self.__opt = opt
        self.__space = opt.space

    @property
    def space(self) -> Base_Space:
        return self.__space

    @property
    def uniqueness(self) -> int:
        return self.__uniqueness

    @uniqueness.setter
    def uniqueness(self, timeout: int) -> None:
        if timeout <= 0:
            self.__uniqueness = 0
        else:
            self.__uniqueness = timeout

    @property
    def sigma(self) -> np.double:
        return self.__sigma

    @sigma.setter
    def sigma(self, value: np.double) -> None:
        self.__sigma = value

    def __new_unique_scout(self, near: Base_Solution = None, sigma: np.double = 1) -> Base_Solution:
        count = 0
        while True:
            S = Base_Solution(self.__space)
            if self.__uniqueness == 0:
                break
            if self.__opt.lookup(S) is None:
                break
            count = count + 1
            if count == self.__uniqueness:
                S = None
                break
        return S

    def scout(self) -> Tuple[Base_Solution, np.uint32]:
        S = self.__new_unique_scout()
        if S is None:
            print(" - Base_Scout::scout] Timeout searching for scout location (bailing).")
        if S is None:
            return None, 0xFFFFFFFF

        rank = self.__opt.evaluate_and_record(S)
        if self.__opt.noisy and rank == 0:
            print("B", end="") # print("    Base_Scout: {cst}".format(cst=S.cost))
        return S, rank

class FrontierScout(Base_Scout):
    def __init__(self) -> None:
        Base_Scout.__init__(self)

    def __new_unique_scout(self, near: Base_Solution) -> Base_Solution:
        count = 0
        while True:
            S = Base_Solution(self.space, self.space.random_nearby_coordinate(near.coordinate, self.sigma))

            if self.uniqueness == 0:
                break
            if self.optimiser.lookup(S) is None:
                break
            count = count + 1
            if count == self.uniqueness:
                S = None
                break
        return S

    def scout(self) -> Tuple[Base_Solution, np.uint32]:
        S = None

        near = self.optimiser.select_solution(10)
        if near is None:
            print(" - FrontierScout::scout] Timeout getting nearby scout location.")
        else:
            S = self.__new_unique_scout(near)
            if S is None:
                print(" - FrontierScout::scout] Timeout searching for nearby scout location.")
        if S is None:
            return super().scout()

        rank = self.optimiser.evaluate_and_record(S)
        if self.optimiser.noisy and rank == 0:
            print("F", end="") # print("    FrontierScout: {cst}".format(cst=S.cost))
        return S, rank

class CascadeScout(FrontierScout):
    def __init__(self) -> None:
        Base_Scout.__init__(self)

    def __new_unique_scout(self) -> Base_Solution:
        cascade = self.optimiser.cascade
        if cascade is None:
            return None
        hull = cascade.hull
        if hull is None:
            return None

        origin_index = len(cascade.pts) - 1
        count = 0
        while True:
            if count == self.uniqueness:
                S = None
                break
            count = count + 1

            simplex = hull.simplices[self.space.rng.choice(len(hull.simplices))]
            sp1 = self.space.rng.choice(len(simplex))
            sp2 = sp1 + 1
            if sp2 == len(simplex):
                sp2 = 0
            iv1 = simplex[sp1]
            if iv1 == origin_index:
                continue
            iv2 = simplex[sp2]
            if iv2 == origin_index:
                continue
            
            S1 = cascade.sols[iv1]
            S2 = cascade.sols[iv2]

            delta = self.space.delta(S1.coordinate, S2.coordinate)
            S = Base_Solution(self.space, self.space.random_nearby_coordinate(S1.coordinate, self.sigma, delta))

            if self.uniqueness == 0:
                break
            if self.optimiser.lookup(S) is None:
                break
        return S

    def scout(self) -> Tuple[Base_Solution, np.uint32]:
        S = self.__new_unique_scout()
        if S is None:
            return super().scout()

        rank = self.optimiser.evaluate_and_record(S)
        if self.optimiser.noisy and rank == 0:
            print("C", end="") # print("    CascadeScout: {cst}".format(cst=S.cost))
        return S, rank

class BA_Patch(Base_Scout):
    __Nbee: int
    __level: int
    __l_max: int
    __scale: np.double
    __best: Base_Solution

    def __init__(self, number_of_bees: int, level_max: int) -> None:
        Base_Scout.__init__(self)
        self.__Nbee = number_of_bees
        self.__level = -1
        self.__l_max = level_max
        self.__scale = 1
        self.__best = None
        assert number_of_bees > 0, "Number of bees in patch must be a positive integer"
        assert level_max > 0, "Maximum level must be a positive integer"

    @property
    def scouts(self) -> int:
        return self.__Nbee

    def __new_unique_scout(self) -> Base_Solution:
        count = 0
        while True:
            S = Base_Solution(self.space, self.space.random_nearby_coordinate(self.__best.coordinate, self.sigma * self.__scale))

            if self.uniqueness == 0:
                break
            if self.optimiser.lookup(S) is None:
                break
            count = count + 1
            if count == self.uniqueness:
                S = None
                break
        return S

    def scout(self) -> Tuple[Base_Solution, np.uint32]:
        S = None

        Nfails = 0
        for b in range(0, self.__Nbee):
            if self.__level == -1:
                self.__best = self.optimiser.select_solution(10)
                if self.__best is None:
                    if self.optimiser.noisy:
                        print(" - BA_Patch::scout: No starting solution available?")
                else:
                    self.__scale = 1
                    self.__level = 0
                    Nfails = 0

            if self.__best is None:
                S, rank = super().scout()
                continue

            S = self.__new_unique_scout()
            if S is None:
                if self.optimiser.noisy:
                    print(" - BA_Patch::scout: No nearby solution - resetting")
                self.__level = -1
                continue
            rank = self.optimiser.evaluate_and_record(S)
            if self.optimiser.noisy and rank == 0:
                print("[BA({l})]".format(l=self.__level), end="")

            old_cost = self.__best.cost
            new_cost = S.cost
            if np.any(new_cost < old_cost):
                self.__best = S
                Nfails = 0
            else:
                Nfails = Nfails + 1
            
        if Nfails == self.__Nbee:
            self.__level = self.__level + 1
            if self.__level == self.__l_max:
                self.__level = -1
            else:
                self.__scale = self.__scale / 2

        return S, rank

