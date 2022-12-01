from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull

from .Base import Base_Space, Base_Solution, Base_Problem, Base_Optimiser

class Base_Scout(object):
    __opt: Base_Optimiser
    __space: Base_Space
    __uniqueness: int   # set to zero to disable testing for uniqueness; otherwise, number of attempts
    __sigma: np.double

    def __init__(self, optimiser: Base_Optimiser) -> None:
        self.__opt = optimiser
        self.__space = optimiser.space
        self.__uniqueness: int = 100
        self.__sigma = 1.0

    @property
    def optimiser(self) -> Base_Optimiser:
        return self.__opt

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
        if rank == 0:
            print("    Base_Scout: {cst}".format(cst=S.cost))
        return S, rank

class FrontierScout(Base_Scout):
    def __init__(self, optimiser: Base_Optimiser) -> None:
        Base_Scout.__init__(self, optimiser)

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
        if rank == 0:
            print("    FrontierScout: {cst}".format(cst=S.cost))
        return S, rank

class CascadeScout(FrontierScout):
    def __init__(self, optimiser: Base_Optimiser) -> None:
        Base_Scout.__init__(self, optimiser)

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
        if rank == 0:
            print("    CascadeScout: {cst}".format(cst=S.cost))
        return S, rank
