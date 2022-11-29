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

        return S, self.__opt.evaluate_and_record(S)

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

        return S, self.optimiser.evaluate_and_record(S)

class CascadeScout(FrontierScout):
    __pts: np.ndarray
    __hull: ConvexHull

    def __init__(self, optimiser: Base_Optimiser) -> None:
        Base_Scout.__init__(self, optimiser)
        self.__pts = None
        self.__hull = None

    @property
    def pts(self) -> np.ndarray:
        return self.__pts

    @property
    def hull(self) -> ConvexHull:
        return self.__hull

    def build(self) -> bool:
        self.__pts = None
        self.__hull = None

        cascade = self.optimiser.cascade
        if cascade is None:
            return False

        sols = cascade.sols
        Nsol = len(sols)
        if Nsol == 0:
            return False

        Ncost = len(sols[0].cost)
        if 1 + Nsol <= Ncost:
            return False

        pts = np.zeros((1+Nsol,Ncost))
        index = 0
        for S in sols:
            pts[index,:] = S.cost
            index = index + 1
        pts[-1,:] = pts[:-1,:].min(axis=0)

        # Project onto unit hypersphere
        ptsn = np.copy(pts) - pts[-1,:]
        norm = np.asarray([np.linalg.norm(ptsn[:-1,:], axis=1)])
        ptsn[:-1,:] = ptsn[:-1,:] / norm.transpose()

        # TODO
        # 1. Must ensure that there are no repeats in the cascade if the history is being trimmed
        # 2. Need to think how to handle degeneracy
        hull = ConvexHull(ptsn)

        self.__pts = pts
        self.__hull = hull
        return True

    def scout(self) -> Tuple[Base_Solution, np.uint32]:
        return super().scout()
