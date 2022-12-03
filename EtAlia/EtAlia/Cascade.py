from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

class ParetoCascade(object):
    __child: 'ParetoCascade'
    __rng: np.random.Generator
    __sols: list
    __wts: np.ndarray
    __pts: np.ndarray
    __hull: ConvexHull
    __dirty: bool

    def __init__(self, base_rng: np.random.Generator, child_cascade: 'ParetoCascade' = None) -> None:
        self.__child = child_cascade
        self.__rng = base_rng
        self.__sols = []
        self.__wts = None
        self.__pts = None
        self.__hull = None
        self.__dirty = True

    @property
    def child(self) -> 'ParetoCascade':
        return self.__child

    @property
    def sols(self) -> list:
        return self.__sols

    @staticmethod
    def check(cost1: np.ndarray, cost2: np.ndarray) -> Tuple[bool,bool]:
        b1Dominates2, b2Dominates1 = True, True
        for d in range(0, cost1.shape[0]):
            if cost1[d] > cost2[d]:
                b1Dominates2 = False
            if cost2[d] > cost1[d]:
                b2Dominates1 = False
        return b1Dominates2, b2Dominates1 # both True if costs are equal

    @property
    def depth(self) -> int:
        if len(self.__sols) == 0:
            return 0
        if self.__child is None:
            return 1
        return 1 + self.__child.depth

    def select_solution(self) -> 'Base_Solution':
        if self.depth == 0:
            return None
        d = np.double(self.depth)
        if d > 1:
            p1 = 2.0 # d
            p2 = 1.0 # d*(d+1)/2
            if self.__rng.choice([0,1],p=[p1/(p1+p2),p2/(p1+p2)]) > 0:
                return self.__child.select_solution()
        if self.__wts is not None:
            return self.__sols[self.__rng.choice(len(self.__sols), p=self.__wts[:-1])]
        return self.__sols[self.__rng.choice(len(self.__sols))]

    def add(self, new_solution: 'Base_Solution') -> np.uint32:
        rank: np.uint32 = 0xFFFFFFFF
        new_cost = new_solution.cost

        set_changed = False
        sols_retain = []
        for sol in self.__sols:
            b1Dominates2, b2Dominates1 = False, False
            if new_solution is not None:
                b1Dominates2, b2Dominates1 = self.check(new_cost, sol.cost)
            if b1Dominates2 and b2Dominates1:
                print("=", end="") # print(" - ParetoCascade::add: * * * unhandled duplicate cost: {c1} = {c2}".format(c1=new_cost, c2=sol.cost)) # FIXME
                sols_retain.append(sol)
            elif b1Dominates2:
                set_changed = True
                if self.__child is not None:
                    self.__child.add(sol)
            else:
                sols_retain.append(sol)
            if b2Dominates1:
                if self.__child is not None:
                    new_rank = self.__child.add(new_solution)
                    if new_rank < rank:
                        rank = new_rank + 1
                new_solution = None

        self.__sols = sols_retain

        if new_solution is not None:
            self.__sols.append(new_solution)
            set_changed = True
            rank = 0
        if set_changed:
            self.__wts = None
            self.__pts = None
            self.__hull = None
            self.__dirty = True
        return rank

    def rank_print(self, cascade: bool = False, rank: int = 0) -> None:
        print("      === Pareto Cascade Rank {r} ===".format(r=rank))
        for sol in self.__sols:
            print('          {cost} @ {soln}'.format(cost=sol.cost, soln=sol.jstr))

        if cascade == True:
            if self.__child is not None:
                self.__child.rank_print(True, rank + 1)

    def __build(self) -> None:
        self.__wts = None
        self.__pts = None
        self.__hull = None
        self.__dirty = False

        sols = self.__sols
        Nsol = len(sols)
        if Nsol == 0:
            return

        Ncost = len(sols[0].cost)
        if 1 + Nsol <= Ncost:
            return

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

        try:
            hull = ConvexHull(ptsn)
            self.__pts = pts
            self.__hull = hull
        except QhullError:
            print("*", end="")

        if self.__hull is not None:
            # Let's measure crowding
            self.__wts = np.zeros(1+Nsol)
            for simplex in self.__hull.simplices:
                Nv = len(simplex)
                for v1 in range(0, Nv-1):
                    for v2 in range(v1+1, Nv):
                        i1 = simplex[v1]
                        if i1 == Nsol: # origin
                            continue
                        i2 = simplex[v2]
                        if i2 == Nsol: # origin
                            continue
                        c1 = self.__pts[i1,:]
                        c2 = self.__pts[i2,:]
                        dc = np.linalg.norm(c2 - c1)
                        self.__wts[i1] = self.__wts[i1] + 1/dc
                        self.__wts[i2] = self.__wts[i2] + 1/dc
            self.__wts = np.max(self.__wts) * 1.1 - self.__wts
            self.__wts[Nsol] = 0
            self.__wts = self.__wts / np.sum(self.__wts)
            #print(">> wts = {w} <<".format(w=self.__wts))

    @property
    def pts(self) -> np.ndarray:
        if self.__dirty:
            self.__build()
        return self.__pts

    @property
    def hull(self) -> ConvexHull:
        if self.__dirty:
            self.__build()
        return self.__hull

