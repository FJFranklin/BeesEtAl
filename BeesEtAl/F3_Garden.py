import math

import numpy as np

from .F3_Fly         import F3_Fly
from .Base_Optimiser import Base_Optimiser

class F3_Garden(Base_Optimiser):

    def __init__(self, range_minima, range_maxima, unisex=True):
        Base_Optimiser.__init__(self, range_minima, range_maxima)

        self.rmin   = 0.01   # search radius - minimum
        self.rmax   = 1      # search radius - maximum

        self.unisex = unisex

        if self.unisex:
            self.trans = 0 
        else:
            self.trans = 0.1 # probability of gender transition

        self.Nflies = 0
        self.flies  = []

    def iterate(self):
        if self.plotter is not None:
            self.plotter.new()

        if self.unisex:
            genders = ['N']
        else:
            genders = ['N', 'F', 'M']

        if self.Nflies == 0:
            self.scout.schedule(21)
            self.scout.evaluate(21)

            if self.unisex:
                orientations = [['N']]
            else:
                orientations = [['M'], ['N'], ['F'], ['M', 'N'], ['N', 'F'], ['F', 'M'], ['M', 'N', 'F']]

            for g in genders:
                for o in orientations:
                    if self.unisex:
                        per_go = 21
                    else:
                        per_go =  1
                    for i in range(0, per_go):
                        self.Nflies = self.Nflies + 1
                        self.flies.append(F3_Fly(self, self.Nflies, g, o))

            for f in self.flies:
                f.new_global_search()
        else:
            for fi in self.flies:
                social_set = [fi]
                for fj in self.flies:
                    if fi.id_no == fj.id_no:
                        continue
                    if fj.gender in fi.orientation:
                        social_set.append(fj)

                rank_abs = []
                for s in social_set:
                    rank_abs.append(self.lookup(s.best_X)[0])

                rank_set = np.asarray(rank_abs)
                rank_set[np.argsort(rank_abs)] = range(0, len(rank_abs))

                radius = self.rmin
                if len(rank_set) > 1:
                    radius = radius * (self.rmax / self.rmin)**(rank_set[0] / (len(rank_set) - 1))

                fi.new_local_search(social_set, rank_set, radius)

        chosen_few = []

        for g in genders:
            gender_set = []
            for f in self.flies:
                if f.gender == g:
                    gender_set.append(f)

            if len(gender_set) > 0:
                rank_abs = []
                for s in gender_set:
                    rank_abs.append(self.lookup(s.best_X)[0])

                chosen_few.append(gender_set[np.argsort(rank_abs)[0]])

        if len(chosen_few) == 1:
            chosen_few[0].bees(9, self.rmin)
        elif len(chosen_few) == 2:
            chosen_few[0].bees(5, self.rmin)
            chosen_few[1].bees(4, self.rmin)
        else:
            chosen_few[0].bees(3, self.rmin)
            chosen_few[1].bees(3, self.rmin)
            chosen_few[2].bees(3, self.rmin)

        if self.plotter is not None:
            self.plotter.done()

        return self.Nrecord

    def set_search_params(self, **kwargs):
        self._set_base_params(kwargs)
