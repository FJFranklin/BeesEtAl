import numpy as np

from .F3_Fly         import F3_Fly
from .Base_Optimiser import Base_Optimiser

class F3_Garden(Base_Optimiser):

    def __init__(self, range_minima, range_maxima, flies_bees):
        Base_Optimiser.__init__(self, range_minima, range_maxima)

        self.flies_bees  = flies_bees

        if len(self.flies_bees) == 3:
            if self.flies_bees[2] == 2:
                self.genders = ['F', 'M']
                self.orients = [['M'], ['F'], ['F', 'M']]
            else:
                self.genders = ['N', 'F', 'M']
                self.orients = [['M', 'N'], ['N', 'F'], ['F', 'M'], ['M', 'N', 'F']]
            self.trans   = 0.1    # probability of gender transition
            self.diverse = True
        else:
            self.trans   = 0
            self.genders = ['N']
            self.orients = [['N']]
            self.diverse = False

        self.Nflies  = 0
        self.flies   = []

        self.fly_rmin   = 0.01    # search radius - minimum
        self.fly_rmax   = self.principal_radius(self.flies_bees[0] * len(self.genders) * len(self.orients))

        self.bee_radius = 0.01
        self.bee_shells = 20

        self.jitter     = None    # neighborhood size following firefly movement, or None for radius / 2
        self.attract    = 'exp'   # nature of attraction, i.e., 'exp' for exp(-r), 'gauss' for exp(-r2)

    def iterate(self, max_solver_runs=None, **kwargs):
        if self.plotter is not None:
            self.plotter.new()

        if 'unisex' in kwargs:    # option to temporarily suppress gender & orientation
            unisex = kwargs['unisex']
        else:
            unisex = not self.diverse

        Nrecord_at_end = self.Nrecord + self.__evaluations_per_iteration()
        if max_solver_runs is not None:
            Nrecord_at_end = min(Nrecord_at_end, max_solver_runs)

        if self.Nflies == 0:
            if self.diverse:
                for g in self.genders:
                    for o in self.orients:
                        for i in range(0, self.flies_bees[0]):
                            self.Nflies = self.Nflies + 1
                            self.flies.append(F3_Fly(self, self.Nflies, g, o))
            else:
                self.Nflies = self.flies_bees[0]

                for i in range(0, self.Nflies):
                    self.flies.append(F3_Fly(self, i + 1, self.genders[0], self.orients[0]))

            self.scout.schedule(self.Nflies)
            self.scout.evaluate(self.Nflies)

            for f in self.flies:
                f.new_global_search()
        else:
            for fi in self.flies:
                if self.Nrecord >= Nrecord_at_end:
                    break

                social_set = [fi]
                for fj in self.flies:
                    if fi.id_no == fj.id_no:
                        continue
                    if (fj.gender in fi.orientation) or unisex:
                        social_set.append(fj)

                rank_abs = []
                for s in social_set:
                    rank_abs.append(self.lookup(s.best_X)[0])

                rank_set = np.asarray(rank_abs)
                rank_set[np.argsort(rank_abs)] = range(0, len(rank_abs))

                radius = self.fly_rmin
                if len(rank_set) > 1:
                    radius = radius * (self.fly_rmax / self.fly_rmin)**(rank_set[0] / (len(rank_set) - 1))

                if self.jitter is None:
                    fi.new_local_search(social_set, rank_set, radius, radius / 2)
                else:
                    fi.new_local_search(social_set, rank_set, radius, self.jitter)

        chosen_few = []

        for g in self.genders:
            gender_set = []
            for f in self.flies:
                if (f.gender == g) or unisex:
                    gender_set.append(f)

            if len(gender_set) > 0:
                rank_abs = []
                for s in gender_set:
                    rank_abs.append(self.lookup(s.best_X)[0])

                chosen_few.append(gender_set[np.argsort(rank_abs)[0]])

            if unisex:
                break

        while self.Nrecord < Nrecord_at_end:
            for c in chosen_few:
                if self.Nrecord >= Nrecord_at_end:
                    break

                c.bees(min([self.flies_bees[1], (Nrecord_at_end - self.Nrecord)]))

        if self.plotter is not None:
            self.plotter.done()

        return self.Nrecord

    def attraction(self, dX, radius):
        if self.attract == 'gauss':
            attr = dX * np.exp2(1 - (np.linalg.norm(self.translate_to_unit_cube(dX)) / radius)**2) 
        else:
            attr = dX * np.exp2(-np.linalg.norm(self.translate_to_unit_cube(dX)) / radius)
        return attr

    def transition(self, g): # not, of course, reflective of human reality
        if np.random.rand(1) < self.trans:
            if self.flies_bees[2] == 2:
                if g == 'M':
                    g = 'F'
                else:
                    g = 'M'
            else:
                if g == 'M':
                    g = 'N'
                elif g == 'N':
                    g = 'F'
                else:
                    g = 'M'

            if self.costfn.verbose:
                print('> transition: gender -> {g} <'.format(g=g))

        return g

    def __evaluations_per_iteration(self):
        return self.flies_bees[0] * len(self.genders) * len(self.orients) + self.flies_bees[1] * len(self.genders)

    def set_search_params(self, **kwargs):
        self._set_base_params(kwargs)

        if 'attraction' in kwargs:     # 'exp' or 'gauss'
            self.attract = kwargs['attraction']
        if 'bee-radius' in kwargs:     # > 0
            self.bee_radius = kwargs['bee-radius']
        if 'bee-shells' in kwargs:     # integer > 0
            self.bee_shells = kwargs['bee-shells']
        if 'fly-radius-min' in kwargs: # > 0
            self.fly_rmin = kwargs['fly-radius-min']
        if 'fly-radius-max' in kwargs: # > 0
            self.fly_rmax = kwargs['fly-radius-max']
        if 'jitter' in kwargs:         # > 0
            self.jitter = kwargs['jitter']
