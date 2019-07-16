import math

import numpy as np

from .BA_Patch import BA_Patch
from .Base_Optimiser import Base_Optimiser

class BA_Garden(Base_Optimiser):

    def __init__(self, range_minima, range_maxima, priorities):
        Base_Optimiser.__init__(self, range_minima, range_maxima)

        self.prior    = priorities                # no. of bees assigned to retained patches, + no. scouts
                                                  # e.g., [3,3,2,2,1]
        self.Npatch   = len(self.prior) - 1       # maximum number of retained flower patches
        self.Nscout   = self.prior[self.Npatch]   # minimum number of scouts on global search
        self.Ntotal   = self.Npatch + self.Nscout # Npatch + Nscout

        self.Nactive  = 0      # number of retained flower patches
        self.patch    = []     # cell array of patches
        self.sequence = 0      # number identifying each search sequence

        self.radius   = 0.1    # search radius for new patches
        self.cooling  = 0.75   # cooling factor for failed searches
        self.Nfails   = 6      # maximum number of failures to allow
        self.dynamic  = True   # whether to update patch immediately on finding better solution

        # a superclass property:
        # self.method = 'ball' # neighborhood shape: ball, sphere, cube, gauss

        self.rfinal   = self.radius * self.cooling**(self.Nfails - 1)

        self.history  = []   # complete reported history of cost evolution (no corresponding solution, however)

        for p in range(0, self.Ntotal):
            self.patch.append(BA_Patch(self, p + 1))

    def iterate(self, max_solver_runs=None, **kwargs):
        if max_solver_runs is not None:
            if self.Nrecord >= max_solver_runs:
                return self.Nrecord # i.e., the number of distinct evaluations of the cost function

        if 'override' in kwargs:
            prior = kwargs['override']
        else:
            prior = self.prior

        bests = np.zeros((self.Ntotal, self.Ndim))
        count = 0
        for p in range(0, self.Nactive):
            if max_solver_runs is not None:
                if self.Nrecord >= max_solver_runs:
                    break
            Nbees = prior[p]
            Nbmax = None
            if max_solver_runs is not None:
                Nbmax = max_solver_runs - self.Nrecord
                Nbees = min([Nbees, Nbmax])
            if self.patch[p].Nfails < self.Nfails:
                best_before = self.patch[p].old_cost
                bests[p,:] = self.patch[p].new_local_search(Nbees)
                best_after  = self.patch[p].old_cost
            else:
                self.scout.schedule(Nbees)
                self.scout.evaluate(Nbmax)
                # Hmm... going nowhere; let's abandon this location
                self.sequence = self.sequence + 1
                bests[p,:] = self.patch[p].new_global_search(self.sequence, 'abandoned')
            count = p + 1

        for p in range(self.Nactive, self.Ntotal): # scouts
            if max_solver_runs is not None:
                if self.Nrecord >= max_solver_runs:
                    break
            if p < self.Npatch:
                Nbees = prior[p]
            else:
                Nbees = 1
            Nbmax = None
            if max_solver_runs is not None:
                Nbmax = max_solver_runs - self.Nrecord
                Nbees = min([Nbees, Nbmax])
            self.scout.schedule(Nbees)
            self.scout.evaluate(Nbmax)
            self.sequence = self.sequence + 1
            bests[p,:] = self.patch[p].new_global_search(self.sequence, 'dropped')
            count = p + 1

        costs = self.lookup(bests)[0] # really the indices, but these act as costs for sorting

        isort = np.argsort(costs[0:count])
        psort = []
        for p in range(0, count):
            psort.append(self.patch[isort[p]])
        for p in range(count, self.Ntotal):
            psort.append(self.patch[p])
        self.patch = psort
        self.Nactive = self.Npatch

        return self.Nrecord # i.e., the number of distinct evaluations of the cost function

    def report(self, seq_no, seq_term, seq_hist):
        self.history.append((seq_no, seq_term, seq_hist))

    def flush_history(self):
        for p in range(0, self.Ntotal):
            self.patch[p].flush_history()

    def set_search_params(self, **kwargs):
        self._set_base_params(kwargs)

        if 'radius' in kwargs:
            self.radius   = kwargs['radius']
        if 'shrink' in kwargs:
            self.cooling  = kwargs['shrink']
        if 'fail_at' in kwargs:
            self.Nfails   = kwargs['fail_at']
        if 'dynamic' in kwargs:
            self.dynamic  = kwargs['dynamic']

        self.rfinal = self.radius * self.cooling**(self.Nfails - 1)

    def initial_radius_and_shrinking(self, Nfail, final_radius, neighborhood=None):
        bBeta = False
        bCube = False
        Nbee  = sum(self.prior)

        if neighborhood is not None:
            if neighborhood == 'beta':
                bBeta = True
            elif neighborhood == 'cube':
                bCube = True

        if bBeta:
            # Adapted beta distribution - maximum std dev. is a half
            r0 = 0.5
        elif bCube:
            # Divide the unit cube up into Nbee equal cubes, where:
            r0 = (1 / Nbee)**(1 / self.Ndim) / 2
        else:
            # Divide the unit cube up into Nbee equal spheres, where:
            r0 = (math.gamma(1 + self.Ndim / 2) / Nbee)**(1 / self.Ndim) / math.sqrt(math.pi)

        sf = (final_radius / r0)**(1 / (Nfail - 1))
        return r0, sf
