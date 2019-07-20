import math

import numpy as np

from .Base_Scout import Base_Scout
from .Base_Sorter import Base_Sorter

class Base_Optimiser(Base_Sorter):

    def __init__(self, range_minima, range_maxima):
        Base_Sorter.__init__(self, len(range_minima))

        self.minima    = range_minima              # minima & maxima of design variable ranges
        self.maxima    = range_maxima

        self.plotter   = None                      # Base_Plotter subclass instance, or None

        self.defaults  = np.zeros(self.Ndim)       # default values for masked variables for scouts
        self.mask      = np.ones(self.Ndim)        # mask indicating which variables to include in algorithm

        self.costfn    = None   # subclass of Base_Coster
        self.method    = 'ball' # neighborhood shape: ball, sphere, cube, gauss
        self.rnudge    = 1E-2   # nudge radius - should probably be a lot smaller than 1

        self.scout     = Base_Scout(self)
        self.threshold = 1E-8

    def _set_base_params(self, kwargs):
        if 'nudge-radius' in kwargs:
            self.rnudge    = kwargs['nudge-radius']
        if 'neighborhood' in kwargs:
            self.method    = kwargs['neighborhood']
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']

    def set_mask_and_defaults(self, mask_active, defaults_inactive):
        self.mask     = mask_active
        self.defaults = defaults_inactive

    def translate_to_unit_cube(self, X): # scales only; result not checked
        return (X - self.minima) / (self.maxima - self.minima)

    def translate_from_unit_cube(self, U=None, crop=False):
        if U is None:
            X = np.random.rand(self.Ndim)
        elif crop:
            X = np.minimum(1, np.maximum(0, U))
        else:
            X = np.copy(U)

            # reflect exterior values into the cube
            for ix in range(0, self.Ndim):
                while True:
                    if X[ix] > 1:
                        X[ix] = 2 - X[ix]
                        continue
                    if X[ix] < 0:
                        X[ix] = -X[ix]
                        continue
                    break

        return self.minima + (self.maxima - self.minima) * X

    def n_cube(self):
        # Ndim dimensions in range -1..1
        norm = 0
        while norm < self.threshold:
            cube = -1 + 2 * np.random.rand(self.Ndim)
            norm = np.linalg.norm(cube)
        return cube

    def n_gauss(self, N=None): # N-dimensional Gaussian distribution, i.e., Normal(mu=0,sigma=1)
        if N is None:
            N = self.Ndim

        norm = 0
        while norm < self.threshold:
            gauss = np.random.normal(0, 1, N)
            norm  = np.linalg.norm(gauss)

        return gauss, norm

    def n_ball(self):
        # Dropped coordinates method (Harman and Lacko, 2010; Voelker, 2017) - see:
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        gauss, norm = self.n_gauss(self.Ndim + 2) # an array of (d+2) normally distributed random variables
        return gauss[0:self.Ndim] / norm          # take the first d coordinates

    def n_sphere(self):
        gauss, norm = self.n_gauss(self.Ndim)
        return gauss / norm

    def new_position_in_neighbourhood(self, X0, radius, method=None):
        if method is None:
            method = self.method

        if method == 'cube':
            u = self.n_cube()
        elif method == 'sphere':
            u = self.n_sphere()
        elif method == 'gauss':
            gauss, norm = self.n_gauss()
            u = gauss
        else:
            u = self.n_ball()

        u = u * radius + self.translate_to_unit_cube(X0)

        X = self.translate_from_unit_cube(u)
        X = X0 * (1 - self.mask) + self.mask * X

        return X

    def new_position(self):
        X = self.translate_from_unit_cube()
        X = self.defaults * (1 - self.mask) + self.mask * X

        return X

    def global_best(self):
        return self.best()[0]

    def nudge(self, X0): # where X0 has been evaluated already
        X = None

        rank_X0 = self.lookup(X0)[0]

        if rank_X0 is None: # else oops - no record of this point: assume the worst
            rank_X0 = self.Nrecord

        B = np.zeros(self.Ndim)

        u = self.translate_to_unit_cube(X0)

        for r in range(0, self.Nrecord):
            rank_Xr, cost_Xr, Xr = self.get_by_index(r)
            if rank_X0 == rank_Xr:
                continue

            v = self.translate_to_unit_cube(Xr)
            dc = rank_Xr - rank_X0               # difference in cost / rank
            dB = v - u                           # vector from u towards v
            norm = np.linalg.norm(dB)
            if norm > self.threshold:
                dB = dB / norm                   # unit vector from X towards Y
                wt = np.exp(-norm / self.rnudge) # distance weighting
                B  = B - dc * wt * dB

        norm = np.linalg.norm(B)
        if norm > self.threshold:
            X = self.translate_from_unit_cube(u + self.rnudge * B / norm)

        if X is None:
            X = np.copy(X0)

        return X

    def principal_radius(self, Nentity):
        return (math.gamma(1 + self.Ndim / 2) / Nentity)**(1 / self.Ndim) / math.sqrt(math.pi)
