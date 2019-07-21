import csv

import numpy as np

from BeesEtAl.F3_Garden   import F3_Garden
from BeesEtAl.F3_Plotter  import F3_Plotter
from BeesEtAl.Base_Coster import Base_Coster

# This is an example of a function that BA handles very poorly

class Test_F7(Base_Coster):
    """
    Function F7 from Gholami & Mohammadi FA-BA Hybrid paper
    """

    @staticmethod
    def extents(Ndim):
        return -1.28 * np.ones(Ndim), 1.28 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA, 4) * np.asarray(range(1, 1 + len(self.XA)))) + np.random.rand(1)

    def meso(self):
        None

Ndim = 30
minima, maxima = Test_F7.extents(Ndim)

G = F3_Garden(minima, maxima, [1,4,3])
G.costfn = Test_F7(G)

P = F3_Plotter(G)
P.surface([0,1], 100, np.zeros(Ndim))

method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
params = { 'neighborhood': method }
G.set_search_params(**params)

max_solver_runs = 1000
solver_runs = 0
it = 0
while solver_runs < max_solver_runs:
    solver_runs = G.iterate(max_solver_runs)
    best_cost, best_X = G.best()

    it = it + 1
    print('Iteration {:4d}: Global best = {c}'.format(it, c=best_cost))
