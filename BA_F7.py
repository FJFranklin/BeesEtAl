import numpy as np

from BeesEtAl.BA_Garden   import BA_Garden
from BeesEtAl.BA_Plotter  import BA_Plotter
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

def F7_norm(cost):
    return np.arctan(cost[0]) * 2 / np.pi

minima, maxima = Test_F7.extents(30)

G = BA_Garden(minima, maxima, [5,2,2,1])
P = BA_Plotter(G, [0, 1])

G.costfn = Test_F7(G)

method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
Nfail  = 6       # i.e., stops at 6th failure
rf     = 0.01    # smallest patch radius
r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor
params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'neighborhood': method, 'dynamic': True }

G.set_search_params(**params)

for it in range(1, 101):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

G.flush_history()
P.history((-85, 265), 'blue', F7_norm)
