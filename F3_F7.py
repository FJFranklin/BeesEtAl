import numpy as np

from BeesEtAl.F3_Garden   import F3_Garden
from BeesEtAl.F3_Plotter  import F3_Plotter
from BeesEtAl.Base_Coster import Base_Coster
from BeesEtAl.MartinGaddy import MartinGaddy
from BeesEtAl.Schwefel    import Schwefel
from BeesEtAl.Viennet     import Viennet

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

#minima, maxima = Test_F7.extents(30)
#minima, maxima = MartinGaddy.extents()
#minima, maxima = Schwefel.extents()
minima, maxima = Viennet.extents()

G = F3_Garden(minima, maxima)
P = F3_Plotter(G, [0, 1])

#G.costfn = Test_F7(G)
#G.costfn = MartinGaddy(G)
#G.costfn = Schwefel(G)
G.costfn = Viennet(G)

#G.costfn.verbose = True

method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
params = { 'neighborhood': method }

G.set_search_params(**params)

for it in range(1, 101):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    #print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))
    print('Iteration {:4d}: Global best = {c}'.format(it, c=best_cost))

if len(best_cost) > 2:
    P.pareto([0,1,2])
