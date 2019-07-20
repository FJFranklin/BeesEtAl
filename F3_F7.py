import csv

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

minima, maxima = Test_F7.extents(30)
#minima, maxima = MartinGaddy.extents()
#minima, maxima = Schwefel.extents()
#minima, maxima = Viennet.extents()

max_solver_runs = 990
flies_bees = [2, 3, 2]
params = { 'neighborhood': 'gauss' }
Nit = 55
Nt = 100
history = np.zeros((Nit,4))

for t in range(0, Nt):
    G = F3_Garden(minima, maxima, flies_bees)

    G.costfn = Test_F7(G)
    #G.costfn = MartinGaddy(G)
    #G.costfn = Schwefel(G)
    #G.costfn = Viennet(G)

    G.set_search_params(**params)

    solver_runs = 0
    it = 0
    while solver_runs < max_solver_runs:
        solver_runs = G.iterate(max_solver_runs)
        best_cost, best_X = G.best()

        if t == 0:
            history[it,0] = solver_runs
            history[it,1] = best_cost[0]
            history[it,2] = best_cost[0] / Nt
            history[it,3] = best_cost[0]
        else:
            if history[it,1] > best_cost[0]:
                history[it,1] = best_cost[0]
            history[it,2] = history[it,2] + best_cost[0] / Nt
            if history[it,3] < best_cost[0]:
                history[it,3] = best_cost[0]

        it = it + 1
        print('Iteration {:4d}: Global best = {c}'.format(it, c=best_cost))
        if it == Nit:
            break

with open('stats-12-6-2G.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for r in range(0, Nit):
        writer.writerow(history[r,:])
