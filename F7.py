import argparse

import numpy as np

from BeesEtAl.Base_Coster import Base_Coster

# A noisy function that BA handles poorly

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

parser = argparse.ArgumentParser(description="Runs the twelve Gholami test functions for convergence statistics.")

parser.add_argument('-o', '--optimiser', help='Select optimiser [BA].',                       default='BA', choices=['BA', 'F3'])
parser.add_argument('--dimension',       help='What dimension of space should be used [30].', default=30,   type=int)
parser.add_argument('--iterations',      help='How many iterations to do [100].',             default=100,  type=int)
parser.add_argument('--no-plot',         help='Do not plot.',                                 action='store_true')

args = parser.parse_args()

if args.no_plot:
    bPlot = False
else:
    bPlot = True

P = None

minima, maxima = Test_F7.extents(args.dimension)

if args.optimiser == 'BA':
    from BeesEtAl.BA_Garden import BA_Garden

    G = BA_Garden(minima, maxima, [5,2,2,1])

    if bPlot:
        from BeesEtAl.BA_Plotter import BA_Plotter
        P = BA_Plotter(G, [0, 1])

    method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
    Nfail  = 6       # i.e., stops at 6th failure
    rf     = 0.01    # smallest patch radius
    r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor
    params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'neighborhood': method, 'dynamic': True }

elif args.optimiser == 'F3':
    from BeesEtAl.F3_Garden import F3_Garden

    G = F3_Garden(minima, maxima, [5,5])

    if bPlot:
        from BeesEtAl.F3_Plotter import F3_Plotter
        P = F3_Plotter(G, [0, 1])

    method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
    params = { 'neighborhood': method }

G.costfn = Test_F7(G)
G.set_search_params(**params)

for it in range(1, 1 + args.iterations):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))
