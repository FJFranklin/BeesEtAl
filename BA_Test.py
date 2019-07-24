import numpy as np

from BeesEtAl.BA_Garden   import BA_Garden
from BeesEtAl.BA_Plotter  import BA_Plotter
from BeesEtAl.MartinGaddy import MartinGaddy
from BeesEtAl.Schwefel    import Schwefel
from BeesEtAl.Viennet     import Viennet

test = 'Viennet'
#test = 'Martin-Gaddy'
#test = 'Schwefel'

# These functions normalise the costs to the range 0-1 for the expanded patch plot BA_Plotter.history()

def MG_norm(cost):
    return np.arctan(cost[0]) * 2 / np.pi

def Schwefel_norm(cost):
    return np.arctan((cost[0] + 2513.9) / 2513.9) * 2 / np.pi

def Viennet_norm(cost):
    return np.arctan(np.linalg.norm(cost - [0,15,-0.1])) * 2 / np.pi

# To set up the optimiser, first need to define the design variable ranges and choose patch priorities
# Plotting is optional, but if there are more than two variables, need to decide which to plot

if test == 'Martin-Gaddy':
    minima = np.asarray([ 0,  0])
    maxima = np.asarray([10, 10])

    plotaxes = [0, 1]

if test == 'Schwefel':
    minima = -500 * np.ones(6)
    maxima =  500 * np.ones(6)

    plotaxes = [5, 2]

if test == 'Viennet':
    minima = np.asarray([-3, -3])
    maxima = np.asarray([ 3,  3])

    plotaxes = [0, 1]

priorities =  [5,2,2,1]
# priorities = [5,2,2,1]; # 3 active patches, one extra scout
# the last item is the number of extra scouts
# the other items are the number of bees in each elite patch

G = BA_Garden(minima, maxima, priorities)
P = BA_Plotter(G, plotaxes)

# you can also specify that only a subset of the design variables are to
# be varied by the optimiser and that the rest should have default values:
#G.set_mask_and_defaults([1,0], [0,6])

# initial radius, cooling factor, number of failures allowed, etc.
method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
Nfail  = 6       # i.e., stops at 6th failure
rf     = 0.01    # smallest patch radius
r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor
params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'neighborhood': method, 'dynamic': True }

G.set_search_params(**params)

# the cost function must subclass Base_Coster (which is very easy to do)

if test == 'Martin-Gaddy':
    G.costfn = MartinGaddy(G)
    norm_fn = MG_norm

if test == 'Schwefel':
    G.costfn = Schwefel(G)
    norm_fn = Schwefel_norm

if test == 'Viennet':
    G.costfn = Viennet(G)
    norm_fn = Viennet_norm

# ==== We're ready to optimise ====

for it in range(1, 101):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

# ==== Plot the results ====

# first an expanded patch plot:

G.flush_history()
P.history((45, 315), 'blue', norm_fn)
P.save('test.png')

# for multi-objective optimisation, i.e., when the cost is not a scalar, it's more interesting
# to look at the set of pareto-optimal solutions; you can choose two or three cost indices to plot

if test == 'Viennet':
    # either get pareto dominant/optimal indices; also, save solutions to a file (optional)
    the_dominant, the_front = G.pareto('pareto.csv')
    # or get pareto dominant/optimal indices, and plot selected (2 or 3)
    the_dominant, the_front = P.pareto([0,1,2])
    if the_dominant is not None:
        if len(the_dominant) > 0:
            rank, cost, X = G.get_by_index(the_dominant)
            print('Pareto-dominant solution: cost = {c} @ {x}'.format(c=cost, x=X))
    if the_front is not None:
        if len(the_front) > 0:
            rank, cost, X = G.get_by_index(the_front)
            print('Pareto-optimal solutions: cost = {c} @ {x}'.format(c=cost, x=X))
