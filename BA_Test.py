import argparse

import numpy as np

from BeesEtAl.MartinGaddy import MartinGaddy
from BeesEtAl.Schwefel    import Schwefel
from BeesEtAl.Viennet     import Viennet

parser = argparse.ArgumentParser(description="Simple tests for the Bees Algorithm.")

parser.add_argument('--function',     help='Function to be optimised [Martin-Gaddy].',                     default='Martin-Gaddy', choices=['Martin-Gaddy', 'Schwefel', 'Viennet'])
parser.add_argument('--iterations',   help='How many iterations to do [100].',                             default=100, type=int)
parser.add_argument('--neighborhood', help='Shape of neighborhood [gauss].',                               default='gauss', choices=['gauss', 'cube', 'ball', 'sphere'])
parser.add_argument('--fail-at',      help='Abandon patch at specified number of failures [6].',           default=6, type=int)
parser.add_argument('--non-dynamic',  help='Do not update patch before all new bees have been evaluated.', action='store_true')
parser.add_argument('--history-out',  help='Save expanded patch plot history to specified file.',          default=None, dest='out', type=str)
parser.add_argument('--f3',           help='Use F3 optimiser instead of BA.',                              action='store_true')

args = parser.parse_args()

test = args.function
save = args.out
hood = args.neighborhood
fail = args.fail_at
Nit  = args.iterations

if args.non_dynamic:
    bDynamic = False
else:
    bDynamic = True

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
    minima, maxima = MartinGaddy.extents()
    plotaxes = [0, 1]

if test == 'Schwefel':
    minima, maxima = Schwefel.extents(2)
    plotaxes = [5, 2]

if test == 'Viennet':
    minima, maxima = Viennet.extents()
    plotaxes = [0, 1]

if args.f3:
    from BeesEtAl.F3_Garden   import F3_Garden
    from BeesEtAl.F3_Plotter  import F3_Plotter

    flies_bees = [2,6,2]

    G = F3_Garden(minima, maxima, flies_bees)
    P = F3_Plotter(G, plotaxes)

    params = { 'neighborhood': hood }
else:
    from BeesEtAl.BA_Garden   import BA_Garden
    from BeesEtAl.BA_Plotter  import BA_Plotter

    priorities = [5,2,2,1]
    # priorities = [5,2,2,1]; # 3 active patches, one extra scout
    # the last item is the number of extra scouts
    # the other items are the number of bees in each elite patch

    G = BA_Garden(minima, maxima, priorities)
    P = BA_Plotter(G, plotaxes)

    # initial radius, cooling factor, number of failures allowed, etc.
    rf     = 0.01    # smallest patch radius
    r0, sf = G.initial_radius_and_shrinking(fail, rf, hood) # or set your own initial radius & shrinking factor
    params = { 'radius': r0, 'shrink': sf, 'fail_at': fail, 'neighborhood': hood, 'dynamic': bDynamic }

G.set_search_params(**params)

# you can also specify that only a subset of the design variables are to
# be varied by the optimiser and that the rest should have default values:
#G.set_mask_and_defaults([1,0], [0,6])

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

for it in range(1, (Nit+1)):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

# ==== Plot the results ====

if not args.f3:
    # first an expanded patch plot:

    G.flush_history()
    P.history((45, 315), 'blue', norm_fn)
    if save is not None:
        P.save(save)

# for multi-objective optimisation, i.e., when the cost is not a scalar, it's more interesting
# to look at the set of pareto-optimal solutions; you can choose two or three cost indices to plot

if test == 'Viennet':
    P.pareto([0,1,2])
    P.sync(10)
