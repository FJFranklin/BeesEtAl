import argparse
import csv
import sys

import numpy as np

from BeesEtAl.Gholami   import Gholami_TestFunction_Extents, Gholami_TestFunction_Coster
from BeesEtAl.BA_Garden import BA_Garden
from BeesEtAl.F3_Garden import F3_Garden

parser = argparse.ArgumentParser(description="Runs the twelve Gholami test functions for convergence statistics.")

parser.add_argument('--dimension', help='What dimension of space should be used [30].',                     default=30,    type=int)
parser.add_argument('--duration',  help='Duration, i.e., how many evaluations to end at [10000].',          default=10000, type=int)
parser.add_argument('--suppress',  help='In case of F3, suppress diversity for specified no. evaluations.', default=0,     type=int)
parser.add_argument('--repeats',   help='How many times to repeat each case [100].',                        default=100,   type=int)
parser.add_argument('--plot',      help='Create a surface plot of the specified function (1-12).',          default=0,     type=int)

args = parser.parse_args()

Ndim     = args.dimension
Nt       = args.repeats
duration = args.duration
suppress = args.suppress

if args.plot > 0:
    from BeesEtAl.Base_Optimiser import Base_Optimiser
    from BeesEtAl.Base_Plotter   import Base_Plotter

    if args.plot <= 12:
        minima, maxima = Gholami_TestFunction_Extents(args.plot, Ndim)
        BO = Base_Optimiser(minima, maxima)
        P  = Base_Plotter(BO, None)
        X0 = np.zeros(Ndim)
        BO.costfn = Gholami_TestFunction_Coster(args.plot, BO)
        P.surface([0,1], 100, X0)
        P.sync(10)

    sys.exit()

cases    = [
    ('F3', [ 6, 18]),
    ('F3', [12, 12]),
    ('F3', [18,  6]),
    ('F3', [ 2,  6,  2]),
    ('F3', [ 1,  4,  3]) ]
#    ('BA', [ 6,  6,  3,  3,  6]) ]
#    ('F3', [ 1,  9,  2]) ]

e_per_it = 24    # Evalutions per iteration
Ncol     = 13
Nrow     = int(duration / e_per_it)
max_runs = Nrow * e_per_it

for c in cases:
    history = np.zeros((Nrow,Ncol))
    history[:,0] = np.asarray(range(1,Nrow+1)) * e_per_it

    solver, flies_bees = c

    file_name = solver
    for fb in flies_bees:
        file_name = file_name + '-' + str(fb)
    file_name = file_name + '.csv'

    for number in range(1, 13):
        minima, maxima = Gholami_TestFunction_Extents(number, Ndim)

        for t in range(0, Nt):
            if solver == 'F3':
                G = F3_Garden(minima, maxima, flies_bees)
            else:
                G = BA_Garden(minima, maxima, flies_bees)

            G.costfn = Gholami_TestFunction_Coster(number, G)

            # For BA & F3:
            method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
            params = { 'neighborhood': method }
            G.set_search_params(**params)

            # For BA only:
            if solver == 'BA':
                Nfail  = 6       # i.e., stops at 6th failure
                rf     = 0.01    # smallest patch radius
                r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor
                params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'dynamic': True }
                G.set_search_params(**params)

            solver_runs = 0
            it = 0
            while solver_runs < max_runs:
                if solver == 'F3' and solver_runs < suppress:
                    solver_runs = G.iterate(max_runs, unisex=True)
                else:
                    solver_runs = G.iterate(max_runs)

                best_cost, best_X = G.best()

                if t == 0:
                    history[it,number] = best_cost[0] / Nt
                else:
                    history[it,number] = history[it,number] + best_cost[0] / Nt

                it = it + 1
                print('Iteration {:4d}: Best = {c} [{f} {n} / {t}]'.format(it, c=best_cost, f=file_name, n=number, t=t))
                if it == Nrow:
                    break

    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['', *range(1, 13)])
        for r in range(0, Nrow):
            writer.writerow(history[r,:])
