import argparse
import csv
import sys

import numpy as np

from BeesEtAl.Gholami   import Gholami_TestFunction_Extents, Gholami_TestFunction_Coster
from BeesEtAl.BA_Garden import BA_Garden
from BeesEtAl.F3_Garden import F3_Garden

parser = argparse.ArgumentParser(description="Runs the twelve Gholami test functions for convergence statistics.")

parser.add_argument('--dimension',     help='What dimension of space should be used [30].',                  default=30,    type=int)
parser.add_argument('--duration',      help='Duration, i.e., how many evaluations to end at [10000].',       default=10000, type=int)
parser.add_argument('--repeats',       help='How many times to repeat each case [100].',                     default=100,   type=int)
parser.add_argument('--plot',          help='Create a surface plot of the specified function (1-12).',       default=0,     type=int)
parser.add_argument('--prefix',        help='Prefix for output file names.',                                 default='',    type=str)
parser.add_argument('-t', '--test',    help='Test specified function (1-12).',                               default=0,     type=int, nargs='+')
parser.add_argument('--ba-pure',       help='BA: Pure bees algorithm case (6/6/3/3+6).',                     action='store_true')
parser.add_argument('--f3-pure',       help='F3: Pure firefly case (24+0).',                                 action='store_true')
parser.add_argument('--f3-2G',         help='F3: Two-gender case only (2+6;2).',                             action='store_true')
parser.add_argument('--f3-3G',         help='F3: Three-gender case only (1+4;3).',                           action='store_true')
parser.add_argument('--f3-standard',   help='F3: One-gender case only (6+18).',                              action='store_true')
parser.add_argument('--f3-suppress',   help='F3: Suppress diversity for specified no. evaluations (1+9;2).', default=0,     type=int)
parser.add_argument('--f3-bee-shells', help='F3: Specify number of bee shells [20].',                        default=20,    type=int)
parser.add_argument('--f3-bee-radius', help='F3: Specify radius of inner bee shell [0.01].',                 default=0.01,  type=float)
parser.add_argument('--f3-min-radius', help='F3: Specify minimum attraction radius for fireflies [0.01].',   default=0.01,  type=float)
parser.add_argument('--f3-max-radius', help='F3: Specify maximum attraction radius for fireflies.',          default=None,  type=float)
parser.add_argument('--f3-jitter',     help='F3: Specify neighborhood radius for fireflies.',                default=None,  type=float)
parser.add_argument('--f3-attraction', help='F3: Specify exponential or Gaussian attraction [exp].',         default='exp', choices=['exp', 'gauss'])

args = parser.parse_args()

Ndim     = args.dimension
Nt       = args.repeats
duration = args.duration
prefix   = args.prefix
suppress = args.f3_suppress

if args.test == 0:
    test_set = range(1, 13)
else:
    test_set = args.test

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

if args.f3_pure:
    cases    = [
        ('F3', [ 24, 0]) ]
elif args.ba_pure:
    cases    = [
        ('BA', [ 6,  6,  3,  3,  6]) ]
elif args.f3_standard:
    cases    = [
        ('F3', [ 6, 18]) ]
elif args.f3_2G:
    cases    = [
        ('F3', [ 2,  6,  2]) ]
elif args.f3_3G:
    cases    = [
        ('F3', [ 1,  4,  3]) ]
elif suppress > 0:
    cases    = [
        ('F3', [ 1,  9,  2]) ]
else:
    cases    = [
        ('F3', [12, 12]),
        ('F3', [18,  6]) ]

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
    file_name = prefix + file_name + '.csv'

    for number in test_set:
        minima, maxima = Gholami_TestFunction_Extents(number, Ndim)

        for t in range(0, Nt):
            if solver == 'F3':
                G = F3_Garden(minima, maxima, flies_bees)
            else:
                G = BA_Garden(minima, maxima, flies_bees)

            G.costfn = Gholami_TestFunction_Coster(number, G)

            params = {}

            # For BA & F3:
            method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
            params['neighborhood'] = method

            # For F3 only:
            if solver == 'F3':
                params['attraction']     = args.f3_attraction
                params['bee-radius']     = args.f3_bee_radius
                params['bee-shells']     = args.f3_bee_shells
                params['fly-radius-min'] = args.f3_min_radius

                if args.f3_max_radius is not None: # if None, calculated automatically
                    params['fly-radius-max'] = args.f3_max_radius
                if args.f3_jitter is not None:     # if None, calculated automatically
                    params['jitter']         = args.f3_jitter

            # For BA only:
            if solver == 'BA':
                Nfail  = 6       # i.e., stops at 6th failure
                rf     = 0.01    # smallest patch radius
                r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor

                params['radius']  = r0
                params['shrink']  = sf
                params['fail_at'] = Nfail
                params['dynamic'] = True

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
