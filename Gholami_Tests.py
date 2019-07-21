import csv

import numpy as np

from BeesEtAl.Gholami     import Gholami_TestFunction_Extents, Gholami_TestFunction_Coster

from BeesEtAl.BA_Garden   import BA_Garden
from BeesEtAl.F3_Garden   import F3_Garden

Ndim     = 30    # Number of dimensions for solution space
Nt       = 100   # How many times to average this over
duration = 500   # Maximum number of cost function evaluations
e_per_it = 24    # Evalutions per iteration

cases    = [
    ('F3', [ 6, 18]),
    ('F3', [12, 12]),
    ('F3', [18,  6]),
    ('F3', [ 2,  6,  2]),
    ('F3', [ 1,  4,  3]),
    ('BA', [ 6,  6,  3,  3,  6]) ]

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
