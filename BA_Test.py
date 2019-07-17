import numpy as np

from BeesEtAl.BA_Garden import BA_Garden
from BeesEtAl.BA_Plotter import BA_Plotter
from BeesEtAl.MartinGaddy import MartinGaddy
from BeesEtAl.Schwefel import Schwefel
from BeesEtAl.Viennet import Viennet

test = 'Viennet'
#test = 'Martin-Gaddy'
#test = 'Schwefel'

def MG_norm(cost):
    return np.arctan(cost[0]) * 2 / np.pi

def Schwefel_norm(cost):
    return np.arctan((cost[0] + 2513.9) / 2513.9) * 2 / np.pi

def Viennet_norm(cost):
    return np.arctan(np.linalg.norm(cost - [0,15,-0.1])) * 2 / np.pi

# design variable ranges

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
method = 'gauss'
Nfail  = 6    # i.e., stops at 6th failure
rf     = 0.01
r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method)
params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'neighborhood': method, 'dynamic': True }

G.set_search_params(**params)

# the cost function must subclass BA_Coster

if test == 'Martin-Gaddy':
    G.costfn = MartinGaddy(G)
    norm_fn = MG_norm

if test == 'Schwefel':
    G.costfn = Schwefel(G)
    norm_fn = Schwefel_norm

if test == 'Viennet':
    G.costfn = Viennet(G)
    norm_fn = Viennet_norm

for it in range(1, 101):
    solver_runs = G.iterate()
    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

if test == 'Viennet':
    the_dominant, the_front = G.pareto()
    print('Dominant  = {t}'.format(t=the_dominant))
    print('The Front = {t}'.format(t=the_front))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = G.record[the_front,1]
    Y = G.record[the_front,2]
    Z = G.record[the_front,3]
    ax.scatter(X, Y, Z)
    plt.show()
else:
    G.flush_history()
    P.history((45, 315), 'blue', norm_fn)
    P.save('test.png')
