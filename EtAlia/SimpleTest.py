import argparse

import numpy as np

from EtAlia.Simple import SimpleSpace, SimpleOptimiser, SimpleProblem
from EtAlia.Tests  import Gholami, Viennet, YueQuLiang

parser = argparse.ArgumentParser(description="Misc. test functions 1-7.")

parser.add_argument('--iterations',    help='How many iterations (not evaluations) to do [1000].',             default=1000, type=int)
parser.add_argument('--test',          help='Test function: 1. Gholami-1 2. Viennet 3. YueQuLiang-1 [1]',      default=1,    type=int)
parser.add_argument('--dimension',     help='What dimension of space should be used (where relevant) [30].',   default=30,   type=int)
parser.add_argument('--trim',          help='Periodically trim history to [50] solutions (or 0 to keep all).', default=50,   type=int)

args = parser.parse_args()

decimals = 0

if args.test == 1: # test == 'Gholami-1':
    Ndim = args.dimension
    test_no = 1
    function = Gholami(Ndim, test_no)
elif args.test == 2: # test == 'Viennet':
    decimals = 1
    function = Viennet()
elif args.test == 3: # 'YueQuLiang-1':
    decimals = 2
    test_no = 1
    function = YueQuLiang(test_no)

extents = function.extents()
space = SimpleSpace(extents)
problem = SimpleProblem(space, function)
optimiser = SimpleOptimiser(problem)

sigma = 1/6
for it in range(0, args.iterations):
    if it % 100 == 99:
        print(".", flush=True)
    else:
        print(".", flush=True, end="")

    if args.trim > 0:
        if it % 100 == 99:
            optimiser.trim_history(args.trim)

    space.granularity = decimals + int(it/100) # no. decimal places
    optimiser.iterate(sigma)
    sigma = sigma * 0.999

print("================================================================================")
if optimiser.cascade is not None:
    optimiser.cascade.rank_print()
else:
    optimiser.print(True)
print("================================================================================")

if optimiser.cascade is not None:
    sols, cost = optimiser.pareto_solutions()

    hull = optimiser.cascade.hull

    import matplotlib.pyplot as plt

    if cost.shape[1] == 2:
        plt.scatter(cost[:,0], cost[:,1], marker='o')

        if hull is not None:
            points = optimiser.cascade.pts
            origin_index = len(points) - 1
            for simplex in hull.simplices:
                if origin_index in simplex:
                    continue
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    if cost.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cost[:,0], cost[:,1], cost[:,2], marker='o')

        if hull is not None:
            points = optimiser.cascade.pts
            origin_index = len(points) - 1
            for simplex in hull.simplices:
                if origin_index in simplex:
                    continue
                x = points[simplex, 0]
                y = points[simplex, 1]
                z = points[simplex, 2]
                ax.plot([x[0],x[1]], [y[0],y[1]], [z[0],z[1]], 'k-')
                ax.plot([x[1],x[2]], [y[1],y[2]], [z[1],z[2]], 'k-')
                ax.plot([x[2],x[0]], [y[2],y[0]], [z[2],z[0]], 'k-')

        ax.set_xlabel('Cost 1')
        ax.set_ylabel('Cost 2')
        ax.set_zlabel('Cost 3')

    plt.show()
