import argparse

import numpy as np

from pymoo.core.problem import Problem
from pymoo.problems import get_problem

from EtAlia.Simple import SimpleSpace, SimpleOptimiser, SimpleProblem, SimpleTestFunction
from EtAlia.Scout  import Base_Scout, FrontierScout, CascadeScout, BA_Patch

parser = argparse.ArgumentParser(description="Uses PyMoo DTLZ test functions 1-7.")

parser.add_argument('--dimension',     help='What dimension of space should be used [12].',                   default=12,   type=int)
parser.add_argument('--iterations',    help='How many iterations (not evaluations) to do [1000].',            default=1000, type=int)
parser.add_argument('--dtlz',          help='Which test function to use (1-7) [4].',                          default=4,    type=int)

args = parser.parse_args()

class DTLZ(SimpleTestFunction):
    '''
    There are 7+ DTLZx problems currently implemented in PyMoo
    https://pymoo.org/problems/many/dtlz.html
    '''
    __Ndim: int    # dimension of solution space
    __pf: Problem  # PyMoo test function

    def __init__(self, Ndim: int, Nobj: int, test_number: int) -> None:
        SimpleTestFunction.__init__(self)
        self.__Ndim = Ndim
        assert (0 < Ndim) and (0 < Nobj), "Numbers of dimensions and objects (costs) must be positive integers"
        assert (0 < test_number) and (test_number <= 7), "Test number should be between 1 and 7"
        test_name = "dtlz" + str(test_number)
        self.__pf = get_problem(test_name, n_var=Ndim, n_obj=Nobj)

    def extents(self) -> np.ndarray:
        return np.ones((2,self.__Ndim)) * np.asarray([[0],[1]])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        cost = self.__pf.evaluate(np.asarray([x]))
        return np.around(cost[0], 6)

Ndim = args.dimension
Nobj = 3
test_no = args.dtlz
function = DTLZ(Ndim, Nobj, test_no)

B = Base_Scout()
F = FrontierScout()
C = CascadeScout()
BA = BA_Patch(3, 5)
scouts = [(B, 2), (F, 3), (BA, 1), (C, 2)]

extents = function.extents()
space = SimpleSpace(extents)
problem = SimpleProblem(space, function)
optimiser = SimpleOptimiser(problem, scouts)

sigma = 1/6
for it in range(0, args.iterations):
    if it % 100 == 99:
        print(".", flush=True)
    else:
        print(".", flush=True, end="")

    space.granularity = 1 + int(it/100) # no. decimal places
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
            for simplex in hull.simplices:
                print(simplex)
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
