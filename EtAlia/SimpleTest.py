import numpy as np

from EtAlia.Simple import SimpleSpace, SimpleOptimiser, SimpleProblem, Gholami, Viennet, YueQuLiang

test = 'YueQuLiang-1'

if test == 'Gholami-1':
    Ndim = 30
    test_no = 1
    function = Gholami(Ndim, test_no)
elif test == 'Viennet':
    function = Viennet()
elif test == 'YueQuLiang-1':
    test_no = 1
    function = YueQuLiang(test_no)

extents = function.extents()
space = SimpleSpace(extents)
problem = SimpleProblem(space, function)
optimiser = SimpleOptimiser(problem)

for it in range(0, 1000):
    space.granularity = int(it/100) # no. decimal places
    optimiser.iterate()

print("================================================================================")
if optimiser.cascade is not None:
    optimiser.cascade.rank_print()
else:
    optimiser.print(True)
print("================================================================================")

if optimiser.cascade is not None:
    sols, cost = optimiser.pareto_solutions()
    print(cost.shape)

    if cost.shape[1] == 2:
        import matplotlib.pyplot as plt

        plt.scatter(cost[:,0], cost[:,1], marker='o')
        plt.show()

    if cost.shape[1] == 3:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cost[:,0], cost[:,1], cost[:,2], marker='o')

        ax.set_xlabel('Cost 1')
        ax.set_ylabel('Cost 2')
        ax.set_zlabel('Cost 3')

        plt.show()
