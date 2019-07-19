import numpy as np

from .Base_Coster import Base_Coster

class Viennet(Base_Coster):
    """
    Viennet multi-objective optimisation test function
    """

    @staticmethod
    def extents():
        return -3 * np.ones(2), 3 * np.ones(2)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        x = self.XA[0]
        y = self.XA[1]

        r2 = x**2 + y**2
        f1 = 0.5 * r2 + np.sin(r2)
        f2 = (3 * x - 2 * y + 4)**2 / 8 + (x - y + 1)**2 / 27 + 15
        f3 = 1 / (r2 + 1) - 1.1 * np.exp(-r2)

        self.cost = [f1,f2,f3]

    def meso(self):
        # Default is to not change XM and therefore have no MESO solution
        # One option is to try the experimental self.nudge()
        None
