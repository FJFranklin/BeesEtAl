import time
import numpy as np

from .Base_Coster import Base_Coster

class Schwefel(Base_Coster):
    """
    Schwefel (ND) cost function with discretisation & default MESO suggestion
    """

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        # Let's restrict solution space to discrete values, and choose the nearest
        return np.around(X, decimals=1)

    def evaluate_cost(self):
        self.cost = -sum(self.XA * np.sin(np.sqrt(abs(self.XA))))
        time.sleep(0.1) # purely for effect

    def meso(self):
        # Default is to not change XM and therefore have no MESO solution
        # One option is to try the experimental self.nudge()
        self.nudge()
