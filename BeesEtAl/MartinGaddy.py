import time
import numpy as np

from .Base_Coster import Base_Coster

class MartinGaddy(Base_Coster):
    """
    MartinGaddy Martin & Gaddy (2D) cost function with discretisation & MESO suggestion
    """

    @staticmethod
    def extents():
        return np.zeros(2), 10 * np.ones(2)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        # Let's restrict solution space to discrete values, and choose the nearest
        return np.around(X, decimals=1)

    def evaluate_cost(self):
        self.cost = (self.XA[0]-self.XA[1])**2 + (self.XA[0]+self.XA[1]-10)**2/9
        time.sleep(0.1) # purely for effect

    def meso(self):
        # Default is to not change XM and therefore have no MESO solution
        # One option is to try the experimental self.nudge()

        # Let's suggest a better position to try:
        self.XM = np.around(self.XA-np.sign(self.XA)*0.05, decimals=1)
