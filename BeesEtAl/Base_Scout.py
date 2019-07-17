import numpy as np

from .Base_Sorter import Base_Sorter

class Base_Scout(Base_Sorter):
    def __init__(self, base_optimiser):
        Base_Sorter.__init__(self, base_optimiser.Ndim)

        self.BO = base_optimiser

        self.pending = 0

    def schedule(self, count):
        self.pending = self.pending + count

    def evaluate(self, count_max):
        count = self.pending
        if count_max is not None:
            if count > count_max:
                count = count_max

        for c in range(0, count):
            if self.BO.costfn.verbose:
                print('==== New global search ====')

            # get a new location from anywhere in solution space & evaluate the cost
            if self.BO.costfn.calculate_cost(self.BO.new_position()) is not None:
                self.pending = self.pending - 1
                self.push(self.BO.costfn.cost, self.BO.costfn.XA, self.BO.costfn.XM)
