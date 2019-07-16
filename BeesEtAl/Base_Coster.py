import numpy as np

class Base_Coster(object):
    """
    Subclasses must implement:
        map_to_solution_space(self, X)
        evaluate_cost(self) <- set self.cost with cost function at self.XA
        meso(self)          <- set self.XM with suggested new value
    """

    def __init__(self, base_optimiser):
        self.BO = base_optimiser

        self.verbose  = False # if True, print a statement at the start of each patch sequence
        self.nudgable = True  # if True, nudge the position if the cost function has already been evaluated

        self.XA   = None      # the true value of X at which the cost is evaluated
        self.XM   = None      # a suggested new value of X (which will be ignored if XA == XM)
        self.cost = None

    def calculate_cost(self, X):
        # Ask the subclass to set self.XA with a valid point in solution space
        self.XA = self.map_to_solution_space(X)

        # Before evaluating anything, let's see if we've already evaluated it before:
        index, cost = self.BO.lookup(self.XA)

        if self.nudgable and index is not None: # we've evaluated it before; let's try nudging it
            if self.verbose:
                print('> match found (nudging) <')

            self.XA = self.map_to_solution_space(self.BO.nudge(self.XA))

            # Before evaluating anything, let's see if we've already evaluated it before:
            index, cost = self.BO.lookup(self.XA)

        if index is not None: # we've evaluated it before
            if self.verbose:
                print('> match found (skipping) <')

        # XA is set; default to no MESO suggestion:
        self.XM   = np.copy(self.XA)
        self.cost = None

        if index is None: # okay, haven't calculated it before, so ask subclass to calculate it
            self.evaluate_cost()

            if np.isscalar(self.cost):
                self.cost = [self.cost] # cost must be an array

            # note the cost at the true position
            self.BO.push(self.cost, self.XA)

            # see if subclass will suggest a better position to try
            self.meso()

        return self.cost # evaluated cost, or None

    def nudge(self):
        if self.verbose:
            print('< nudging >')

        self.XM = self.map_to_solution_space(self.BO.nudge(self.XA))
