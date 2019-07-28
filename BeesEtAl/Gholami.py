# *** References ***

# Gholami & Mohammadi, A Novel Combination of Bees and Firefly Algorithm to Optimize Continuous Problems

# Türker Tuncer, LDW-SCSA: Logistic Dynamic Weight based Sine Cosine Search Algorithm for Numerical Functions Optimization 
# https://arxiv.org/ftp/arxiv/papers/1809/1809.03055.pdf

# Hartmut Pohlheim, Examples of Objective Functions
# http://www.geatbx.com/download/GEATbx_ObjFunExpl_v38.pdf

# Wikipedia, Test functions for optimization
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np

from .Base_Coster import Base_Coster

class F1(Base_Coster):
    """
    Function F1 from Gholami & Mohammadi FA-BA Hybrid paper
    De Jong / Sphere (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -5.12 * np.ones(Ndim), 5.12 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA, 2))

    def meso(self):
        None

class F2(Base_Coster):
    """
    Function F2 from Gholami & Mohammadi FA-BA Hybrid paper
    Schwefel 2.22 (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -10 * np.ones(Ndim), 10 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.abs(self.XA)) + np.prod(np.abs(self.XA))

    def meso(self):
        None

class F3(Base_Coster):
    """
    Function F3 from Gholami & Mohammadi FA-BA Hybrid paper
    Schwefel 1.2 - Rotated hyper-ellipsoid (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -65.536 * np.ones(Ndim), 65.536 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = 0
        for i in range(0, len(self.XA)):
            self.cost = self.cost + (sum(self.XA[0:(i+1)]))**2

    def meso(self):
        None

class F4(Base_Coster):
    """
    Function F4 from Gholami & Mohammadi FA-BA Hybrid paper
    Schwefel 2.21 (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -100 * np.ones(Ndim), 100 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = max(np.abs(self.XA))

    def meso(self):
        None

class F5(Base_Coster):
    """
    Function F5 from Gholami & Mohammadi FA-BA Hybrid paper
    Rosenbrock (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -2.048 * np.ones(Ndim), 2.048 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(100 * np.power(self.XA[1:len(self.XA)] - np.power(self.XA[0:(len(self.XA)-1)], 2), 2) + np.power(1 - self.XA[0:(len(self.XA)-1)], 2))

    def meso(self):
        None

class F6(Base_Coster):
    """
    Function F6 from Gholami & Mohammadi FA-BA Hybrid paper
    Step (ND) cost function; optimum @ (-0.5,...
    """

    @staticmethod
    def extents(Ndim):
        return -100 * np.ones(Ndim), 100 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA + 0.5, 2))

    def meso(self):
        None

class F7(Base_Coster):
    """
    Function F7 from Gholami & Mohammadi FA-BA Hybrid paper
    Noise (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -1.28 * np.ones(Ndim), 1.28 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA, 4) * np.asarray(range(1, 1 + len(self.XA)))) + np.random.rand(1)

    def meso(self):
        None

class F8(Base_Coster):
    """
    Function F8 from Gholami & Mohammadi FA-BA Hybrid paper
    Schwefel (ND) cost function
    """

    @staticmethod
    def extents(Ndim):
        return -500 * np.ones(Ndim), 500 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = -sum(self.XA * np.sin(np.sqrt(abs(self.XA))))

    def meso(self):
        None

class F9(Base_Coster):
    """
    Function F9 from Gholami & Mohammadi FA-BA Hybrid paper
    Rastrigin (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -5.12 * np.ones(Ndim), 5.12 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA, 2) - 10 * np.cos(2 * np.pi * self.XA) + 10)

    def meso(self):
        None

class F10(Base_Coster):
    """
    Function F10 from Gholami & Mohammadi FA-BA Hybrid paper
    Ackley (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -32.768 * np.ones(Ndim), 32.768 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    @staticmethod
    def rms(X):
        return np.sqrt(X.dot(X) / len(X))

    def evaluate_cost(self):
        self.cost = np.exp(1) + 20 * (1 - np.exp(-F10.rms(self.XA) / 5)) - np.exp(sum(np.cos(2 * np.pi * self.XA)) / len(self.XA))

    def meso(self):
        None

class F11(Base_Coster):
    """
    Function F11 from Gholami & Mohammadi FA-BA Hybrid paper
    Griewangk (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -600 * np.ones(Ndim), 600 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        self.cost = sum(np.power(self.XA, 2)) / 4000 - np.prod(np.cos(np.power(self.XA, 2) / np.power(range(1, 1+len(self.XA)), 0.5))) + 1

    def meso(self):
        None

class F12(Base_Coster):
    """
    Function F12 from Gholami & Mohammadi FA-BA Hybrid paper
    Generalised Penalised 1 (ND) cost function; optimum @ (0,...
    """

    @staticmethod
    def extents(Ndim):
        return -50 * np.ones(Ndim), 50 * np.ones(Ndim)

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    @staticmethod
    def u(xi, a, k, m):
        if xi > a:
            v = k * (xi - a)**m
        elif xi < -a:
            v = k * (-xi - a)**m
        else:
            v = 0
        return v
        
    def evaluate_cost(self):
        y = 1 + (self.XA + 1) / 4

        c = 0
        for i in range(0, len(self.XA)):
            c = c + F12.u(self.XA[i], 10, 100, 4)

        self.cost = sum(np.power(y[0:(len(self.XA)-1)] - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * y[1:len(self.XA)]), 2)))
        self.cost = (self.cost + 10 * np.sin(np.pi * y[0]) + (y[len(self.XA)-1] - 1)**2) * np.pi / len(self.XA) + c

    def meso(self):
        None

def Gholami_TestFunction_Extents(number, Ndim=30):
    minima = None
    maxima = None

    if number == 1:
        minima, maxima = F1.extents(Ndim)
    if number == 2:
        minima, maxima = F2.extents(Ndim)
    if number == 3:
        minima, maxima = F3.extents(Ndim)
    if number == 4:
        minima, maxima = F4.extents(Ndim)
    if number == 5:
        minima, maxima = F5.extents(Ndim)
    if number == 6:
        minima, maxima = F6.extents(Ndim)
    if number == 7:
        minima, maxima = F7.extents(Ndim)
    if number == 8:
        minima, maxima = F8.extents(Ndim)
    if number == 9:
        minima, maxima = F9.extents(Ndim)
    if number == 10:
        minima, maxima = F10.extents(Ndim)
    if number == 11:
        minima, maxima = F11.extents(Ndim)
    if number == 12:
        minima, maxima = F12.extents(Ndim)

    return minima, maxima

def Gholami_TestFunction_Coster(number, base_optimiser):
    coster = None

    if number == 1:
        coster = F1(base_optimiser)
    if number == 2:
        coster = F2(base_optimiser)
    if number == 3:
        coster = F3(base_optimiser)
    if number == 4:
        coster = F4(base_optimiser)
    if number == 5:
        coster = F5(base_optimiser)
    if number == 6:
        coster = F6(base_optimiser)
    if number == 7:
        coster = F7(base_optimiser)
    if number == 8:
        coster = F8(base_optimiser)
    if number == 9:
        coster = F9(base_optimiser)
    if number == 10:
        coster = F10(base_optimiser)
    if number == 11:
        coster = F11(base_optimiser)
    if number == 12:
        coster = F12(base_optimiser)

    return coster
