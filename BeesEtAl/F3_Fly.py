import numpy as np

from .Base_Automaton import Base_Automaton

class F3_Fly(object):

    def __init__(self, garden, id_no, gender, orientation):
        self.G           = garden      # the F3_Garden object
        self.id_no       = id_no       # a reference number to identify this fly
        self.gender      = gender      # 'M', 'N' or 'F'
        self.orientation = orientation # list of one or more genders

        self.automaton   = Base_Automaton(2 + self.G.bee_shells, self.G.bee_reward, self.G.bee_punish)

        self.X           = None   # current position
        self.best_X      = None   # best personal position
        self.best_XM     = None   # associated MESO position

    def X_from_MESO(self):
        indices = []

        if np.array_equal(self.best_X, self.best_XM):
            X = self.best_X
        else:
            X = np.copy(self.best_X)

            for ix in range(0, len(X)):
                if X[ix] != self.best_XM[ix]:
                    if np.random.rand(1) < 0.5:
                        X[ix] = self.best_XM[ix]
                        indices.append(ix)

        if self.G.costfn.verbose:
            print(' >8< Bee: MESO = {i}'.format(i=indices))

        return X

    def bees(self, count):
        if self.G.costfn.verbose:
            print('==== Fly {p} (gender={g}, orientation={o}): #bees={b}, radius={r}'.format(p=self.id_no, g=self.gender, o=self.orientation, b=count, r=self.G.bee_radius))

        for b in range(0, count):
            meso_X = self.X_from_MESO()

            cell  = self.automaton.cell()
            if cell == 0:
                new_X = self.G.new_position_in_neighbourhood(meso_X, self.G.bee_radius, 'gauss')
            elif cell < (self.automaton.count - 1):
                new_X = self.G.new_position_in_neighbourhood(meso_X, self.G.bee_radius * cell, 'sphere')
            else:
                radius = self.G.bee_radius * (self.automaton.count - 1)
                radius = radius + self.G.rand_exp(radius)
                new_X  = self.G.new_position_in_neighbourhood(meso_X, radius, 'sphere')

            if self.G.costfn.calculate_cost(new_X) is not None:
                if self.G.plotter is not None:
                    self.G.plotter.bee(self.G.costfn.XA)

                if self.G.compare(self.G.costfn.XA, self.best_X):
                    if self.G.costfn.verbose:
                        print('(updating personal best)')
                    if self.G.plotter is not None:
                        self.G.plotter.fly(self.gender, self.G.costfn.XA, self.X, None)

                    self.best_X  = self.G.costfn.XA
                    self.best_XM = self.G.costfn.XM
                    self.X       = self.G.costfn.XA

                    self.automaton.reward(cell)
                else:
                    self.automaton.punish(cell)

        if False: # this is very noisy
            self.automaton.summarise()

    def new_local_search(self, flies, ranks, radius, jitter):
        if self.G.costfn.verbose:
            print('==== Fly {p} (gender={g}, orientation={o}): rank={k}, radius={r}'.format(p=self.id_no, g=self.gender, o=self.orientation, k=ranks[0], r=radius))

        if ranks[0] == 0: # self-fly is superior to any it is attracted to; let's be narcissistic
            new_X  = self.best_X
        else:
            old_X  = self.G.baseline(self.X, radius)
            new_X  = np.zeros(self.G.Ndim)
            weight = np.zeros(len(flies))

            for f in range(1, len(flies)):
                if ranks[f] < ranks[0]: # a better fly than self-fly
                    weight[f] = 1 / (1 + ranks[f])

            weight = weight / sum(weight) # weight must sum to 1; it's a probability set

            for f in range(1, len(flies)):
                if ranks[f] < ranks[0]: # a better fly than self-fly
                    new_X = new_X + weight[f] * self.G.attraction(flies[f].best_X - old_X, radius)

            new_X = new_X + old_X

        new_X = self.G.new_position_in_neighbourhood(new_X, jitter)

        if self.G.costfn.calculate_cost(new_X) is not None:
            if self.G.compare(self.G.costfn.XA, self.best_X):
                if self.G.costfn.verbose:
                    print('(updating personal best)')
                if self.G.plotter is not None:
                    self.G.plotter.fly(self.gender, self.G.costfn.XA, self.X, None)
                self.best_X  = self.G.costfn.XA
                self.best_XM = self.G.costfn.XM
                self.X       = self.G.costfn.XA
            else:
                if self.G.plotter is not None:
                    self.G.plotter.fly(self.gender, self.G.costfn.XA, self.X, self.best_X)
                self.X      = self.G.costfn.XA
        else:
            if self.G.plotter is not None:
                self.G.plotter.fly(self.gender, self.X, None, self.best_X)

        self.gender = self.G.transition(self.gender)

        return self.best_X # return the local best solution, even if old

    def new_global_search(self):
        cost, XA, XM = self.G.scout.pop()

        while cost is None: # shouldn't happen, but could (if solution space is small), so just in case...
            print('* * * No scouts banked! * * *')
            self.G.scout.schedule(1)
            self.G.scout.evaluate(1)
            cost, XA, XM = self.G.scout.pop() # although, if we exhaust all of space, this will go infinite

        self.X       = XA
        self.best_X  = XA
        self.best_XM = XM

        if self.G.plotter is not None:
            self.G.plotter.fly(self.gender, self.X, None, None)

        return self.best_X # return the local best solution, even if old
