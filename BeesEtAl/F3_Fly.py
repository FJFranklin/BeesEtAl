import numpy as np

from .Base_Automaton import Base_Automaton

class F3_Fly(object):

    def __init__(self, garden, id_no, gender, orientation):
        self.G           = garden      # the F3_Garden object
        self.id_no       = id_no       # a reference number to identify this fly
        self.gender      = gender      # 'M', 'N' or 'F'
        self.orientation = orientation # list of one or more genders

        self.automaton   = Base_Automaton(2 + self.G.bee_shells)

        self.X           = None   # current position
        self.best_X      = None   # best personal position

    def bees(self, count):
        if self.G.costfn.verbose:
            print('==== Fly {p} (gender={g}, orientation={o}): #bees={b}, radius={r}'.format(p=self.id_no, g=self.gender, o=self.orientation, b=count, r=self.G.bee_radius))

        for b in range(0, count):
            cell  = self.automaton.cell()
            if cell == 0:
                new_X = self.G.new_position_in_neighbourhood(self.best_X, self.G.bee_radius, 'gauss')
            elif cell < (self.automaton.count - 1):
                new_X = self.G.new_position_in_neighbourhood(self.best_X, self.G.bee_radius * cell, 'sphere')
            else:
                radius = self.G.bee_radius * (self.automaton.count - 1)
                radius = radius + self.G.rand_exp(radius)
                new_X  = self.G.new_position_in_neighbourhood(self.best_X, radius, 'sphere')

            if self.G.costfn.calculate_cost(new_X) is not None:
                if self.G.plotter is not None:
                    self.G.plotter.bee(self.G.costfn.XA)

                if self.G.compare(self.G.costfn.XA, self.best_X):
                    if self.G.costfn.verbose:
                        print('(updating personal best)')
                    if self.G.plotter is not None:
                        self.G.plotter.fly(self.gender, self.G.costfn.XA, self.X, None)

                    self.best_X = self.G.costfn.XA
                    self.X      = self.G.costfn.XA

                    self.automaton.reward(cell)
                else:
                    self.automaton.punish(cell)

    def new_local_search(self, flies, ranks, radius, jitter):
        if self.G.costfn.verbose:
            print('==== Fly {p} (gender={g}, orientation={o}): rank={k}, radius={r}'.format(p=self.id_no, g=self.gender, o=self.orientation, k=ranks[0], r=radius))

        if ranks[0] == 0: # self-fly is superior to any it is attracted to; let's be narcissistic
            new_X  = self.best_X
        else:
            new_X  = np.zeros(self.G.Ndim)
            weight = np.zeros(len(flies))

            for f in range(1, len(flies)):
                if ranks[f] < ranks[0]: # a better fly than self-fly
                    weight[f] = 1 / (1 + ranks[f])

            weight = weight / sum(weight) # weight must sum to 1; it's a probability set

            for f in range(1, len(flies)):
                if ranks[f] < ranks[0]: # a better fly than self-fly
                    new_X = new_X + weight[f] * self.G.attraction(flies[f].best_X - self.X, radius)

            new_X = new_X + self.X

        new_X = self.G.new_position_in_neighbourhood(new_X, jitter)

        if self.G.costfn.calculate_cost(new_X) is not None:
            if self.G.compare(self.G.costfn.XA, self.best_X):
                if self.G.costfn.verbose:
                    print('(updating personal best)')
                if self.G.plotter is not None:
                    self.G.plotter.fly(self.gender, self.G.costfn.XA, self.X, None)
                self.best_X = self.G.costfn.XA
                self.X      = self.G.costfn.XA
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

        self.X      = XA
        self.best_X = XA

        if self.G.plotter is not None:
            self.G.plotter.fly(self.gender, self.X, None, None)

        return self.best_X # return the local best solution, even if old
