import numpy as np

class F3_Fly(object):

    def __init__(self, garden, id_no, gender, orientation):
        self.G           = garden      # the F3_Garden object
        self.id_no       = id_no       # a reference number to identify this fly
        self.gender      = gender      # 'M', 'N' or 'F'
        self.orientation = orientation # list of one or more genders

        self.X           = None   # current position
        self.best_X      = None   # best personal position

    def transition(self): # not, of course, reflective of human reality
        if np.random.rand(1) < self.G.trans:
            if self.gender == 'M':
                self.gender = 'N'
            elif self.gender == 'N':
                self.gender = 'F'
            else:
                self.gender = 'M'

            if self.G.costfn.verbose:
                print('> transition: gender -> {g} <'.format(g=self.gender))

    def bees(self, count, radius):
        if self.G.costfn.verbose:
            print('==== Fly {p} (gender={g}, orientation={o}): #bees={b}, radius={r}'.format(p=self.id_no, g=self.gender, o=self.orientation, b=count, r=radius))

        for b in range(0, count):
            new_X = self.G.new_position_in_neighbourhood(self.best_X, radius)

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

    def new_local_search(self, flies, ranks, radius):
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
                    dX    = flies[f].best_X - self.X
                    attr  = np.exp2(-np.linalg.norm(self.G.translate_to_unit_cube(dX)) / radius)
                    new_X = new_X + weight[f] * attr * dX

            new_X = new_X + self.X

        new_X = self.G.new_position_in_neighbourhood(new_X, radius / 2)

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

        self.transition()

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