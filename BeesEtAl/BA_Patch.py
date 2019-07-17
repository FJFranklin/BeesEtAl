import numpy as np

class BA_Patch(object):

    def __init__(self, garden, id_no):
        self.G        = garden # the BA_Garden object
        self.id_no    = id_no  # a reference number to identify this patch
        self.Nfails   = 0      # total number of local failures to find better position
        self.radius   = 0      # current neighbourhood radius
        self.try_X    = None   # a suggestion for where to try next; may be empty
        self.old_X    = None   # the current local best position
        self.old_cost = 0      # cost at old_X
        self.sequence = 0      # number identifying each search sequence
        self.history  = None

    def X_from_MESO(self):
        X = np.copy(self.old_X)

        indices = []
        for i in range(0, len(self.try_X)):
            if self.old_X[i] != self.try_X[i]:
                indices.append(i)
        Ni = len(indices)

        Nc = np.random.binomial(Ni, (1 + self.G.Nfails - self.Nfails) / (2 + self.G.Nfails))
        # equivalent to rolling the dice for each index - which, in retrospect, would be simpler to implement

        if Nc == 0:
            if self.G.costfn.verbose:
                print('MESO: difference in indices: {i} -> (none)'.format(i=indices))
        else:
            changes = np.random.permutation(indices)

            if self.G.costfn.verbose:
                print('MESO: difference in indices: {i} -> {c}'.format(i=indices, c=changes[0:Nc]))

            for i in range(0, Nc):
                X[changes[i]] = self.try_X[changes[i]]

        return X

    def new_local_search(self, prior):
        self.history = [self.history]
        
        if self.G.costfn.verbose:
            print('==== Patch {p}: #bees={b}, #fails={f}, cost={c}, radius={r}'.format(p=self.id_no, b=prior, f=self.Nfails, c=self.old_cost, r=self.radius))

        bPatch  = True # whether we should try to plot the patch
        bFirst  = True
        bFailed = True

        Neval = 0 # number of evaluations of the cost function

        for p in range(0, prior): # prior is the number of bees attracted to this patch
            if bPatch:
                if self.G.plotter:
                    self.G.plotter.patch(self.old_X, self.radius)
                bPatch = False

            if self.try_X is not None:
                if bFirst:
                    X = self.X_from_MESO()
                else:
                    X = self.G.new_position_in_neighbourhood(self.X_from_MESO(), self.radius)
            else:
                X = self.G.new_position_in_neighbourhood(self.old_X, self.radius)

            if self.G.costfn.calculate_cost(X) is not None:
                cost = self.G.costfn.cost
                XA   = self.G.costfn.XA
                XM   = self.G.costfn.XM

                if self.G.plotter:
                    self.G.plotter.bee(XA)
            else:
                if self.G.costfn.verbose:
                    print('(skip - bank 1 scout)')
                self.G.scout.schedule(1)
                continue

            Neval = Neval + 1

            if self.G.compare(XA, self.old_X):
                bFailed = False

            if self.G.dynamic:
                if self.G.compare(XA, self.old_X):
                    if self.G.costfn.verbose:
                        print('(updating patch)')

                    self.old_X    = XA
                    self.old_cost = cost

                    if np.array_equal(XA, XM):
                        self.try_X = None
                    else:
                        self.try_X = XM

                    bPatch = True
                    bFirst = True
                else:
                    bFirst = False
            else:
                if bFirst:
                    best_X    = X
                    best_XA   = XA
                    best_XM   = XM
                    best_cost = cost
                    bFirst    = False
                elif self.G.compare(XA, self.best_X):
                    best_X    = X
                    best_XA   = XA
                    best_XM   = XM
                    best_cost = cost

            self.history.append(cost)

        # ... and we're done

        if bFailed: # shrink the neighbourhood
            self.radius = self.radius * self.G.cooling
            self.Nfails = self.Nfails + 1
        elif not self.G.dynamic:
            self.old_X    = best_XA
            self.old_cost = best_cost

            if np.array_equal(best_XA, best_XM):
                self.try_X = None
            else:
                self.try_X = best_XM

        return self.old_X # return the local best solution, even if old

    def new_global_search(self, seq_id, seq_term): # function cost = new_global_search(sequence_number > 0, termination cause)
        if self.history is not None:
            self.G.report(self.sequence, seq_term, self.history)

        self.sequence = seq_id
        
        cost, XA, XM = self.G.scout.pop()

        while cost is None: # shouldn't happen, but could (if solution space is small), so just in case...
            print('* * * No scouts banked! * * *')
            self.G.scout.schedule(1)
            self.G.scout.evaluate(1)
            cost, XA, XM = self.G.scout.pop() # although, if we exhaust all of space, this will go infinite

        self.old_X    = XA
        self.old_cost = cost

        if np.array_equal(XA, XM):
            self.try_X = None
        else:
            self.try_X = XM

        self.radius = self.G.radius
        self.Nfails = 0

        self.history = [self.old_cost]

        return self.old_X

    def flush_history(self):
        if self.history is not None:
            self.G.report(self.sequence, 'incomplete', self.history)
