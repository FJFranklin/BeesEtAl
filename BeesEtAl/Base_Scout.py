import numpy as np

class Base_Scout(object):
    def __init__(self, base_optimiser, max_records=10):
        self.BO = base_optimiser

        self.Nrecmax = max_records
        self.Nrecord = 0
        self.record  = None
        self.pending = 0

    def pop(self):
        cost = None
        XA   = None
        XM   = None

        if self.Nrecord > 0:
            cost = self.record[0,0:self.BO.Ncost]
            XA   = self.record[0,  self.BO.Ncost:(self.BO.Ndim + self.BO.Ncost)]
            XM   = self.record[0, (self.BO.Ndim + self.BO.Ncost):]

            self.Nrecord = self.Nrecord - 1
            self.record  = np.delete(self.record, 0, 0)

        return cost, XA, XM

    def push(self, cost, XA, XM):
        if self.Nrecord == 0:
            self.Nrecord = 1
            self.record  = np.asarray([[*cost, *XA, *XM],])
        else:
            self.Nrecord = self.Nrecord + 1;
            self.record  = np.append(self.record, [[*cost, *XA, *XM],], axis=0)

        if self.Nrecord > 1: # need to multi-sort
            multi = np.zeros(self.Nrecord)
            for c in range(0, self.BO.Ncost):
                order = self.record[:,c].argsort()
                for r in range(0, self.Nrecord):
                    multi[order[r]] = multi[order[r]] + r
            self.record = self.record[multi.argsort(),]

        if self.Nrecord > self.Nrecmax: # maintain record of [ten] best scouts
            self.Nrecord = self.Nrecord - 1
            self.record  = np.delete(self.record, self.Nrecord, 0)

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
