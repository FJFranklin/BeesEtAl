import numpy as np

class Base_Sorter(object):
    def __init__(self):
        self.ibest   = None # index of best record
        self.record  = None
        self.Nrecord = 0
        self.Ndim    = 0
        self.Ncost   = 0
        self.bMESO   = False

    def best(self):
        if self.ibest is not None:
            cost = self.record[self.ibest,1:(1+self.Ncost)]
            X    = self.record[self.ibest,(1+self.Ncost):(1+self.Ncost+self.Ndim)]
        else:
            cost = None
            X    = None

        return cost, X

    def lookup(self, X):
        index, rank = self.__lookup(X)

        if rank is not None:
            cost = self.record[index,1:(1+self.Ncost)]
        else:
            cost = None

        return rank, cost

    def compare(self, X, Y): # returns True if X < Y
        lt = False

        xi, xr = self.__lookup(X)
        if xr is not None:
            yi, yr = self.__lookup(Y)
            if yr is not None:
                if xr < yr:
                    lt = True

        return lt

    def __lookup(self, X):
        rank  = None
        index = None
        imin  = 0
        imax  = self.Nrecord

        for ix in range(0, self.Ndim):
            ir = ix + self.Ncost + 1

            iL = np.searchsorted(self.record[imin:imax,ir], X[ix], side='left')
            iR = np.searchsorted(self.record[imin:imax,ir], X[ix], side='right')

            imin = iL
            imax = iR

            if imin == imax:
                index = imin
                break

        if index is None:
            index = imin
            rank  = self.record[index,0]

        return index, rank

    def pop(self):
        cost = None
        X    = None
        M    = None

        if self.ibest is not None:
            cost  = self.record[self.ibest,1:(1+self.Ncost)]
            X     = self.record[self.ibest,(1+self.Ncost):(1+self.Ncost+self.Ndim)]
            if self.bMESO:
                M = self.record[self.ibest,(1+self.Ncost+self.Ndim):]

        if self.Nrecord == 1:
            self.ibest   = None
            self.record  = None
            self.Nrecord = 0
        elif self.Nrecord > 1:
            self.record  = np.delete(self.record, self.ibest, axis=0)
            self.Nrecord = self.Nrecord - 1
            self.__rank()

        return cost, X, M

    def push(self, cost, X, M=None):
        if self.record is None:
            if M is None:
                self.record  = np.asarray([[0, *cost, *X],], dtype=np.float64)
            else:
                self.record  = np.asarray([[0, *cost, *X, *M],], dtype=np.float64)
                self.bMESO   = True
            self.Nrecord = 1
            self.Ndim    = len(X)
            self.Ncost   = len(cost)
            self.ibest   = 0
        else:
            index, rank = self.__lookup(X)

            if rank is None:
                if M is None:
                    self.record = np.insert(self.record, index, [[0, *cost, *X],], axis=0)
                else:
                    self.record = np.insert(self.record, index, [[0, *cost, *X, *M],], axis=0)
                self.Nrecord = self.Nrecord + 1
                self.__rank()

    def __rank(self):
        multi = np.zeros(self.Nrecord)

        for ic in range(0, self.Ncost):
            order = self.record[:,(1+ic)].argsort()
            for r in range(0, self.Nrecord):
                multi[order[r]] = multi[order[r]] + r

        order = multi.argsort()
        for r in range(0, self.Nrecord):
            self.record[order[r],0] = r

        self.ibest = order[0]
