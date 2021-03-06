import csv

import numpy as np

class Base_Sorter(object):
    def __init__(self, Ndim, bPareto=False):
        self.Ndim    = Ndim

        self.ibest   = None # index of best record
        self.record  = None
        self.Nrecord = 0
        self.Ncost   = 0
        self.bMESO   = False
        self.bPareto = bPareto

    def best(self):
        if self.ibest is not None:
            cost = self.record[self.ibest,1:(1+self.Ncost)]
            X    = self.record[self.ibest,(1+self.Ncost):(1+self.Ncost+self.Ndim)]
        else:
            cost = None
            X    = None

        return cost, X

    def lookup(self, X):
        if X.ndim == 1:
            index, rank = self.__lookup(X)

            if rank is not None:
                cost = self.record[index,1:(1+self.Ncost)]
            else:
                cost = None
        else:
            rank = []
            cost = []

            for ix in range(0, len(X)):
                index, rthis = self.__lookup(X[ix])

                if rthis is not None:
                    cthis = self.record[index,1:(1+self.Ncost)]
                else:
                    cthis = None

                rank.append(rthis)
                cost.append(cthis)

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

        if imax > imin:
            for ix in range(0, self.Ndim):
                ir = ix + self.Ncost + 1

                iL = np.searchsorted(self.record[imin:imax,ir], X[ix], side='left')
                iR = np.searchsorted(self.record[imin:imax,ir], X[ix], side='right')

                imax = imin + iR
                imin = imin + iL

                if imin == imax:
                    index = imin
                    break

            if index is None:
                index = imin
                rank  = int(self.record[index,0])

        return index, rank

    def pop(self, index=None):
        cost = None
        X    = None
        M    = None

        if self.bPareto: # return a random (or by index, if specified) Pareto solution without removing it
            if self.Nrecord > 0:
                if index is None:
                    i = np.random.randint(self.Nrecord)
                else:
                    i = index

                cost  = self.record[i,1:(1+self.Ncost)]
                X     = self.record[i,(1+self.Ncost):(1+self.Ncost+self.Ndim)]
                if self.bMESO:
                    M = self.record[i,(1+self.Ncost+self.Ndim):]
        else:
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

            #print('Pop: cost={c}, X={x}, M={m}'.format(c=cost, x=X, m=M))

        return cost, X, M

    def push(self, cost, X, M=None):
        if self.record is None:
            if M is None:
                self.record  = np.asarray([[0, *cost, *X],], dtype=np.float64)
            else:
                self.record  = np.asarray([[0, *cost, *X, *M],], dtype=np.float64)
                self.bMESO   = True
            self.Nrecord = 1
            self.Ncost   = len(cost)
            self.ibest   = 0
        elif self.bPareto:
            dominated = []

            bDominant  = True
            bDominated = False

            for ir in range(0, self.Nrecord):
                rcost = self.record[ir,1:(1+self.Ncost)]

                if self.dominates(rcost, cost):
                    bDominated = True
                    bDominant  = False
                    break
                if self.dominates(cost, rcost):
                    dominated.append(ir)
                else:
                    bDominant  = False

            if bDominant:
                if M is None:
                    self.record  = np.asarray([[0, *cost, *X],], dtype=np.float64)
                else:
                    self.record  = np.asarray([[0, *cost, *X, *M],], dtype=np.float64)
                self.Nrecord = 1
                self.ibest   = 0
            elif not bDominated:
                if len(dominated) > 0:
                    self.record  = np.delete(self.record, dominated, axis=0)
                    self.Nrecord = self.Nrecord - len(dominated)
                    self.__rank()

                index, rank = self.__lookup(X)

                if rank is None:
                    if M is None:
                        self.record = np.insert(self.record, index, [[0, *cost, *X],], axis=0)
                    else:
                        self.record = np.insert(self.record, index, [[0, *cost, *X, *M],], axis=0)
                    self.Nrecord = self.Nrecord + 1
                    self.__rank()

                #print('~~~~( Optimal: {r} removed; new total = {t} )~~~~'.format(r=len(dominated), t=self.Nrecord))
        else:
            index, rank = self.__lookup(X)

            if rank is None:
                if M is None:
                    self.record = np.insert(self.record, index, [[0, *cost, *X],], axis=0)
                else:
                    self.record = np.insert(self.record, index, [[0, *cost, *X, *M],], axis=0)
                self.Nrecord = self.Nrecord + 1
                self.__rank()
        #print('Push: X={x}, record={r}'.format(x=X, r=self.record))

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

    def get_by_index(self, r):
        rank = self.record[r,0]
        cost = self.record[r,1:(1+self.Ncost)]
        X    = self.record[r,(1+self.Ncost):(1+self.Ncost+self.Ndim)]
        return rank, cost, X

    def dominates(self, X_cost, Y_cost): # returns true if X dominates Y
        bDominates = True

        for ic in range(0, self.Ncost):
            if X_cost[ic] < Y_cost[ic]:
                continue

            bDominates = False
            break

        return bDominates

    def pareto(self, file_name=None):
        if self.Nrecord == 0:
            return None, None

        the_dominant = []
        the_front    = []

        for ip in range(0, self.Nrecord):
            pcost = self.record[ip,1:(1+self.Ncost)]

            bDominant  = True
            bDominated = False

            for ir in range(0, self.Nrecord):
                if ip == ir:
                    continue

                rcost = self.record[ir,1:(1+self.Ncost)]

                if self.dominates(rcost, pcost):
                    bDominated = True
                    bDominant  = False
                    break
                if self.dominates(pcost, rcost) == False:
                    bDominant  = False

            if bDominant:
                the_dominant.append(ip)
                if file_name is not None:
                    self.__save('Dominant', the_dominant, file_name)
            elif bDominated == False:
                the_front.append(ip)
                if file_name is not None:
                    self.__save('Optimal', the_front, file_name)

        return the_dominant, the_front

    def __save(self, title, indices, file_name):
        rank, cost, X = self.get_by_index(indices)

        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([title] + [''] * (len(cost[0]) + len(X[0]) - 1))
            for i in range(0, len(indices)):
                writer.writerow([*(cost[i,:]), *(X[i,:])])
