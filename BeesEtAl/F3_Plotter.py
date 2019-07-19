import numpy as np

from .Base_Plotter import Base_Plotter

class F3_Plotter(Base_Plotter):

    def __init__(self, garden, plotaxes=None):
        Base_Plotter.__init__(self, garden, plotaxes)

    def new(self):
        if self._plotaxes is not None:
            self._default_axes()
        
    def done(self):
        if self._plotaxes is not None:
            self.sync(0.1)
        
    def bee(self, X):
        if self._plotaxes is not None:
            self._ax.scatter(X[self._plotaxes[0]], X[self._plotaxes[1]], marker='o', c='yellow', edgecolors='black')

    def fly(self, gender, X, old_X, best_X):
        if self._plotaxes is not None:
            if gender == 'M':
                color='red'
            elif gender == 'F':
                color='blue'
            else:
                color='green'

            if best_X is not None:
                self._ax.scatter(best_X[self._plotaxes[0]], best_X[self._plotaxes[1]], marker='*', c=color)
                self._ax.plot([X[self._plotaxes[0]], best_X[self._plotaxes[0]]], [X[self._plotaxes[1]], best_X[self._plotaxes[1]]], ':', marker=None, c=color)

            if old_X is not None:
                self._ax.plot([X[self._plotaxes[0]], old_X[self._plotaxes[0]]], [X[self._plotaxes[1]], old_X[self._plotaxes[1]]], '-', marker=None, c=color)

            self._ax.scatter(X[self._plotaxes[0]], X[self._plotaxes[1]], marker='X', c=color, edgecolors='black')
