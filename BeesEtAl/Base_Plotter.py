from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class Base_Plotter(object):

    def __init__(self, base_optimiser, plotaxes):
        self.BO = base_optimiser

        self.BO.plotter = self

        self._plotaxes = plotaxes # [i, j] - indices to use for plotting; defaults to [0, 1] for 2D

        self._fig = None
        self._ax  = None

        if self.BO.Ndim == 2 and self._plotaxes is None:
            self._plotaxes = [0, 1]
        if self._plotaxes is not None:
            self._open_plot_window()

            self._ax.set_xlim([self.BO.minima[self._plotaxes[0]], self.BO.maxima[self._plotaxes[0]]])
            self._ax.set_ylim([self.BO.minima[self._plotaxes[1]], self.BO.maxima[self._plotaxes[1]]])

            self.sync()

    def _open_plot_window(self):
        if self._ax is None:
            xsize = 1500
            ysize = 1500
            dpi_osx = 192 # Something very illogical here.
            self._fig = plt.figure(figsize=(xsize / dpi_osx, ysize / dpi_osx), dpi=(dpi_osx/2))

            self._ax = self._fig.add_subplot(111)
            self._ax.set_facecolor('white')
            self._ax.set_position([0.07, 0.06, 0.90, 0.90])

            plt.ion()
            plt.show()
        else:
            plt.cla()

    def save(self, file_name):
        if self._ax is not None:
            plt.savefig(file_name)

    def sync(self, dt=0.000001):
        plt.draw()
        plt.pause(dt)

    def pareto(self, cost_indices):
        the_dominant, the_front = self.BO.pareto()

        if the_front is not None and cost_indices is not None and self.BO.Ncost > 1:
            if len(the_front) > 0:
                if len(cost_indices) == 2:
                    X = self.BO.record[the_front,(1+cost_indices[0])]
                    Y = self.BO.record[the_front,(1+cost_indices[1])]

                    self._open_plot_window()
                    self._ax.scatter(X, Y)

                    self.sync()

                if len(cost_indices) == 3:
                    X = self.BO.record[the_front,(1+cost_indices[0])]
                    Y = self.BO.record[the_front,(1+cost_indices[1])]
                    Z = self.BO.record[the_front,(1+cost_indices[2])]

                    self._open_plot_window()

                    self._ax = self._fig.add_subplot(111, projection='3d')
                    self._ax.scatter(X, Y, Z)

                    self.sync()

        return the_dominant, the_front
