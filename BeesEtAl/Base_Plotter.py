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
