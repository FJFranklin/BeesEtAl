import numpy as np

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from .Base_Plotter import Base_Plotter

class BA_Plotter(Base_Plotter):

    def __init__(self, garden, plotaxes=None):
        Base_Plotter.__init__(self, garden, plotaxes)

    def bee(self, X):
        if self._plotaxes is not None:
            self.__plot_bee(X, 'black', '+')

    def __plot_bee(self, X, color, marker):
        self._ax.scatter(X[self._plotaxes[0]], X[self._plotaxes[1]], marker=marker, color=color)
        self.sync()

    def patch(self, XO, r):
        if self._plotaxes is not None:
             self.__plot_patch(XO, r)

    def __plot_patch(self, X, radius):
        self._ax.scatter(X[self._plotaxes[0]], X[self._plotaxes[1]], marker='.', color='blue')

        x_axis = 2 * radius * (self.BO.maxima[self._plotaxes[0]] - self.BO.minima[self._plotaxes[0]])
        y_axis = 2 * radius * (self.BO.maxima[self._plotaxes[1]] - self.BO.minima[self._plotaxes[1]])
        patch = mpatches.Ellipse((X[self._plotaxes[0]], X[self._plotaxes[1]]), x_axis, y_axis, edgecolor='blue', facecolor=(1,1,1,0.5))
        self._ax.add_collection(PatchCollection([patch], match_original=True))

        self.sync()

    def __plot_sequence(self, seq_hist, theta, dtheta, color, fn):
        if isinstance(seq_hist[0], list):
            old_cost, old_coord, old_up = self.__plot_sequence(seq_hist[0], theta, dtheta, color, fn)
            old_x, old_y = old_coord

            cost  = old_cost
            coord = None

            if old_up:
                new_up = False
                t = theta - dtheta / 2
            else:
                new_up = True
                t = theta + dtheta / 2

            for c in range(1, len(seq_hist)):
                new_cost = fn(seq_hist[c])
                r = 100 + (10 - 100) * new_cost
                new_x = r * np.cos(t)
                new_y = r * np.sin(t)
                new_coord = (new_x, new_y)

                if new_cost < old_cost:
                    if new_cost < cost:
                        cost  = new_cost
                        coord = new_coord
                        up    = new_up
                    self._ax.plot([old_x, new_x], [old_y, new_y], '-', color=color, linewidth=0.5)
                else:
                    self._ax.plot([old_x, new_x], [old_y, new_y], ':', color=color, linewidth=0.5)

            if coord is None:
                cost  = old_cost
                coord = old_coord
                up    = old_up
        else:
            # this is the initial point - start up.
            cost = fn(seq_hist[0])
            r = 100 + (10 - 100) * cost
            t = theta + dtheta / 2
            x = r * np.cos(t)
            y = r * np.sin(t)
            coord = (x, y)
            up = True
            self._ax.scatter(x, y, marker='.', color='black')

        return cost, coord, up

    def history(self, start_end, color, fn, rings=None):
        Nsequence = len(self.BO.history)
        angle_start, angle_end = start_end
        if angle_start != angle_end and Nsequence > 0:
            angle_start = angle_start * np.pi / 180
            angle_end   = angle_end   * np.pi / 180
            dtheta      = abs(angle_end - angle_start) / Nsequence

            self._open_plot_window()

            self._ax.set_xlim([-102, 102])
            self._ax.set_ylim([-102, 102])

            patch = mpatches.Ellipse((0, 0), 200, 200, edgecolor=(0.75,0.75,0.75,1), fill=False)
            self._ax.add_collection(PatchCollection([patch], match_original=True))

            patch = mpatches.Ellipse((0, 0), 20, 20, edgecolor=(0.75,0.75,0.75,1), fill=False)
            self._ax.add_collection(PatchCollection([patch], match_original=True))

            if rings is not None:
                for r in rings:
                    d = 20 + int(180 * (1 - r))
                    patch = mpatches.Ellipse((0, 0), d, d, edgecolor=(0.75,0.75,0.75,1), fill=False, linewidth=0.5, linestyle='--')
                    self._ax.add_collection(PatchCollection([patch], match_original=True))

            for s in range(0, Nsequence+1):
                theta = angle_start + (angle_end - angle_start) * s / Nsequence
                x1 = 10  * np.cos(theta)
                y1 = 10  * np.sin(theta)
                x2 = 100 * np.cos(theta)
                y2 = 100 * np.sin(theta)
                self._ax.plot([x1, x2], [y1, y2], '-', color=(0.85,0.85,0.85,1), linewidth=0.25)

            for s in range(0, Nsequence):
                theta = angle_start + (angle_end - angle_start) * (s + 0.5) / Nsequence
                seq_no, seq_term, seq_hist = self.BO.history[s]
                cost, coord, up = self.__plot_sequence(seq_hist, theta, dtheta, color, fn)

                if seq_term == 'abandoned':
                    # mark the best solution
                    x, y = coord
                    self._ax.scatter(x, y, marker='o', color='black')

            self.sync(10)
