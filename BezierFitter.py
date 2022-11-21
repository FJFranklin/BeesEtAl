import argparse

import numpy as np

from BeesEtAl.Base_Coster import Base_Coster

# A noisy function that BA handles poorly

class BezierFitter(Base_Coster):
    """
    Fits a Bezier spline to a quarter sine curve
    """

    @staticmethod
    def extents():
        return np.asarray([0,0,0.45]), np.asarray([1,np.pi/2,0.55])

    def __init__(self, base_optimiser):
        Base_Coster.__init__(self, base_optimiser)

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        alpha = self.XA[0]
        beta  = self.XA[1]
        t     = self.XA[2]

        cp0 = np.asarray([0,0])
        cp1 = np.asarray([alpha,alpha])
        cp2 = np.asarray([beta,1])
        cp3 = np.asarray([np.pi/2,1])

        xy = (1-t)**3*cp0 + 3*(1-t)**2*t*cp1 + 3*(1-t)*t**2*cp2 + t**3*cp3
        dxy = -3*(1-t)**2*cp0 + 3*((1-t)**2 - 2*(1-t)*t)*cp1 + 3*(2*(1-t)*t - t**2)*cp2 + 3*t**2*cp3

        f1 = (xy[1] - np.sin(xy[0]))**2
        f2 = (dxy[1] - dxy[0]*np.cos(xy[0]))**2

        self.cost = [f1,f2]

    def meso(self):
        None

parser = argparse.ArgumentParser(description="Test/Demo script for plotting/comparing BA/F3 while fitting a Bezier spline to a sine curve.")

parser.add_argument('-o', '--optimiser', help='Select optimiser [BA].',                            default='BA', choices=['BA', 'F3'])
parser.add_argument('--iterations',      help='How many iterations to do [100].',                  default=100,  type=int)
parser.add_argument('--no-plot',         help='Do not plot.',                                      action='store_true')
parser.add_argument('--suppress',        help='In case of F3, suppress diversity.',                action='store_true')
parser.add_argument('--attraction',      help='In case of F3, use exponential or Gaussian [exp].', default='exp', choices=['exp', 'gauss'])
parser.add_argument('--out',             help='Specify output file name [pareto.csv].',            default='pareto.csv', type=str)

args = parser.parse_args()

if args.no_plot:
    bPlot = False
else:
    bPlot = True

P = None

bSuppress = False

minima, maxima = BezierFitter.extents()

if args.optimiser == 'BA':
    from BeesEtAl.BA_Garden import BA_Garden

    G = BA_Garden(minima, maxima, [5,2,2,1])

    if bPlot:
        from BeesEtAl.BA_Plotter import BA_Plotter
        P = BA_Plotter(G, [0, 2])

    method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
    Nfail  = 6       # i.e., stops at 6th failure
    rf     = 0.01    # smallest patch radius
    r0, sf = G.initial_radius_and_shrinking(Nfail, rf, method) # or set your own initial radius & shrinking factor
    params = { 'radius': r0, 'shrink': sf, 'fail_at': Nfail, 'neighborhood': method, 'dynamic': True }

elif args.optimiser == 'F3':
    from BeesEtAl.F3_Garden import F3_Garden

    G = F3_Garden(minima, maxima, [1,6,3])

    if bPlot:
        from BeesEtAl.F3_Plotter import F3_Plotter
        P = F3_Plotter(G, [0, 1])

    method = 'gauss' # default is 'ball'; other options are 'cube' and 'sphere' (on rather than in)
    params = { 'neighborhood': method, 'attraction': args.attraction }

    if args.suppress:
        bSuppress = True

G.costfn = BezierFitter(G)
G.set_search_params(**params)

for it in range(1, 1 + args.iterations):
    if bSuppress:
        solver_runs = G.iterate(unisex=True)
    else:
        solver_runs = G.iterate()

    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

G.pareto(args.out)
P.pareto([0,1])
P.sync(10)
