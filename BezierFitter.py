import argparse

import numpy as np

from BeesEtAl.Base_Coster import Base_Coster

def f_sin(x):
    return np.sin(x)

def df_sin(x):
    return np.cos(x)

def f_gauss(x):
    return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

def df_gauss(x):
    return -x * np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

class BezierFitter(Base_Coster):
    """
    Fits a Bezier spline to a specified function across a set of x-values
    """

    m_x   = None # specified x-values
    m_y   = None # corresponding y-values
    m_g   = None # corresponding gradients
    m_f   = None # function to be fit
    m_df  = None # derivative of function to be fit

    def extents(x):
        dx = x[1:] - x[:-1]
        dx_min = np.asarray([dx[0],*np.minimum(dx[1:],dx[:-1]),dx[-1]])
        return np.zeros(len(dx_min)), dx_min

    def __init__(self, base_optimiser, func, deriv, x):
        Base_Coster.__init__(self, base_optimiser)
        self.m_x   = x
        self.m_y   = func(x)
        self.m_g   = deriv(x)
        self.m_f   = func
        self.m_df  = deriv

    def map_to_solution_space(self, X):
        return X

    def evaluate_cost(self):
        cp0 = np.asarray([self.m_x[:-1],self.m_y[:-1]])
        cp1 = np.asarray([self.m_x[:-1] + self.XA[:-1],self.m_y[:-1] + self.XA[:-1] * self.m_g[:-1]])
        cp2 = np.asarray([self.m_x[1:]  - self.XA[1:], self.m_y[1:]  - self.XA[1:]  * self.m_g[1:] ])
        cp3 = np.asarray([self.m_x[1:], self.m_y[1:]])

        t = 0.5
        xy  = (1-t)**3*cp0 + 3*(1-t)**2*t*cp1 + 3*(1-t)*t**2*cp2 + t**3*cp3
        dxy = -3*(1-t)**2*cp0 + 3*((1-t)**2 - 2*(1-t)*t)*cp1 + 3*(2*(1-t)*t - t**2)*cp2 + 3*t**2*cp3

        cost_xy  = ( xy[1,:] - self.m_f(xy[0,:]))**2
        cost_dxy = (dxy[1,:] - dxy[0,:] * self.m_df(xy[0,:]))**2

        self.cost = [np.sum(cost_xy),np.sum(cost_dxy)]

    def meso(self):
        None

parser = argparse.ArgumentParser(description="Test/Demo script for plotting/comparing BA/F3 while fitting a Bezier spline to a sine curve or Gaussian distribution.")

parser.add_argument('-o', '--optimiser', help='Select optimiser [BA].',                            default='BA', choices=['BA', 'F3'])
parser.add_argument('--function',        help='Select function to fit [sin].',                     default='sin', choices=['sin', 'gauss'])
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

if args.function == 'gauss':
    X = np.asarray([0,1,2,3])
else:
    X = np.asarray([0,np.pi/2])
minima, maxima = BezierFitter.extents(X)

if args.optimiser == 'BA':
    from BeesEtAl.BA_Garden import BA_Garden

    G = BA_Garden(minima, maxima, [5,2,2,1])

    if bPlot:
        from BeesEtAl.BA_Plotter import BA_Plotter
        P = BA_Plotter(G, [0, 1])

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

if args.function == 'gauss':
    G.costfn = BezierFitter(G, f_gauss, df_gauss, X)
else:
    G.costfn = BezierFitter(G, f_sin, df_sin, X)
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
