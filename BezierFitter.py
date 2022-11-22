from datetime import datetime
import argparse

import numpy as np

from BeesEtAl.Base_Coster import Base_Coster

parser = argparse.ArgumentParser(description="Test/Demo script for plotting/comparing BA/F3 while fitting a Bezier spline to a sine curve or Gaussian distribution.")

parser.add_argument('-o', '--optimiser', help='Select optimiser [BA].',                            default='F3', choices=['BA', 'F3'])
parser.add_argument('--function',        help='Select function to fit [sin].',                     default='sin', choices=['sin','tan','gauss','exp'])
parser.add_argument('--iterations',      help='How many iterations to do [100].',                  default=100,  type=int)
parser.add_argument('--no-plot',         help='Do not plot.',                                      action='store_true')
parser.add_argument('--multiobjective',  help='Multiobjective optimisation.',                      action='store_true')
parser.add_argument('--suppress',        help='In case of F3, suppress diversity.',                action='store_true')
parser.add_argument('--attraction',      help='In case of F3, use exponential or Gaussian [exp].', default='exp', choices=['exp', 'gauss'])
parser.add_argument('--out',             help='Specify output file name [pareto.csv].',            default='pareto.csv', type=str)
parser.add_argument('--eps-plot',        help='Create an EPS plot.',                               action='store_true')
parser.add_argument('--eps-scale',       help='How much to scale the EPS [100].',                  default=100,  type=int)
parser.add_argument('--eps-out',         help='Specify output EPS file name [bezier.eps].',        default='bezier.eps', type=str)

args = parser.parse_args()

def f_exp(x):
    return np.exp(x)

def df_exp(x):
    return np.exp(x)

def f_gauss(x):
    return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

def df_gauss(x):
    return -x * np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

def f_tan(x):
    return np.tan(x)

def df_tan(x):
    return 1 + np.tan(x)**2

def f_sin(x):
    return np.sin(x)

def df_sin(x):
    return np.cos(x)

if args.function == 'exp':
    bf_func  = f_exp
    bf_deriv = df_exp
    bf_X     = np.asarray([-3,-2,-1,0,1,2,3])
    bf_type  = 'plain'
elif args.function == 'gauss':
    bf_func  = f_gauss
    bf_deriv = df_gauss
    bf_X     = np.asarray([0,1,2,3])
    bf_type  = 'symmetric'
elif args.function == 'tan':
    bf_func  = f_tan
    bf_deriv = df_tan
    bf_X     = np.asarray([0,1,1.4])
    bf_type  = 'qtr-tan'
else:
    bf_func  = f_sin
    bf_deriv = df_sin
    bf_X     = np.asarray([0,np.pi/2])
    bf_type  = 'qtr-sin'

class BezierFitter(Base_Coster):
    """
    Fits a Bezier spline to a specified function across a set of x-values
    """

    m_x   = None # specified x-values
    m_y   = None # corresponding y-values
    m_g   = None # corresponding gradients
    m_f   = None # function to be fit
    m_df  = None # derivative of function to be fit
    m_cp0 = None
    m_cp1 = None
    m_cp2 = None
    m_cp3 = None

    @staticmethod
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

    def evaluate_control_points(self, X):
        self.m_cp0 = np.asarray([self.m_x[:-1],self.m_y[:-1]])
        self.m_cp1 = np.asarray([self.m_x[:-1] + X[:-1],self.m_y[:-1] + X[:-1] * self.m_g[:-1]])
        self.m_cp2 = np.asarray([self.m_x[1:]  - X[1:], self.m_y[1:]  - X[1:]  * self.m_g[1:] ])
        self.m_cp3 = np.asarray([self.m_x[1:], self.m_y[1:]])

    def evaluate_cost(self):
        self.evaluate_control_points(self.XA)
        
        t = 0.5
        xy  = (1-t)**3*self.m_cp0 + 3*(1-t)**2*t*self.m_cp1 + 3*(1-t)*t**2*self.m_cp2 + t**3*self.m_cp3
        dxy = -3*(1-t)**2*self.m_cp0 + 3*((1-t)**2 - 2*(1-t)*t)*self.m_cp1 + 3*(2*(1-t)*t - t**2)*self.m_cp2 + 3*t**2*self.m_cp3

        cost_xy  = ( xy[1,:] - self.m_f(xy[0,:]))**2
        cost_dxy = (dxy[1,:] - dxy[0,:] * self.m_df(xy[0,:]))**2

        if args.multiobjective:
            self.cost = [np.sum(cost_xy),np.sum(cost_dxy)]
        else:
            self.cost = np.sum(cost_xy) + np.sum(cost_dxy)

    def meso(self):
        None

minima, maxima = BezierFitter.extents(bf_X)

if args.no_plot:
    bPlot = False
else:
    bPlot = True

P = None

bSuppress = False

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

fitter = BezierFitter(G, bf_func, bf_deriv, bf_X)
G.costfn = fitter
G.set_search_params(**params)

for it in range(1, 1 + args.iterations):
    if bSuppress:
        solver_runs = G.iterate(unisex=True)
    else:
        solver_runs = G.iterate()

    best_cost, best_X = G.best()
    print('Iteration {:4d}: Global best = {c} @ {x}'.format(it, c=best_cost, x=best_X))

if args.multiobjective:
    G.pareto(args.out)
    P.pareto([0,1])
    P.sync(10)
if args.eps_plot:
    # Get best solution and determine corresponding control points
    best_cost, best_X = G.best()
    fitter.evaluate_control_points(best_X)
    # TODO construct curves
    curves = []
    if bf_type == 'symmetric':
        curve = [(fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        count = fitter.m_cp0.shape[1]
        for ic in range(0, count):
            curve.append((fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (-fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.insert(0, (-fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.insert(0, (-fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
        curves.append(curve)
    elif bf_type == 'qtr-sin':
        count = fitter.m_cp0.shape[1]
        x_qtr = fitter.m_cp3[0,-1]
        curve = [(fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        for ic in range(0, count):
            curve.append((2*x_qtr-fitter.m_cp2[0,count-ic-1],fitter.m_cp2[1,count-ic-1]))
            curve.append((2*x_qtr-fitter.m_cp1[0,count-ic-1],fitter.m_cp1[1,count-ic-1]))
            curve.append((2*x_qtr-fitter.m_cp0[0,count-ic-1],fitter.m_cp0[1,count-ic-1]))
            curve.insert(0, (-(2*x_qtr-fitter.m_cp2[0,count-ic-1]),-fitter.m_cp2[1,count-ic-1]))
            curve.insert(0, (-(2*x_qtr-fitter.m_cp1[0,count-ic-1]),-fitter.m_cp1[1,count-ic-1]))
            curve.insert(0, (-(2*x_qtr-fitter.m_cp0[0,count-ic-1]),-fitter.m_cp0[1,count-ic-1]))
        for ic in range(0, count):
            curve.append((2*x_qtr+fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.append((2*x_qtr+fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.append((2*x_qtr+fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
            curve.insert(0, (-(2*x_qtr+fitter.m_cp1[0,ic]),fitter.m_cp1[1,ic]))
            curve.insert(0, (-(2*x_qtr+fitter.m_cp2[0,ic]),fitter.m_cp2[1,ic]))
            curve.insert(0, (-(2*x_qtr+fitter.m_cp3[0,ic]),fitter.m_cp3[1,ic]))
        for ic in range(0, count):
            curve.append((4*x_qtr-fitter.m_cp2[0,count-ic-1],-fitter.m_cp2[1,count-ic-1]))
            curve.append((4*x_qtr-fitter.m_cp1[0,count-ic-1],-fitter.m_cp1[1,count-ic-1]))
            curve.append((4*x_qtr-fitter.m_cp0[0,count-ic-1],-fitter.m_cp0[1,count-ic-1]))
            curve.insert(0, (-(4*x_qtr-fitter.m_cp2[0,count-ic-1]),fitter.m_cp2[1,count-ic-1]))
            curve.insert(0, (-(4*x_qtr-fitter.m_cp1[0,count-ic-1]),fitter.m_cp1[1,count-ic-1]))
            curve.insert(0, (-(4*x_qtr-fitter.m_cp0[0,count-ic-1]),fitter.m_cp0[1,count-ic-1]))
        curves.append(curve)
        # add a secondary 'cos' curve that's symmetric with the same period
        curve = [(-x_qtr+fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((-x_qtr+fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((-x_qtr+fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((-x_qtr+fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (-x_qtr-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (-x_qtr-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (-x_qtr-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        for ic in range(0, count):
            curve.append((x_qtr-fitter.m_cp2[0,count-ic-1],fitter.m_cp2[1,count-ic-1]))
            curve.append((x_qtr-fitter.m_cp1[0,count-ic-1],fitter.m_cp1[1,count-ic-1]))
            curve.append((x_qtr-fitter.m_cp0[0,count-ic-1],fitter.m_cp0[1,count-ic-1]))
            curve.insert(0, (-(3*x_qtr-fitter.m_cp2[0,count-ic-1]),-fitter.m_cp2[1,count-ic-1]))
            curve.insert(0, (-(3*x_qtr-fitter.m_cp1[0,count-ic-1]),-fitter.m_cp1[1,count-ic-1]))
            curve.insert(0, (-(3*x_qtr-fitter.m_cp0[0,count-ic-1]),-fitter.m_cp0[1,count-ic-1]))
        for ic in range(0, count):
            curve.append((x_qtr+fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.append((x_qtr+fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.append((x_qtr+fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
            curve.insert(0, (-(3*x_qtr+fitter.m_cp1[0,ic]),fitter.m_cp1[1,ic]))
            curve.insert(0, (-(3*x_qtr+fitter.m_cp2[0,ic]),fitter.m_cp2[1,ic]))
            curve.insert(0, (-(3*x_qtr+fitter.m_cp3[0,ic]),fitter.m_cp3[1,ic]))
        for ic in range(0, count):
            curve.append((3*x_qtr-fitter.m_cp2[0,count-ic-1],-fitter.m_cp2[1,count-ic-1]))
            curve.append((3*x_qtr-fitter.m_cp1[0,count-ic-1],-fitter.m_cp1[1,count-ic-1]))
            curve.append((3*x_qtr-fitter.m_cp0[0,count-ic-1],-fitter.m_cp0[1,count-ic-1]))
        for ic in range(0, count):
            curve.append((3*x_qtr+fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((3*x_qtr+fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((3*x_qtr+fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
        curves.append(curve)
    elif bf_type == 'qtr-tan':
        count = fitter.m_cp0.shape[1]
        x_qtr = np.pi/2
        curve = [(fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        curves.append(curve)
        curve = [(2*x_qtr+fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((2*x_qtr+fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((2*x_qtr+fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((2*x_qtr+fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (2*x_qtr-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (2*x_qtr-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (2*x_qtr-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        curves.append(curve)
        curve = [(-2*x_qtr+fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((-2*x_qtr+fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((-2*x_qtr+fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((-2*x_qtr+fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
            curve.insert(0, (-2*x_qtr-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (-2*x_qtr-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (-2*x_qtr-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        curves.append(curve)
        curve = [(4*x_qtr+fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.insert(0, (4*x_qtr-fitter.m_cp1[0,ic],-fitter.m_cp1[1,ic]))
            curve.insert(0, (4*x_qtr-fitter.m_cp2[0,ic],-fitter.m_cp2[1,ic]))
            curve.insert(0, (4*x_qtr-fitter.m_cp3[0,ic],-fitter.m_cp3[1,ic]))
        curves.append(curve)
        curve = [(-4*x_qtr+fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        for ic in range(0, count):
            curve.append((-4*x_qtr+fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((-4*x_qtr+fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((-4*x_qtr+fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
        curves.append(curve)
    else: # just add the curve as-is
        curve = [(fitter.m_cp0[0,0],fitter.m_cp0[1,0])]
        count = fitter.m_cp0.shape[1]
        for ic in range(0, count):
            curve.append((fitter.m_cp1[0,ic],fitter.m_cp1[1,ic]))
            curve.append((fitter.m_cp2[0,ic],fitter.m_cp2[1,ic]))
            curve.append((fitter.m_cp3[0,ic],fitter.m_cp3[1,ic]))
        curves.append(curve)
    # Determine plot bounds with scaling and add margins
    scale = args.eps_scale
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for c in curves:
        for cp in c:
            x, y = cp
            if x_min > x:
                x_min = x
            if x_max < x:
                x_max = x
            if y_min > y:
                y_min = y
            if y_max < y:
                y_max = y
    x0 = 10 - scale * x_min
    y0 = 10 - scale * y_min
    width  = 20 + scale * (x_max - x_min)
    height = 20 + scale * (y_max - y_min)
    # Write the EPS file
    filename = args.eps_out
    with open(filename, 'w') as eps:
        eps.write("%!PS-Adobe-3.1 EPSF-3.0\n")
        eps.write("%%BoundingBox: 0 0 {w} {h}\n".format(w=int(np.ceil(width)), h=int(np.ceil(height))))
        eps.write("%%Title: " + filename + "\n")
        eps.write("%%Creator: BeesEtAl BezierFitter.py\n")
        eps.write("%%CreationDate: " + datetime.now().strftime("%d-%b-%Y") + "\n")
        eps.write("%%Pages: 1\n")
        eps.write("%%LanguageLevel: 1\n")
        eps.write("%%EndComments\n")
        eps.write("%%BeginProlog\n")
        eps.write("%%EndProlog\n")

        eps.write("0 setgray\n")
        eps.write("2 setlinewidth\n")
        eps.write("newpath {x1:.1f} {y1:.1f} moveto {x2:.1f} {y2:.1f} lineto stroke\n".format(x1=x0+scale*x_min-5, y1=y0, x2=x0+scale*x_max+5, y2=y0))
        eps.write("newpath {x1:.1f} {y1:.1f} moveto {x2:.1f} {y2:.1f} lineto stroke\n".format(x1=x0, y1=y0+scale*y_min-5, x2=x0, y2=y0+scale*y_max+5))
        eps.write("1 setlinewidth\n")

        for c in curves:
            eps.write("newpath\n")
            index = 0
            for cp in c:
                x, y = cp
                eps.write("{x1:.1f} {y1:.1f} ".format(x1=x0+scale*x, y1=y0+scale*y))
                if index == 0:
                    eps.write("moveto\n")
                elif index % 3 == 0:
                    eps.write("curveto\n")
                index = index + 1
            eps.write("stroke\n")
