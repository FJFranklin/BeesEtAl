import argparse
import csv

import numpy as np

from BeesEtAl.Base_Optimiser import Base_Optimiser

parser = argparse.ArgumentParser(description="Reload a results file and determine the Pareto-optimal set.")

parser.add_argument('source',         help='Specify input file name.', type=str)
parser.add_argument('column',         help='Specify one or more column indices (from 0) as costs.', type=int, nargs='+')
parser.add_argument('--out',          help='Specify output file name [pareto.csv].', type=str, default='pareto.csv')
parser.add_argument('--save',         help='Specify output file name for image [pareto.png].', type=str, default='pareto.png')

args = parser.parse_args()

file_name = args.source
indices   = args.column

BO = None

print('Source file: {s}'.format(s=file_name))
with open(file_name, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        values = np.asarray(list(map(float, row)))
        cost   = values[indices]

        if BO is None:
            BO = Base_Optimiser(values, values)
        BO.push(cost, values)

BO.pareto(args.out)

if len(indices) > 1:
    from BeesEtAl.Base_Plotter import Base_Plotter

    BP = Base_Plotter(BO, None)

    if len(indices) < 3:
        BP.pareto([0, 1])
        BP.save(args.save)
        BP.sync(10)
    else:
        BP.pareto([0, 1, 2])
        BP.save(args.save)
        BP.sync(10)
