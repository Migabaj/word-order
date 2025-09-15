import papermill as pm
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('--input', type=str, default='notebooks/plot_logitlens_past-part.ipynb', help='Input notebook path')
arg_parser.add_argument('--output', type=str, default='notebooks/plot_logitlens_past-part.ipynb', help='Output notebook path')
args = arg_parser.parse_args()

pm.execute_notebook(
   args.input,
   args.output
)