import yaml
import papermill as pm
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('--input', type=str, default='notebooks/plot_logitlens_past-part.ipynb', help='Input notebook path')
arg_parser.add_argument('--output', type=str, default='notebooks/plot_logitlens_past-part.ipynb', help='Output notebook path')
arg_parser.add_argument('--kernel', type=str, default='python3', help='Kernel name to use when executing the notebook (overrides notebook metadata)')
arg_parser.add_argument('--params-path', type=str, default=None, help='Path to yaml file with parameters, if needed')
args = arg_parser.parse_args()

if args.params_path is not None:
   with open(args.params_path) as f:
      params = yaml.load(f, Loader=yaml.FullLoader)
# Pass the kernel name to papermill to avoid "No kernel name found" errors when
# the notebook metadata lacks a kernelspec. This lets users override the kernel
# from the command line or rely on the sensible default 'python3'.
   pm.execute_notebook(
      args.input,
      args.output,
      kernel_name=args.kernel,
      parameters=params,
      log_output=True
   )
else:
   pm.execute_notebook(
      args.input,
      args.output,
      kernel_name=args.kernel,
      log_output=True
   )