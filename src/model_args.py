import argparse
from dataclasses import dataclass

from modeling.wrapper import load_mgpt, load_gptj, GPTWrapper, GPTJWrapper

@dataclass
class ParseArg:
    argname: str
    argtype: type
    default: str = None
    help: str = ""
    nargs: int = None

modelname2setting = {
    "gptj": (load_gptj, GPTJWrapper),
    "mgpt": (load_mgpt, GPTWrapper)
    }

def parse_model_args(*args: ParseArg) -> argparse.Namespace:
    """
    Parse model arguments from command line.
    :param args: Additional arguments to be added to the parser.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model", type=str)
    for argument in args:
        parser.add_argument(f"{argument.argname}", help=argument.help, nargs=argument.nargs, type=argument.argtype, default=argument.default)
    parser.add_argument("--cache", default=None, help="Cache directory", type=str)
    return parser.parse_args()