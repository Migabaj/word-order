import argparse
import dataclasses
from typing import Union, List, Optional
from dataclasses import dataclass

from modeling.wrapper import load_mgpt, load_gptj, load_gptneo, load_llama, load_eurollm, \
    GPTWrapper, GPTJWrapper, LLamaWrapper, EuroLLMWrapper

model_to_num_layers_attr = {
    "GPT2LMHeadModel": "config.n_layer",
    "CohereForCausalLM": "config.num_hidden_layers",
    "LlamaForCausalLM": "config.num_hidden_layers",
}

model_to_num_heads_attr = {
    "GPT2LMHeadModel": "config.n_head",
    "CohereForCausalLM": "config.num_attention_heads",
    "LlamaForCausalLM": "config.num_attention_heads",
}

model_to_short = {
    "ai-forever/mGPT": "mgpt",
    "CohereLabs/aya-expanse-8b": "aya-expanse-8b",
    "meta-llama/Meta-Llama-3-8B": "llama-3-8b",
}

model_to_nounadj_head = {
    "ai-forever/mGPT": (11, 2),
    "CohereLabs/aya-expanse-8b": (15, 0),
    "meta-llama/Meta-Llama-3-8B": (14, 25),
}

@dataclass
class ParseArg:
    argname: Union[str, List[str]]
    type: type = None
    default: str = None
    help: str = ""
    nargs: Optional[Union[int, str]] = None
    action: Optional[Union[str, argparse.Action]] = None

modelname2setting = {
    "gptj": (load_gptj, GPTJWrapper),
    "mgpt": (load_mgpt, GPTWrapper),
    "gptneo": (load_gptneo, GPTWrapper),
    "llama": (load_llama, LLamaWrapper),
    "eurollm": (load_eurollm, EuroLLMWrapper)
    }

def parse_plain_args(*args: ParseArg) -> argparse.Namespace:
    """
    Parse arguments from command line.
    :param args: Additional arguments to be added to the parser.

    """
    parser = argparse.ArgumentParser()
    for argument in args:
        # TODO: Do you really need parse_args?
        # Create a dict, so that unexpected kwargs don't appear in add_argument
        argument_keywords = {
            'help': argument.help,
            'default': argument.default,
            'action': argument.action,
        }
        fields = dataclasses.fields(argument)
        for fieldname in ['nargs', 'type']:
            if getattr(argument, fieldname) is not None:
                argument_keywords[fieldname] = getattr(argument, fieldname)
        if isinstance(argument.argname, str):
            argument.argname = [argument.argname]
        parser.add_argument(
            *argument.argname,
            **argument_keywords
            )
    return parser.parse_args()

def parse_model_args(*args: ParseArg) -> argparse.Namespace:
    """
    Parse model arguments from command line.
    :param args: Additional arguments to be added to the parser.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model", type=str)
    for argument in args:
        parser.add_argument(
            f"{argument.argname}",
            help=argument.help,
            nargs=argument.nargs,
            type=argument.type,
            default=argument.default,
            action=argument.action
            )
    parser.add_argument("--cache", default=None, help="Cache directory", type=str)
    return parser.parse_args()