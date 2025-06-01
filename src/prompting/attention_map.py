import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import circuitsvis as cv
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
from typing import Optional

from modeling.wrapper import load_gptj, GPTJWrapper

PROMPT_PREFIX = "Q: Translate this phrase from English to German:" # TODO: more universal

def get_args():
    """Create an argument parser and return the user input"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model", type=str)
    parser.add_argument(
        "dataset", help="path to dataset", type=str
    )
    parser.add_argument(
        "savedir", help="path to save visualisations", type=str
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="cache directory for model",
        type=str,
    )
    parser.add_argument(
        "--device", required=False, default="cpu", help="device", type=str
    )
    args = vars(parser.parse_args())
    return args


def get_wrapper(model_name : str, cache_dir : Optional[str], device : str) -> GPTJWrapper:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    wrapper = GPTJWrapper(model, tokenizer) # TODO: for different models!!
    return wrapper

# TODO: figure out all this
def get_attentions(row : pd.Series, offset : int) -> torch.Tensor:
    """Extracts attention grid from prompt with cutoff.
    """

def convert_attentions_to_html(attentions : torch.Tensor) -> str:
    """Converts attention grid to HTML.
    """

def combine_attention_htmls(main_html : str, htmls : List[str]) -> str:
    """Combine layer attention HTMLs into one.
    """

def main():
    """Main function
    """
    args = get_args()
    wrapper = get_wrapper(args["model"], args["cache"], args["device"])
    n_layer = wrapper.model.config.n_layer
    n_head = wrapper.model.config.n_head
    dataset = pd.read_csv(args["dataset"])

    for sent_i, row in tqdm(dataset.iterrows()):
        sentence = row.sentence_eng
        translation = row.sentence_ger
        prompt = f"{PROMPT_PREFIX} {sentence}\nA: {translation}"
        prompt_ids, prompt_tok = wrapper.tokenize(prompt, output_string=True)
        first_word_marker = -prompt_tok[::-1].index(":")

        for j in range(first_word_marker, 1):
            

            if j == 0:
                offset = None  # [:None] instead of [:0]
            else:
                offset = j
            inp_ids_segment = prompt_ids[:, :offset]
            inp_tok_segment = prompt_tok[:offset]
            # logits, cache = model.run_with_cache(
            #     inp_tokens_segment, remove_batch_dim=True
            # )
            _, attn = wrapper.get_layers_w_attns(inp_ids_segment)
            attention_pattern = torch.zeros(
                n_layer,
                1,  # TODO: do something about this
                n_head,
                len(inp_ids_segment[0]),
                len(inp_ids_segment[0]),
            )
            for l in range(n_layer):
                attention_pattern[l] = attn[l].unsqueeze(0)
                # attention_pattern[l] = cache["pattern", l, "attn"]

            torch.save(
                attention_pattern, os.path.join(args["savedir"], f"{sent_i}-off{-j}.pt")
            )

            # Extracting and combining HTML pages into one per prompt.
            first_layer_html = cv.attention.attention_patterns(
                tokens=inp_tok_segment, attention=attention_pattern[0, 0]
            )
            first_layer_html_str = str(first_layer_html)
            first_layer_soup = BeautifulSoup(first_layer_html_str, "html.parser")
            first_layer_script = first_layer_soup.script
            div_ids = [first_layer_soup.script.string.split("\n")[3].strip(' ",')]
            for l in range(1, n_layer):  # skipping the first layer
                layer_html = cv.attention.attention_patterns(
                    tokens=inp_tok_segment, attention=attention_pattern[l, 0]
                )
                layer_html_str = str(layer_html)
                layer_soup = BeautifulSoup(layer_html_str, "html.parser")
                script = layer_soup.script
                # var_l = f'\nvar l_{l-1} = document.getElementById("{div_ids[-1]}")'
                first_layer_script.string += (
                    f'\nvar l_{l-1} = document.getElementById("{div_ids[-1]}")'
                )
                first_layer_script.string += "\n" + "\n".join(
                    script.string.split("\n")[2:]
                )

                new_div_id = script.string.split("\n")[3].strip(' ",')
                new_div = layer_soup.new_tag(
                    "div", style="margin: 15px 0;", id=new_div_id
                )
                first_layer_soup.append(new_div)
                div_ids.append(new_div_id)

            first_layer_script.string += (
                f'\nvar l_{l} = document.getElementById("{div_ids[-1]}")'
            )
            for l in range(n_layer):
                div_code = f"""
                var div_{l} = document.createElement("div");
                div_{l}.id = "{div_ids[l]}";
                div_{l}.style.margin = "15px 0";

                var h3 = document.createElement("h3");
                h3.innerText = "Attention Layer {l}"
                div_{l}.appendChild(h3);
                div_{l}.appendChild(l_{l});
                document.getElementsByTagName("body").item(0).appendChild(div_{l})
                """
                first_layer_script.string += div_code

            with open(
                os.path.join(args['savedir'], f"{sent_i}-off{offset}-layers.html"),
                "w",
                encoding="utf-8",
            ) as fw:
                fw.write(str(first_layer_soup))
            print("Done!")

            # with open(
            #     f"word_order_logitlens/attn/{sent_i}-off{-j}-l{l}.html",
            #     "w",
            #     encoding="utf-8",
            # ) as fw:
            #     fw.write(str(rendered_html))


if __name__ == "__main__":
    main()
