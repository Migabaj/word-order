import os
import torch
import pandas as pd
from modeling import load_gptj, GPTJWrapper
import circuitsvis as cv
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

from constants import CACHE_DIR

DEVICE = "cuda"
NUM_HEADS = 16
NUM_LAYERS = 28
DATASET_PATH = "word_order_logitlens/dataset.csv"
PROMPT_PREFIX = "Q: Translate this phrase from English to German:"

dataset = pd.read_csv("word_order_logitlens/dataset.csv")
# model = HookedTransformer.from_pretrained(
#     "EleutherAI/gpt-j-6B", cache_dir=CACHE_DIR, device=DEVICE
# )
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=CACHE_DIR)
model, tokenizer = load_gptj(cache_dir=CACHE_DIR)
print(model)
wrapper = GPTJWrapper(model, tokenizer)


def test():
    prompt_ids = wrapper.tokenize(PROMPT_PREFIX)
    _, attn = wrapper.get_layers_w_attns(prompt_ids)
    print(len(attn[1]))
    print(attn[1][0].shape)


def main():
    for sent_i, row in enumerate(dataset.values):
        filelist = os.listdir("word_order_logitlens/attn/") # TODO: don't hardcode
        # if any(filename.startswith(f"{sent_i}-") for filename in filelist):
        #     continue
        # if sent_i < 15:
        #     continue
        sentence = row[0]
        translation = row[1]
        prompt = f"{PROMPT_PREFIX} {sentence}\nA: {translation}"
        prompt_ids, prompt_tok = wrapper.tokenize_with_string(prompt)
        # prompt_ids = wrapper.tokenize(PROMPT_PREFIX)
        first_word_marker = -prompt_tok[::-1].index(":")

        for j in range(first_word_marker, 1):
            if j == 0:
                offset = None # [:None] instead of [:0]
            else:
                offset = j
            inp_ids_segment = prompt_ids[:, :offset]
            inp_tok_segment = prompt_tok[:offset]
            # logits, cache = model.run_with_cache(
            #     inp_tokens_segment, remove_batch_dim=True
            # )
            _, attn = wrapper.get_layers_w_attns(inp_ids_segment)
            attention_pattern = torch.zeros(
                NUM_LAYERS,
                1, # TODO: do something about this
                NUM_HEADS,
                len(inp_ids_segment[0]),
                len(inp_ids_segment[0]),
            )
            for l in range(NUM_LAYERS):
                attention_pattern[l] = attn[l].unsqueeze(0)
                # attention_pattern[l] = cache["pattern", l, "attn"]

            torch.save(
                attention_pattern, f"word_order_logitlens/attn/{sent_i}-off{-j}.pt"
            )

            for l in range(NUM_LAYERS):
                rendered_html = cv.attention.attention_patterns(
                    tokens=inp_tok_segment, attention=attention_pattern[l, 0]
                )
                with open(
                    f"word_order_logitlens/attn/{sent_i}-off{-j}-l{l}.html",
                    "w",
                    encoding="utf-8",
                ) as fw:
                    fw.write(str(rendered_html))


if __name__ == "__main__":
    main()
    # test()
