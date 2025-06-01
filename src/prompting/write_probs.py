import os
import re
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt

from modeling.wrapper import load_gptj, load_mgpt, ModelWrapper, GPTWrapper, GPTJWrapper
from model_args import parse_model_args, modelname2setting, ParseArg

def generate_prompt(row : pd.Series, prompt_format : str, col_e : str, col_f : str) -> str:
    """Generate the prompt for the model
    
    :param row: The row of the dataframe
    :param prompt_format: The format of the prompt
    :param col_e: The column that contains the text in the source language
    :param col_f: The column that contains the text in the target language
    :return: The prompt
    """
    sentence_e = row[col_e]
    sentence_f = row[col_f]

    prompt = prompt_format.format(sentence_e=sentence_e, sentence_f=sentence_f)
    return prompt

def get_probability_matrix(wrapper: ModelWrapper, inp_ids, take_first_layer=False):
    """Get the probability matrix for the input ids
    
    :param wrapper: The model wrapper
    :param inp_ids: The input ids
    :return: The probability matrix"""
    logits = wrapper.get_layers(inp_ids)
    probs = wrapper.get_probs_per_layer(logits, take_first_layer=take_first_layer)
    return probs

def main():
    args = parse_model_args(
        ParseArg("data", argtype=str, help="Path to the dataset containing sentences"),
        ParseArg("--prompt-format", argtype=str, default="Q: Translate this phrase from English to German: {sentence_e} A: {sentence_f}", help="Format for prompt"),
        ParseArg("--column-e", argtype=str, default="sentence_e", help="Column with English sentences"),
        ParseArg("--columns-f", argtype=str, nargs="+", help="Columns with sentences to have as the prompt"),
        ParseArg("--torch-save",  argtype=str, help="Directory to save the probabilities")
    )

    model_func, wrapper_class = modelname2setting[args.model]
    model, tokenizer = model_func(args.cache)
    wrapper = wrapper_class(model, tokenizer)
    dataset = pd.read_csv(args.data, index_col=0)

    for column_f in args.columns_f:
        save_path = os.path.join(args.torch_save, f"probs_{column_f}.pt")
        prob_matrix = torch.zeros(len(dataset), wrapper.num_layers, len(wrapper.tokenizer), dtype=torch.float16)

        for i, (row_index, row) in tqdm(enumerate(list(dataset.iterrows()))):
            prompt = generate_prompt(
                row, args.prompt_format, args.column_e, column_f
            )
            with torch.no_grad():
                probs = get_probability_matrix(wrapper, wrapper.tokenize(prompt), take_first_layer=False)
                probs_cpu = probs.cpu()  # Move tensor to CPU to free GPU memory
                del probs
                torch.cuda.empty_cache()  # Clear CUDA cache to free memory
            prob_matrix[i] = probs_cpu
            del probs_cpu
            torch.cuda.empty_cache()
        torch.save(prob_matrix.cpu(), save_path)

if __name__ == "__main__":
    main()
