"""Computes projected probabilities for all layers of a given autoregressive model
and writes them to a .pt file."""

import os
import re
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt

from modeling.wrapper import load_gptj, load_mgpt, load_llama, ModelWrapper, GPTWrapper, GPTJWrapper, LLamaWrapper
from utils.model_args import parse_model_args, modelname2setting, ParseArg

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
        ParseArg("data", type=str, help="Path to the dataset containing sentences"),
        ParseArg("--prompt-col", type=str, help="Column with prompt"),
        ParseArg("--torch-save",  type=str, help="Directory to save the probabilities")
    )

    model_func, wrapper_class = modelname2setting[args.model]
    model, tokenizer = model_func(args.cache)
    wrapper = wrapper_class(model, tokenizer)
    dataset = pd.read_csv(args.data)
    dataset = dataset.fillna("")

    # prompt_format = args.prompt_format.replace('\\n', '\n')
    save_path = args.torch_save
    logits_matrix = torch.zeros(len(dataset), wrapper.num_layers, wrapper.hidden_size)

    for i, (row_index, row) in tqdm(enumerate(list(dataset.iterrows()))):
        prompt = row[args.prompt_col].strip()
        # TODO: remove this
        if i == 0:
            print(prompt)
        with torch.no_grad():
            logits = wrapper.get_logits(wrapper.tokenize(prompt))
            logits = torch.stack(logits, dim=0).squeeze()  # Stack logits to create a tensor
            logits_cpu = logits.cpu()  # Move tensor to CPU to free GPU memory
            del logits
            torch.cuda.empty_cache()  # Clear CUDA cache to free memory
        logits_matrix[i] = logits_cpu[1:, -1, :] # Exclude the embedding layer
        del logits_cpu
        torch.cuda.empty_cache()
    torch.save(logits_matrix.cpu(), save_path)

if __name__ == "__main__":
    main()
