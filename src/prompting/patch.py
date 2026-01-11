import os
import re
import json
import torch
import argparse
import pandas as pd
import pyvene as pv
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
import torch.nn.functional as F
from modeling.wrapper import get_device

from modeling.wrapper import load_gptj, load_mgpt, load_llama, ModelWrapper, GPTWrapper, GPTJWrapper, LLamaWrapper
from utils.model_args import parse_model_args, modelname2setting, ParseArg

def get_token_interpretable(sentence, token, tokenizer, add_space: bool = False) -> int:
    if add_space:
        # Add a space before the token to make it interpretable
        sentence_contd = sentence + " " + token
    else:
        sentence_contd = sentence+token
    tokens_bare = tokenizer(sentence).input_ids
    tokens_contd = tokenizer(sentence_contd).input_ids
    token_index = tokens_contd[len(tokens_bare)]
    return token_index

def intervene(model,
    base_encoding: torch.tensor,
    source_encoding: torch.tensor,
    layer_set: list,
    component: str = "block_output",
    ) -> torch.tensor:

    base_last_token_index = len(base_encoding['input_ids']) - 1  # Use len() to get the length of the sequence
    source_last_token_index = len(source_encoding['input_ids']) - 1

    # Create intervention for specific layers
    config = pv.IntervenableConfig([{
            "layer": l,
            "component": "block_output", #"mlp_output",
            "intervention_type": pv.VanillaIntervention,
            } for l in layer_set], # Pass a list instead of a single layer
            mode="parallel"
    )
    pv_model = pv.IntervenableModel(config, model=model)

    _, intervened_outputs = pv_model(
        # the base input
        base=base_encoding,
        # the source input
        sources=source_encoding,
        # the location to intervene at (last token)
        unit_locations={"sources->base": (source_last_token_index, base_last_token_index)},
    )

    distrib = intervened_outputs.logits
    logits = distrib[0][-1]

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=-1)

    return probabilities

def main():
    args = parse_model_args(
        ParseArg("data", type=str, help="Path to the dataset containing sentences"),
        ParseArg("--columns", type=str, nargs="+", help="Columns with tokens of interest"),
        ParseArg("--prompts-base", type=str, default="base", help="Base prompt column"),
        ParseArg("--prompts-source", type=str, default="source", help="Source prompt column"),
        ParseArg("--save-df", type=str, default="./probs.csv", help="Path to save the probabilities"),
        ParseArg("--login-hf", type=str, default=None, help="Huggingface login token"),
    )
    if args.login_hf:
        from huggingface_hub import login
        login(args.login_hf)
    device = get_device()
    model_func, wrapper_class = modelname2setting[args.model]
    model, tokenizer = model_func(args.cache)
    wrapper = wrapper_class(model, tokenizer)
    dataset = pd.read_csv(args.data)
    dataset = dataset.fillna("")
    layer_combinations = [[i] for i in range(0, wrapper.num_layers)] + \
        [[i, i+1] for i in range(0, wrapper.num_layers - 1)]
    
    # Get list of relevant tokens per row
    relevant_tokens = []
    for row_index, row in dataset.iterrows():
        tokens = {}
        for col in args.columns:
            tokens[col] = get_token_interpretable(row[args.prompts_base], row[col], tokenizer)
        relevant_tokens.append(tokens)

    probs_df = pd.DataFrame(columns=["prompt_base", "prompt_source", "layer_combination", "probabilities"])
    for prompt_id, prompt_row in dataset.iterrows():
        # TODO: change the strip. some prompts might need the newline
        prompt_base = prompt_row[args.prompts_base].strip()
        prompt_source = prompt_row[args.prompts_source].strip()
        rel_tokens = relevant_tokens[prompt_id]

        # Tokenize the prompts
        base_enc = tokenizer(prompt_base, return_tensors='pt').to(device)
        source_enc = tokenizer(prompt_source, return_tensors='pt').to(device)
        print("Processing prompt:")
        print(prompt_base)
        print(prompt_source)
        for layer_set in tqdm(layer_combinations):
            probabilities_dict = {}
            probabilities_dict_test = {}
            probabilities = intervene(
                model,
                base_encoding=base_enc,
                source_encoding=source_enc,
                layer_set=layer_set
            )

            for token_type, token_index in rel_tokens.items():
                probabilities_dict[token_type] = probabilities[token_index].item()
                probabilities_dict_test[token_index] = probabilities[token_index].item()
            
            if layer_set == [0]:
                print("AT LAYER 0 INTERVENTION PROBS:")
                print(probabilities_dict)
                print(probabilities_dict_test)
                # print("TOP TOKENS: ")
                # print(tokenizer.convert_ids_to_tokens(torch.topk(probabilities[0, -1], 10).indices[0]))

            probs_df = pd.concat([probs_df, pd.DataFrame({
                "prompt_base": [prompt_base],
                "prompt_source": [prompt_source],
                "layer_combination": [layer_set],
                "probabilities": [json.dumps(probabilities_dict, ensure_ascii=False)]
            })], ignore_index=True)
    probs_df.to_csv(args.save_df, index=False)

if __name__ == "__main__":
    main()