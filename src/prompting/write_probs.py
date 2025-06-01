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

# model, tokenizer = load_gptj(cache_dir=CACHE_DIR)  # load_gpt2('gpt2-medium')
# model = model.float()
# wrapper = GPTJWrapper(model, tokenizer)
PROMPT_HEADER = "Q: Translate this phrase from English to German:"
# UNTIL_WORD = r".+?ha((be)|(st)|t)"


def token_word_overlap(token: str, word: str, wrapper: ModelWrapper) -> bool:
    """See whether the token is the same as the first subtoken of a given word"""
    tokenized_word = wrapper.tokenizer.tokenize(" " + word)
    return tokenized_word[0] == token


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

def generate_until_space():
    pass

def get_probability_matrix(wrapper: ModelWrapper, inp_ids, take_first_layer=False):
    """Get the probability matrix for the input ids
    
    :param wrapper: The model wrapper
    :param inp_ids: The input ids
    :return: The probability matrix"""
    logits = wrapper.get_layers(inp_ids)
    probs = wrapper.get_probs_per_layer(logits, take_first_layer=take_first_layer)
    return probs


def add_predictions(
    wrapper: ModelWrapper, row: pd.Series, prompt: str, pred_dict: Dict[str, str], topk_dict, token_columns, k, save_torch, return_probs=[]
):
    """Add the predictions to the dictionary
    
    :param wrapper: The model wrapper
    :param row: The row of the dataframe
    :param prompt: The prompt
    :param pred_dict: The dictionary to store the predictions
    :param topk_dict: The dictionary to store the top-k predictions
    :param token_columns: The columns that contain the tokens
    :param k: The number of top-k predictions
    :param save_torch: The path to save the probabilities
    :param return_probs: The indices of the tokens to return the probabilities for
    :return: The dictionaries with the predictions"""
    inp_ids = wrapper.tokenize(prompt)
    logits = wrapper.get_layers(inp_ids)
    ids_per_layer = wrapper.get_top_ids_per_layer(logits, k=k)

    for layer_i, layer_preds in enumerate(ids_per_layer[1:]):
        tokens = wrapper.tokenizer.convert_ids_to_tokens(layer_preds)
        for i, token in enumerate(tokens):
            pred_dict[f"layer_{layer_i}_pred_{i}"].append(token)
        for i, col in enumerate(token_columns):
            word = row[col]
            if any(token_word_overlap(token, word, wrapper) for token in tokens):
                topk_dict[f"top{k}_{layer_i}_{col}"].append(True)
            else:
                topk_dict[f"top{k}_{layer_i}_{col}"].append(False)
    if return_probs:
        token_probs = wrapper.get_probs_per_layer(logits)[:, return_probs]
        torch.save(token_probs, save_torch)
    return pred_dict, topk_dict

def insert_dicts_into_dataframe(pred_dict, topk_dict, df):
    df_pred = pd.DataFrame.from_dict(pred_dict)
    df_topk = pd.DataFrame.from_dict(topk_dict)
    df = pd.concat([df, df_pred, df_topk], axis=1)
        
    # for label, preds in pred_dict.items():
    #     df.insert(len(df.columns), label, preds)
    # for layer_i, pred_dict in topk_dict.items():
    #     for key, predicted in topk_dict[layer_i].items():
    #         # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top3", predicted)
    #         df[f"layer_{layer_i}_{key}_top3"] = predicted
    #         # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top1", predicted)
    return df

def pred_dicts(k, token_columns, wrapper):
    layer_predictions_dict = {}
    for i in range(k):
        for l in range(len(wrapper.model.transformer.h)): # TODO: other models
            layer_predictions_dict[f"layer_{l}_pred_{i}"] = []

    word_predicted_topk_dict = {}
    for col in token_columns:
        for l in range(len(wrapper.model.transformer.h)):
            word_predicted_topk_dict[f"top{k}_{l}_{col}"] = []
    # word_predicted_topk_dict = {
    #     l: {f"{col}": [] for col in token_columns}
    #     for l in range(len(wrapper.model.transformer.h))
    # }
    return layer_predictions_dict, word_predicted_topk_dict

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
