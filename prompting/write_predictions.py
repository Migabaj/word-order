import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from modeling.wrapper import load_gptj, load_mgpt, ModelWrapper, GPTWrapper, GPTJWrapper

modelname2setting = {"gptj": (load_gptj, GPTJWrapper), "mgpt": (load_mgpt, GPTWrapper)}

# model, tokenizer = load_gptj(cache_dir=CACHE_DIR)  # load_gpt2('gpt2-medium')
# model = model.float()
# wrapper = GPTJWrapper(model, tokenizer)
PROMPT_HEADER = "Q: Translate this phrase from English to German:"
# UNTIL_WORD = r".+?ha((be)|(st)|t)"


def token_word_overlap(token: str, word: str, wrapper: ModelWrapper) -> bool:
    """See whether the token is the same as the first subtoken of a given word"""
    tokenized_word = wrapper.tokenizer.tokenize(" " + word)
    return tokenized_word[0] == token


def generate_prompt(row, prompt_format, until_regex, col_e, col_f):
    sentence_e = row[col_e]
    sentence_f = row[col_f]
    sentence_f = re.search(until_regex, sentence_f).group(0)

    prompt = prompt_format.format(sentence_e=sentence_e, sentence_f=sentence_f)
    return prompt


def add_predictions(
    wrapper, row, prompt, pred_dict, topk_dict, token_columns, k
):
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
        # for key in topk_dict[layer_i].keys():
        #     word = row[key]
        #     if any(token_word_overlap(token, word, wrapper) for token in tokens):
        #         topk_dict[layer_i][key].append(True)
        #     else:
        #         topk_dict[layer_i][key].append(False)

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


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model", type=str)
    parser.add_argument("data", help="Sentence dataset path", type=str)
    parser.add_argument(
        "-p",
        "--prompt-format",
        default="Q: Translate this phrase from English to German: {sentence_e} A: {sentence_f}",
        help="Format for prompt",
    )
    parser.add_argument(
        "-u", "--until", help="Regex that defines where to stop the target sentence."
    )
    parser.add_argument("-w", "--write", help="Path to save prediction info", type=str)
    parser.add_argument(
        "-k", default=3, help="Number of top-k tokens to document", type=int
    )
    parser.add_argument(
        "--token-columns",
        default="",
        nargs="+",
        help="Dataset's columns containing specific tokens to investigate the model's predictions",
        type=str,
    )
    parser.add_argument(
        "--column-e",
        default="sentence_e",
        help="Dataset's column that contains the text in the source language",
        type=str,
    )
    parser.add_argument(
        "--column-f",
        default="sentence_f",
        help="Dataset's column that contains the text in the target language",
        type=str,
    )
    parser.add_argument("--cache", default=None, help="Cache directory", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    model_func = modelname2setting[args.model][0]
    wrapper_class = modelname2setting[args.model][1]
    model, tokenizer = model_func(args.cache)
    wrapper = wrapper_class(model, tokenizer)
    pred_dict, topk_dict = pred_dicts(args.k, args.token_columns, wrapper)
    dataset = pd.read_csv(args.data, index_col=0)

    for _, row in tqdm(dataset.iterrows()):
        prompt = generate_prompt(row, args.prompt_format, args.until, args.column_e, args.column_f)
        pred_dict, topk_dict = add_predictions(wrapper, row, prompt, pred_dict, topk_dict, args.token_columns, args.k)
    
    dataset = insert_dicts_into_dataframe(pred_dict, topk_dict, dataset)
    dataset.to_csv(args.write)

    #     sentence_eng = row.sentence_eng
    #     sentence_ger = row.sentence_ger

    #     sentence_ger_until_haben = re.search(UNTIL_HABEN, sentence_ger).group(0)
    #     prompt = f"{PROMPT_HEADER} {sentence_eng}\nA: {sentence_ger_until_haben}"
    #     inp_ids = wrapper.tokenize(prompt)
    #     logits = wrapper.get_layers(inp_ids)
    #     ids_per_layer = wrapper.get_top_ids_per_layer(logits, k=K)

    #     for layer_i, layer_preds in enumerate(ids_per_layer[1:]):
    #         tokens = wrapper.tokenizer.convert_ids_to_tokens(layer_preds)
    #         for i, token in enumerate(tokens):
    #             layer_predictions_dict[f"layer_{layer_i}_pred_{i}"].append(token)
    #         for key in word_predicted_topk_dict[layer_i].keys():
    #             word = row[key]
    #             if any(token_word_overlap(token, word) for token in tokens):
    #                 word_predicted_topk_dict[layer_i][key].append(True)
    #             else:
    #                 word_predicted_topk_dict[layer_i][key].append(False)

    # for label, preds in layer_predictions_dict.items():
    #     dataset.insert(len(dataset.columns), label, preds)
    # for layer_i, pred_dict in word_predicted_topk_dict.items():
    #     for key, predicted in word_predicted_topk_dict[layer_i].items():
    #         # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top3", predicted)
    #         dataset[f"layer_{layer_i}_{key}_top3"] = predicted
    #         # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top1", predicted)
    #         dataset["layer_{layer_i}_{key}_top1"] = predicted

if __name__ == "__main__":
    main()
