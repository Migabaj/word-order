import os
import re
import json

PWD = "/nethome/msonkin/word_order_logitlens/"
import sys
sys.path.append(PWD)
os.chdir(PWD)

import pandas as pd
from modeling import load_gptj, GPTJWrapper
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import CACHE_DIR, SENTENCE_CSV_PATH, VERBNOUN_COUNT_TOP1_PATH, VERBNOUN_COUNT_TOP3_PATH

model, tokenizer = load_gptj(cache_dir=CACHE_DIR) # load_gpt2('gpt2-medium')
model = model.float()
dataset = pd.read_csv(SENTENCE_CSV_PATH, index_col=0)
wrapper = GPTJWrapper(model, tokenizer)
PROMPT_HEADER = "Q: Translate this phrase from English to German:"
UNTIL_HABEN = r'.+?ha((be)|(st)|t)'
K = 3

# def token_word_overlap(token, word, bchar="Ġ"):
#     tokenized_word = wrapper.tokenizer.decode(wrapper.tokenizer(word))
#     if not token.startswith(bchar):
#         return False
#     token = token[len(bchar):]
#     if word.lower().startswith(token.lower()):
#         return True
#     return False

def token_word_overlap(token, word, bchar="Ġ"):
    tokenized_word = wrapper.tokenizer.tokenize(" "+word)
    return tokenized_word[0] == token

layer_predictions_dict = {}
for i in range(3):
    for l in range(len(wrapper.model.transformer.h)):
        layer_predictions_dict[f"layer_{l}_pred_{i}"] = []

word_predicted_top3_dict = {
    l:
        {
            "perf_eng": [],
            "past_eng": [],
            "perf_ger": [],
            "noun_eng": [],
            "noun_ger": []
        }
    for l in range(len(wrapper.model.transformer.h))
}
word_predicted_top1_dict = {
    l:
        {
            "perf_eng": [],
            "past_eng": [],
            "perf_ger": [],
            "noun_eng": [],
            "noun_ger": []
        }
    for l in range(len(wrapper.model.transformer.h))
}

for row_i, row in tqdm(dataset.iterrows()):
    sentence_eng = row.sentence_eng
    sentence_ger = row.sentence_ger

    sentence_ger_until_haben = re.search(UNTIL_HABEN, sentence_ger).group(0)
    prompt = f"{PROMPT_HEADER} {sentence_eng}\nA: {sentence_ger_until_haben}"
    inp_ids = wrapper.tokenize(prompt)
    logits = wrapper.get_layers(inp_ids)
    ids_per_layer = wrapper.get_top_ids_per_layer(logits, k=K)

    for layer_i, layer_preds in enumerate(ids_per_layer[1:]):
        tokens = wrapper.tokenizer.convert_ids_to_tokens(layer_preds)
        for i, token in enumerate(tokens):
            layer_predictions_dict[f"layer_{layer_i}_pred_{i}"].append(token)
        for key in word_predicted_top3_dict[layer_i].keys():
            word = row[key]
            if any(token_word_overlap(token, word) for token in tokens):
                word_predicted_top3_dict[layer_i][key].append(True)
            else:
                word_predicted_top3_dict[layer_i][key].append(False)
            
            if token_word_overlap(tokens[0], word):
                word_predicted_top1_dict[layer_i][key].append(True)
            else:
                word_predicted_top1_dict[layer_i][key].append(False)

for label, preds in layer_predictions_dict.items():
    dataset.insert(len(dataset.columns), label, preds)
for layer_i, pred_dict in word_predicted_top3_dict.items():
    for key, predicted in word_predicted_top3_dict[layer_i].items():
        # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top3", predicted)
        dataset[f"layer_{layer_i}_{key}_top3"] = predicted
for layer_i, pred_dict in word_predicted_top1_dict.items():
    for key, predicted in word_predicted_top1_dict[layer_i].items():
        # dataset.insert(len(dataset.columns), f"layer_{layer_i}_{key}_top1", predicted)
        dataset["layer_{layer_i}_{key}_top1"] = predicted

dataset.to_csv(SENTENCE_CSV_PATH)
