import os
import re
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from modeling.wrapper import load_gptj, load_mgpt, ModelWrapper, GPTWrapper, GPTJWrapper, get_device
from dataset_generation.generate_sentences import generate_sentences, format_sentence
from constants import VERBNOUN_CSV_PATH

modelname2setting = {"gptj": (load_gptj, GPTJWrapper), "mgpt": (load_mgpt, GPTWrapper)}

format_eng = " {np_sub} {verb_past} the {noun_obj}"
format_ger = " {np_sub} {haben} {article_acc} {noun_obj} {verb_perf}"
FORMATS_GER = [
    "",
    " {np_sub}",
    " {np_sub} {haben}",
    " {np_sub} {haben} {article_acc}",
    " {np_sub} {haben} {article_acc} {noun_obj}"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model", type=str)
    parser.add_argument("--cache", default=None, help="Cache directory", type=str)
    return parser.parse_args()

def main():
    PROMPT = "Q: Translate this phrase from English to German:{sentence_e}. A:{sentence_f}"
    verbnoun_df = pd.read_csv(VERBNOUN_CSV_PATH)
    print("Loading...")
    PHRASE = "Q: Translate this phrase from English to German: I ate the apple. A:"
    ANSWER = "Q: Translate this phrase from English to German: I ate the apple. A: Ich habe den Apfel gegessen."
    args = parse_args()
    model_func = modelname2setting[args.model][0]
    wrapper_class = modelname2setting[args.model][1]
    print("Initializing...")
    model, tokenizer = model_func(args.cache)
    wrapper = wrapper_class(model, tokenizer)

    eng_sentences, ger_sentences = generate_sentences(verbnoun_df, format_eng=format_eng, format_ger=format_ger)

    is_equal_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }
    for i, row in verbnoun_df.iterrows():
        cut_sentences = []
        eng_sentence = format_sentence(row, format_eng, 'en', np_sub="Ich", np_sub_eng="I")
        if not eng_sentence:
            continue

        ger_sentence = format_sentence(row, format_ger, 'ge', np_sub="Ich", np_sub_eng="I")
        prompt_full = PROMPT.format(sentence_e=eng_sentence, sentence_f=ger_sentence)

        for format_cut in FORMATS_GER:
            ger_sentence_cut = format_sentence(row, format_cut, 'ge', np_sub="Ich", np_sub_eng="I")
            cut_sentences.append(ger_sentence_cut)
        
        for sent_i, sent in enumerate(cut_sentences):
            prompt = PROMPT.format(sentence_e=eng_sentence, sentence_f=sent)
            input_ids = wrapper.tokenizer(prompt, return_tensors='pt').input_ids.to(get_device())
            input_ids_full = wrapper.tokenizer(prompt_full, return_tensors='pt').input_ids.to(get_device())
            output = wrapper.model.generate(input_ids, max_length=(input_ids.shape[1]+1))
            is_equal = (output[:, input_ids.shape[1]:] == input_ids_full[:, input_ids.shape[1]:input_ids.shape[1]+1])
            is_equal_dict[sent_i].append(is_equal[0])

            # print("============")
            # print(prompt)
            # print(input_ids)
            # print(prompt_full)
            # print(input_ids_full)
            # print(input_ids_full[:, input_ids.shape[1]:])
            # print(output[:, input_ids.shape[1]:])
            # print(is_equal)
            # print("============")

    print(torch.tensor(is_equal_dict[0], dtype=torch.int))
    print(torch.tensor(is_equal_dict[1], dtype=torch.int))
    print(torch.tensor(is_equal_dict[2], dtype=torch.int))
    print(torch.tensor(is_equal_dict[3], dtype=torch.int))
    print(torch.tensor(is_equal_dict[4], dtype=torch.int))

    print(torch.tensor(is_equal_dict[0], dtype=torch.float).mean())
    print(torch.tensor(is_equal_dict[1], dtype=torch.float).mean())
    print(torch.tensor(is_equal_dict[2], dtype=torch.float).mean())
    print(torch.tensor(is_equal_dict[3], dtype=torch.float).mean())
    print(torch.tensor(is_equal_dict[4], dtype=torch.float).mean())







    # print('HHH')
    # print(wrapper.tokenizer(PHRASE).input_ids)
    # input_ids = wrapper.tokenizer(PHRASE, return_tensors='pt').input_ids.to('cuda:0')
    # output = wrapper.model.generate(input_ids)
    # print(output)
    # print(wrapper.tokenizer.decode(output[0]))

if __name__ == "__main__":
    main()