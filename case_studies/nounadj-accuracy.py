"""Compute accuracy for nouns and adjective predicitons (compared to noise)"""
import torch
import string
import random
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from modeling.wrapper import load_llama, LLamaWrapper

prompt_template = """INSTRUCTION: Translate the phrase "{sentence_e}" from English to French.
ANSWER: English: "{sentence_e}" - Français: \"{sentence_f}"""
NOUN_ADJ_DATA = "word-order-thesis/data/noun-adj.csv"

FRENCH_COLUMNS = ["phrase_empty", "noun_fre"]
FRENCH_WORDS = ["noun_fre", "adj_fre"]

def main():
    hf_token = "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"
    login(hf_token)
    model, tokenizer = load_llama(cache_dir="/scratch/msonkin/word-order-thesis/cache/")
    wrapper = LLamaWrapper(model, tokenizer)

    df = pd.read_csv(NOUN_ADJ_DATA)
    df = df.fillna("")
    nouns = df.noun_eng

    average_prob_nouns = torch.zeros(wrapper.num_layers)
    average_prob_adjectives = torch.zeros(wrapper.num_layers)

    for col_f, word_f in zip(FRENCH_COLUMNS, FRENCH_WORDS):
        average_token_prob = torch.zeros(wrapper.num_layers)
        for i, row in df.iterrows():
            prompt = prompt_template.format(sentence_e=row.phrase_eng, sentence_f=row[col_f])
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            out = wrapper.model(input_ids=prompt_ids, output_hidden_states=True, max_length=124)
            logits = torch.stack(wrapper.layer_decode(out.hidden_states)).squeeze()[1:]
            probs = wrapper.get_probs_per_layer(logits, dtype=torch.float32)
            for i, layer in enumerate(probs):
                average_token_prob[i] += layer[tokenizer(row[word_f]).input_ids[1]]
        print(prompt)
        print(tokenizer(row[word_f]))
        average_token_prob = average_token_prob / len(nouns)
        print(f"Average probability of the 'correct' token for word {word_f}, column {col_f}:")
        print(average_token_prob)

if __name__ == "__main__":
    main()
