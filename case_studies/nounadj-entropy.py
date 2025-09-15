"""Compute entropy for nouns and adjective predicitons (compared to noise)"""
import torch
import string
import random
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from modeling.wrapper import load_llama, LLamaWrapper

prompt_template = """INSTRUCTION: Translate this word from English to French.
English: "{word}".
French: \""""
NOUN_ADJ_DATA = "word-order-thesis/data/noun-adj.csv"

def main():
    hf_token = "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"
    login(hf_token)
    model, tokenizer = load_llama(cache_dir="/scratch/msonkin/word-order-thesis/cache/")
    wrapper = LLamaWrapper(model, tokenizer)

    df = pd.read_csv(NOUN_ADJ_DATA)
    nouns = df.noun_eng
    adjectives = df.adj_eng

    average_entropy_nouns = torch.zeros(wrapper.num_layers)
    average_entropy_adjectives = torch.zeros(wrapper.num_layers)
    average_entropy_noise = torch.zeros(wrapper.num_layers)
    for noun in tqdm(nouns):
        prompt = prompt_template.format(word=noun)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = wrapper.model(input_ids=prompt_ids, output_hidden_states=True, max_length=124)
        logits = torch.stack(wrapper.layer_decode(out.hidden_states)).squeeze()[1:]
        probs = wrapper.get_probs_per_layer(logits, dtype=torch.float32)
        for i, layer in enumerate(probs):
            average_entropy_nouns[i] += wrapper.entropy(layer)
    average_entropy_nouns = average_entropy_nouns / len(nouns)
    
    for adj in tqdm(adjectives):
        prompt = prompt_template.format(word=adj)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = wrapper.model(input_ids=prompt_ids, output_hidden_states=True, max_length=124)
        logits = torch.stack(wrapper.layer_decode(out.hidden_states)).squeeze()[1:]
        probs = wrapper.get_probs_per_layer(logits, dtype=torch.float32)
        for i, layer in enumerate(probs):
            average_entropy_adjectives[i] += wrapper.entropy(layer)
    average_entropy_adjectives = average_entropy_adjectives / len(adjectives)

    for _ in tqdm(range(len(adjectives))):
        noise = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(3,8)))
        prompt = prompt_template.format(word=noise)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = wrapper.model(input_ids=prompt_ids, output_hidden_states=True, max_length=124)
        logits = torch.stack(wrapper.layer_decode(out.hidden_states)).squeeze()[1:]
        probs = wrapper.get_probs_per_layer(logits, dtype=torch.float32)
        for i, layer in enumerate(probs):
            average_entropy_noise[i] += wrapper.entropy(layer)
    average_entropy_noise = average_entropy_noise / len(adjectives)
    
    print(average_entropy_nouns)
    print(average_entropy_adjectives)
    print(average_entropy_noise)

if __name__ == "__main__":
    main()
