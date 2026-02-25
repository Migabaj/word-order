import torch
import tqdm as tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from create_datasets.parallel_dataset import ParallelDataset

MODELS = [
    {"model_id": "ai-forever/mGPT", "short": "mgpt", "head": (11, 2)},
    {"model_id": "CohereLabs/aya-expanse-8b", "short": "aya-expanse-8b", "head": (15, 0)},
    {"model_id": "meta-llama/Meta-Llama-3-8B", "short": "llama-3-8b", "head": (14, 25)},
]
LANGS = ["eng", "ger", "fre", "rus", "zho", "ita", "ned", "zho", "vie"]
DATAPATH = "data/noun-adj.csv"
CACHE_DIR = "/scratch/msonkin/word-order-thesis/cache/"

def create_parallel_dataset(src_lang, tgt_lang, df, model_id, sentences_src_prefix, random_seed=None):
    parallel_dataset = ParallelDataset(
        model_id,
        dataframe=df,
        lang_src=src_lang,
        lang_tgt=tgt_lang,
        sentences_src_prefix=sentences_src_prefix,
        sentences_tgt_prefix=sentences_src_prefix,
        random_seed=random_seed,
    )
    return parallel_dataset

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    df = pd.from_csv(DATAPATH)
    for model_dict in MODELS:
        model_id = model_dict['model_id']
        model_short = model_dict['short']
        head = model_dict['head']
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        with tqdm(total=len(LANGS)**2) as pbar:
            for src_lang in tqdm(LANGS):
                for tgt_lang in tqdm(LANGS):
                    parallel_dataset = create_parallel_dataset(src_lang, tgt_lang, df)
                    sources = [source.to(device) for source in source_tokens]
                    mean_act = collect_mean_activation(
                        model,
                        tokenizer,
                        sources,
                        "head_attention_value_output",
                        layer_i=model_dict['head'][0],
                        head_i=model_dict['head'][1],
                        unit="h.pos"
                    )

                    filename = f"output/activations/{model_short}_{src_lang}-{tgt_lang}_layer{head[0]}_head{head[1]}.pt"
                    torch.save(mean_act, filename)

                    pbar.update(1)

if __name__ == "__main__":
    main()