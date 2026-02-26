import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from create_datasets.parallel_dataset import ParallelDataset
from prompting.intervene import collect_mean_activation

MODELS = [
    {"model_id": "ai-forever/mGPT", "short": "mgpt", "head": (11, 2)},
    {"model_id": "CohereLabs/aya-expanse-8b", "short": "aya-expanse-8b", "head": (15, 0)},
    {"model_id": "meta-llama/Meta-Llama-3-8B", "short": "llama-3-8b", "head": (14, 25)},
]
LANGS = ["eng", "ger", "fre", "rus", "zho", "ita", "ned", "vie"]
DATAPATH = "data/noun-adj.csv"
CACHE_DIR = "/scratch/msonkin/word-order-thesis/cache/"
ONESHOT_TEMPLATE = "{lang_src}: \"{sentence_src}\" - {lang_tgt}: \"{sentence_tgt}\""
LAST_TEMPLATE = "{lang_src}: \"{sentence_src}\" - {lang_tgt}: \""
NUM_SHOTS = 1
HF_TOKEN = "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"

def create_parallel_dataset(src_lang, tgt_lang, df, model_id, sentences_prefix, random_seed=None):
    parallel_dataset = ParallelDataset(
        model_id,
        dataframe=df,
        lang_src=src_lang,
        lang_tgt=tgt_lang,
        sentences_src_prefix=sentences_prefix,
        sentences_tgt_prefix=sentences_prefix,
        random_seed=random_seed,
    )
    return parallel_dataset

def main():
    login(HF_TOKEN)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataframe = pd.read_csv(DATAPATH)
    print(dataframe.head())

    for model_dict in MODELS:
        model_id = model_dict['model_id']
        model_short = model_dict['short']
        head = model_dict['head']
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        with tqdm(total=len(LANGS)**2) as pbar:
            for src_lang in LANGS:
                for tgt_lang in LANGS:
                    if any(l in ["zho", "vie"] for l in [src_lang, tgt_lang]):
                        df = dataframe.sample(n=50, random_state=42).reset_index(drop=True)
                    else:
                        df = dataframe

                    parallel_dataset = create_parallel_dataset(src_lang, tgt_lang, df, model_id, "phrase", random_seed=42)
                    prompts = parallel_dataset.format(
                        ONESHOT_TEMPLATE,
                        shots=NUM_SHOTS,
                        last_prompt_template=LAST_TEMPLATE,
                        shot_data_src=f"phrase-{src_lang}",
                        shot_data_tgt=f"phrase-{tgt_lang}",
                        shuffle_shots=False,
                    )
                    print(prompts[0])
                    prompts = parallel_dataset.prompts_to_tokens()
                    prompts = [tokens.to(device) for tokens in prompts]
                    mean_act = collect_mean_activation(
                        model,
                        tokenizer,
                        prompts,
                        "head_attention_value_output",
                        layer_i=model_dict['head'][0],
                        head_i=model_dict['head'][1],
                        unit="h.pos"
                    )

                    filename = f"output/activations/{model_short}_{src_lang}-{tgt_lang}_layer{head[0]}_head{head[1]}.pt"
                    torch.save(mean_act, filename)

                    torch.cuda.empty_cache()
                    pbar.update(1)

if __name__ == "__main__":
    main()