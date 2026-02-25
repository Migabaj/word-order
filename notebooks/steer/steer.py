import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from create_datasets.parallel_dataset import ParallelDataset
from prompting.intervene import collect_mean_activation, steer
from collect_mean_activations import create_parallel_dataset

# static parameters
MODELS = [
    # {"model_id": "ai-forever/mGPT", "short": "mgpt", "head": (11, 2)},
    {"model_id": "CohereLabs/aya-expanse-8b", "short": "aya-expanse-8b", "head": (15, 0)},
    # {"model_id": "meta-llama/Meta-Llama-3-8B", "short": "llama-3-8b", "head": (14, 25)},
]
LANGS = ["ger", "fre", "rus", "zho", "ita", "ned", "vie"]
POS_PREFIXES = ["noun", "adj"]
DATAPATH = "data/noun-adj.csv"
CACHE_DIR = "/scratch/msonkin/word-order-thesis/cache/"
ONESHOT_TEMPLATE = "{lang_src}: \"{sentence_src}\" - {lang_tgt}: \"{sentence_tgt}\""
LAST_TEMPLATE = "{lang_src}: \"{sentence_src}\" - {lang_tgt}: \""
NUM_SHOTS = 1
HF_TOKEN = "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"

# parameters for token prediction (with vs. without space)
start_with_space_base = False
start_with_space_plant = False

# set up cuda
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
# set up softmax conversion
sm = torch.nn.Softmax(dim=-1)

def make_parallel_dataset(
    filepath: str,
    model_id: str,
    src_lang: str,
    tgt_lang: str,
    sentences_src_prefix: str = "phrase",
    sentences_tgt_prefix: str = "phrase",
    sample_size: int = 199,
    random_seed: int = 42
    ):
    df = pd.read_csv(filepath)
    base_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    base_dataset = ParallelDataset(
        model_id,
        dataframe=base_df,
        lang_src=src_lang,
        lang_tgt=tgt_lang,
        sentences_src_prefix=sentences_src_prefix,
        sentences_tgt_prefix=sentences_tgt_prefix,
        random_seed=random_seed
    )
    return base_dataset

def run_experiment(
    model,
    tokenizer,
    base_dataset,
    mean_act_source,
    tgt_lang,
    tgt_lang_plant,
    layer_i,
    head_i,
    save_path,
    shot_data_src_prefix = "phrase",
    shot_data_tgt_prefix = "phrase",
    ):
    # set up data
    data = []
    #set up prompts
    base_prompts = base_dataset.format(
        ONESHOT_TEMPLATE,
        shots=NUM_SHOTS,
        last_prompt_template=LAST_TEMPLATE,
        shot_data_src_prefix=shot_data_src_prefix,
        shot_data_tgt_prefix=shot_data_tgt_prefix,
        shuffle_shots=False,
    )
    for row_i, row in base_dataset.df.iterrows():
        tokentype2token = {}
        for S in POS_PREFIXES:
            token_text = row[f'{S}-{tgt_lang}']
            # token_text_plant = row[f'{S}-{tgt_lang_plant}']
            if start_with_space_base:
                token_text = " "+token_text
            # if start_with_space_plant:
            #     token_text_plant = " "+token_text_plant
            tokentype2token[f"{S}-{tgt_lang}"] = tokenizer(token_text, add_special_tokens=False).input_ids[0]
            # tokentype2token[f"{S}-{tgt_lang_plant}"] = tokenizer(token_text_plant).input_ids[0]
        
        prompt_base = tokenizer(base_prompts[row_i], return_tensors="pt").to(device)
        base_out, cf_out = steer(model, tokenizer, prompt_base, mean_act_source, "head_attention_value_output", layer_i=layer_i, head_i=head_i)
        base_logits = base_out.logits[0, -1]
        plant_logits = cf_out.logits[0, -1]
        base_probs = sm(base_logits[[tok for _, tok in sorted(tokentype2token.items())]])
        plant_probs = sm(plant_logits[[tok for _, tok in sorted(tokentype2token.items())]])
        for token_type_i, (token_type, token) in enumerate(sorted(tokentype2token.items())):
            data.append(
                {
                    "sentence_id": row_i,
                    "token_type": token_type,
                    "token": tokenizer.decode(token),
                    "prob_base": base_probs[token_type_i].item(),
                    "prob_plant": plant_probs[token_type_i].item(),
                    "type": "head_attention_value_output",
                }
            )
    df = pd.DataFrame(data)
    df.to_csv(save_path)

def main():
    login(HF_TOKEN)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    for model_dict in MODELS:

        # set up variables
        model_id = model_dict['model_id']
        model_short = model_dict['short']
        layer, head = model_dict['head']

        # set up model
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)

        # hardcode source languages
        src_lang = "eng"
        src_lang_plant = "eng"

        for tgt_lang in tqdm(LANGS):
            # check sample size
            if any([lang in [src_lang, tgt_lang] for lang in ["vie", "zho"]]):
                sample_size = 50
            else:
                sample_size = 199

            # define database
            dataset_base = make_parallel_dataset(DATAPATH, model_id, src_lang, tgt_lang, sample_size=sample_size)

            # for different plant settings
            for tgt_lang_plant in LANGS:

                # skip pair of identical tgt languages
                if tgt_lang == tgt_lang_plant:
                    continue
                
                # set up variables
                save_path = f"output/steer/probs/noun-adj_{model_short}_{src_lang}-{tgt_lang}_{src_lang_plant}-{tgt_lang_plant}_layer{layer}_head{head}.csv"
                mean_act_filepath = f"output/activations/{model_short}_{src_lang_plant}-{tgt_lang_plant}_layer{layer}_head{head}.pt"
                mean_act = torch.load(mean_act_filepath)

                # run experiment
                run_experiment(model, tokenizer, dataset_base, mean_act, tgt_lang, tgt_lang_plant, layer, head, save_path)
    
if __name__ == "__main__":
    main()