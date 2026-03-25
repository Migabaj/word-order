# This script generates a YAML file for every combination of model and language settings, and runs the notebook script for each.
import os
import yaml
import torch
import argparse
import itertools
import subprocess
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from pyvene.models.modeling_utils import getattr_for_torch_module

from create_datasets.parallel_dataset import ParallelDataset, create_parallel_dataset
from prompting.intervene import collect_mean_activation, steer
from utils.model_args import model_to_short, model_to_nounadj_head
from utils.model_args import model_to_num_layers_attr, model_to_num_heads_attr
from utils.langs import start_with_space
from prompting.intervene import intervention_config, intervention_data

def get_args():
    parser = argparse.ArgumentParser(description="Steering experiment runner")
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model ID from HuggingFace"
    )

    
    # Intervention type
    parser.add_argument(
        "--component",
        type=str,
        default="block_output",
        help="Component to intervene on"
    )
    
    # Language arguments
    parser.add_argument(
        "--src-lang-base",
        type=str,
        default=None,
        help="Source language (Base)"
    )
    parser.add_argument(
        "--tgt-lang-base",
        type=str,
        default=None,
        help="Target language (Base)"
    )
    parser.add_argument(
        "--src-lang-plant",
        type=str,
        default=None,
        help="Source language (Plant)"
    )
    parser.add_argument(
        "--tgt-lang-plant",
        type=str,
        default=None,
        help="Target language (Plant)"
    )
    
    # Data
    parser.add_argument(
        "--datapath",
        type=str,
        help="Path to dataset CSV"
    )

    # Cache
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/msonkin/word-order-thesis/cache/",
        help="HuggingFace cache directory"
    )

    # Sample size
    parser.add_argument(
        "--sample-size",
        type=int,
        default=199,
        help="Number of samples to use from dataset"
    )
    
    # POS tags
    parser.add_argument(
        "--pos-prefixes",
        type=str,
        nargs="+",
        help="POS tag prefixes to use"
    )
    
    # Prompting arguments
    parser.add_argument(
        "--oneshot-template",
        type=str,
        default="{lang_src}: \"{sentence_src}\" - {lang_tgt}: \"{sentence_tgt}\"",
        help="One-shot prompt template"
    )
    parser.add_argument(
        "--last-template",
        type=str,
        default="{lang_src}: \"{sentence_src}\" - {lang_tgt}: \"{sentence_tgt}",
        help="Last prompt template (without closing quote)"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=1,
        help="Number of shots for few-shot learning"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--start-with-space-base",
        action="store_true",
        help="Whether to prepend space to tokens"
    )
    parser.add_argument(
        "--start-with-space-plant",
        action="store_true",
        help="Whether to prepend space to tokens"
    )
    
    parser.add_argument(
        "--sentences-src-prefix",
        type=str,
        default="phrase",
        help="Prefix for source sentences"
    )
    parser.add_argument(
        "--sentences-tgt-prefix",
        type=str,
        default="phrase",
        help="Prefix for target sentences"
    )
    parser.add_argument(
        "--shot-data-src-prefix",
        type=str,
        default="phrase",
        help="Prefix for source sentences in shot data"
    )
    parser.add_argument(
        "--shot-data-tgt-prefix",
        type=str,
        default="phrase",
        help="Prefix for target sentences in shot data"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save probabilities"
    )

    # edge case
    # TODO: clunky
    parser.add_argument(
        "--plant-datapath",
        type=str,
        default=None,
        help="Path for plant prompts. If None, samples the original database."
    )

    # language setup grid
    parser.add_argument(
        "--lang-setup-grid-path",
        type=str,
        default=None,
        help="Path to CSV file containing language setup grid. If provided, overrides individual language arguments"
    )

    parser.add_argument(
        "--check-for-filename",
        action="store_true",
        help="Whether to check for existing save path filename before running experiment"
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layers to intervene on (e.g. '0 1 2' or '0-2')"
    )

    args = vars(parser.parse_args())
    return args

def get_token2tokentype(tokenizer, row_i, df_base, df_plant, lang_base, lang_plant, pos_prefixes, start_with_space_base, start_with_space_plant):
    tokentype2token = {}
    for L in [lang_base, lang_plant]:
        for S in pos_prefixes:
            tokentype2token[f"{S}-{L}-base"] = df_base.iloc[row_i][f'{S}-{L}']
            tokentype2token[f"{S}-{L}-plant"] = df_plant.iloc[row_i][f'{S}-{L}']
    
    if start_with_space_base:
        for token_type, token in tokentype2token.items():
            if lang_base in token_type:
                tokentype2token[token_type] = " "+token
    if start_with_space_plant:
        for token_type, token in tokentype2token.items():
            if lang_plant in token_type:
                tokentype2token[token_type] = " "+token
    return tokentype2token

def get_base_yaml(template_path="notebooks/intervention/params/template.yaml"):
    # Use template.yaml as base
    with open(template_path) as f:
        base = yaml.safe_load(f)
    return base

def make_filename(src_setting, tgt_setting, model_short, postfix=""):
    # Example: noun-adj_aya-expanse-8b_eng-fre_eng-ger.yml
    # Example with postfix: noun-adj_aya-expanse-8b_eng-fre_eng-ger_test.yml
    postfix_str = f"_{postfix}" if postfix else ""
    return f"noun-adj_{model_short}_{src_setting['src_lang_base']}-{tgt_setting['tgt_lang_base']}_{src_setting['src_lang_source']}-{tgt_setting['tgt_lang_source']}{postfix_str}.yml"

def run_experiment_with_args(
    args,
    src_lang_base,
    tgt_lang_base,
    src_lang_plant,
    tgt_lang_plant,
    save_path=None
    ):
    # Extract arguments
    model_id = args["model_id"]
    datapath = args["datapath"]
    cache_dir = args["cache_dir"]
    sample_size = args["sample_size"]
    pos_prefixes = args["pos_prefixes"]
    oneshot_template = args["oneshot_template"]
    last_template = args["last_template"]
    num_shots = args["num_shots"]
    start_with_space_base = args["start_with_space_base"]
    start_with_space_plant = args["start_with_space_plant"]
    component = args["component"]
    shot_data_src_prefix = args["shot_data_src_prefix"]
    shot_data_tgt_prefix = args["shot_data_tgt_prefix"]

    layers = args["layers"]


    print(f"Target language (Base): {tgt_lang_base}, Target language (Plant): {tgt_lang_plant}")
    print(start_with_space_base, start_with_space_plant)

    check_for_filename = args["check_for_filename"]
    if check_for_filename and save_path is not None and os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping experiment.")
        return

    model_short = model_to_short[model_id]

    sentence_src_prefix = args["sentences_src_prefix"]
    sentence_tgt_prefix = args["sentences_tgt_prefix"]

    plant_datapath = args["plant_datapath"]
    lang_setup_grid_path = args["lang_setup_grid_path"]

    login(args["hf_token"])
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    df = pd.read_csv(datapath)
    df_base = df.sample(n=sample_size, random_state=42).reset_index(drop=True)


    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir,  attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model_class = model.__class__.__name__
    num_heads = getattr_for_torch_module(model, model_to_num_heads_attr[model_class])


    if plant_datapath is not None:
        df_plant = pd.read_csv(plant_datapath)
        df_plant = df_plant.sample(n=sample_size, random_state=43).reset_index(drop=True) # new dataset for plant
    else:
        df_plant = df_base.sample(n=sample_size, random_state=42).reset_index(drop=True) # shuffled base

    # Create datasets
    dataset_base, prompts_base = create_parallel_dataset(
        model_id,
        df=df_base,
        src_lang=src_lang_base,
        tgt_lang=tgt_lang_base,
        sentences_src_prefix=sentence_src_prefix,
        sentences_tgt_prefix=sentence_tgt_prefix,
        oneshot_template=oneshot_template,
        last_prompt_template=last_template,
        shot_data_src_prefix=shot_data_src_prefix,
        shot_data_tgt_prefix=shot_data_tgt_prefix,
        num_shots=num_shots,
        sample_size=sample_size,
        random_seed=42,
        return_prompts=True,
    )

    dataset_plant, prompts_plant = create_parallel_dataset(
        model_id,
        df=df_plant,
        src_lang=src_lang_plant,
        tgt_lang=tgt_lang_plant,
        sentences_src_prefix=sentence_src_prefix,
        sentences_tgt_prefix=sentence_tgt_prefix,
        oneshot_template=oneshot_template,
        last_prompt_template=last_template,
        shot_data_src_prefix=shot_data_src_prefix,
        shot_data_tgt_prefix=shot_data_tgt_prefix,
        num_shots=num_shots,
        sample_size=sample_size,
        random_seed=42,
        return_prompts=True,
    )
    print("Datasets created. Example prompts:")
    print(prompts_base[0])
    print()
    print(prompts_plant[0])
    print()


    data = []
    data_topk = []
    for row_i, row in tqdm(dataset_base.df.iterrows()):
        prompt_base = tokenizer(prompts_base[row_i], return_tensors="pt").to(device)
        prompt_plant = tokenizer(prompts_plant[row_i], return_tensors="pt").to(device)

        pos_base = prompt_base.input_ids.size(1) - 1
        pos_plant = prompt_plant.input_ids.size(1) - 1

        # token type to token dict
        tokentype2token = get_token2tokentype(
            tokenizer,
            row_i,
            df_base,
            df_plant,
            lang_base=tgt_lang_base,
            lang_plant=tgt_lang_plant,
            pos_prefixes=pos_prefixes,
            start_with_space_base=start_with_space_base,
            start_with_space_plant=start_with_space_plant
        )
        for head_i in range(num_heads):
            data, data_topk = intervention_data(
                model,
                tokenizer,
                prompt_base,
                prompt_plant, 
                pos_base,
                pos_plant,
                tokentype2token,
                sentence_index=row_i,
                head_i=head_i,
                component_type="head_attention_value_output",
                data=data,
                data_topk=data_topk,
                write_down_top_k=10,
                patch_layers=layers
            )
    df_probs = pd.DataFrame(data)
    df_probs.to_csv(save_path)

    df_topk = pd.DataFrame(data_topk)
    df_topk.to_csv(os.path.splitext(save_path)[0] + "_topk.csv")

def run_experiment():
    args = get_args()
    lang_setup_grid_path = args["lang_setup_grid_path"]

    if lang_setup_grid_path is not None:
        print(f"Running grid search over language setups from CSV file {lang_setup_grid_path}")
        lang_setup_df = pd.read_csv(lang_setup_grid_path)
        for lang_setup_i, lang_setup_row in lang_setup_df.iterrows():
            src_lang_base = lang_setup_row['src_lang_base']
            tgt_lang_base = lang_setup_row['tgt_lang_base']
            src_lang_plant = lang_setup_row['src_lang_plant']
            tgt_lang_plant = lang_setup_row['tgt_lang_plant']
            args["start_with_space_base"] = start_with_space[tgt_lang_base]
            args["start_with_space_plant"] = start_with_space[tgt_lang_plant]
            save_path = args["save_path"].format(src_lang_base=src_lang_base, tgt_lang_base=tgt_lang_base, src_lang_plant=src_lang_plant, tgt_lang_plant=tgt_lang_plant)
            print(f"Running language setup {lang_setup_i}: {src_lang_base}-{tgt_lang_base} (Base) and {src_lang_plant}-{tgt_lang_plant} (Plant)")
            run_experiment_with_args(args, src_lang_base, tgt_lang_base, src_lang_plant, tgt_lang_plant, save_path=save_path)
    else:
        src_lang_base = args["src_lang_base"]
        tgt_lang_base = args["tgt_lang_base"]
        src_lang_plant = args["src_lang_plant"]
        tgt_lang_plant = args["tgt_lang_plant"]
        save_path = args["save_path"]
        print(f"Running language setup {lang_setup_i}: {src_lang_base}-{tgt_lang_base} (Base) and {src_lang_plant}-{tgt_lang_plant} (Plant)")
        run_experiment_with_args(args, src_lang_base, tgt_lang_base, src_lang_plant, tgt_lang_plant, save_path=save_path)

def main():
    run_experiment()

if __name__ == "__main__":
    main()