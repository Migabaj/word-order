import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from create_datasets.parallel_dataset import ParallelDataset, create_parallel_dataset
from prompting.intervene import collect_mean_activation, steer
from collect_mean_activations import create_parallel_dataset
from utils.model_args import model_to_short, model_to_nounadj_head

def get_args():
    parser = argparse.ArgumentParser(description="Steering experiment runner")
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model ID from HuggingFace"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for attention head"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Head index for attention head"
    )
    
    # Language arguments
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=["eng", "ger", "fre", "pol"],
        help="Target languages to evaluate"
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="eng",
        help="Source language"
    )
    
    # Data arguments
    parser.add_argument(
        "--datapath",
        type=str,
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/msonkin/word-order-thesis/cache/",
        help="HuggingFace cache directory"
    )
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
        default="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",
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--start-with-space",
        type=bool,
        default=True,
        help="Whether to prepend space to tokens"
    )
    
    args = vars(parser.parse_args())
    return args

def variable_setup():
    global device, sm, data
    # set up cuda
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    # set up softmax conversion
    sm = torch.nn.Softmax(dim=-1)
    # set up data
    data = []

def get_tokentype2token(tokenizer, df_row, tgt_lang, pos_prefixes, start_with_space):
    tokentype2token = {}
    for S in pos_prefixes:
        token_text = df_row[f'{S}-{tgt_lang}']
        if start_with_space:
            token_text = " "+token_text
        tokentype2token[f"{S}-{tgt_lang}"] = tokenizer(token_text, add_special_tokens=False).input_ids[0]
    return tokentype2token

def get_base_and_plant_probs(base_output, plant_output, tokentype2token):
    base_logits = base_output.logits[0, -1]
    plant_logits = plant_output.logits[0, -1]
    base_probs = sm(base_logits[[tok for _, tok in sorted(tokentype2token.items())]])
    plant_probs = sm(plant_logits[[tok for _, tok in sorted(tokentype2token.items())]])
    return base_probs, plant_probs

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
    pos_prefixes=["verb", "obj"],
    shot_data_src_prefix="phrase",
    shot_data_tgt_prefix="phrase",
    start_with_space=True,
    oneshot_template=None,
    last_template=None,
    num_shots=1,
    ):
    # set up variables
    variable_setup()

    #set up prompts
    base_prompts = base_dataset.format(
        oneshot_template,
        shots=num_shots,
        last_prompt_template=last_template,
        shot_data_src_prefix=shot_data_src_prefix,
        shot_data_tgt_prefix=shot_data_tgt_prefix,
        shuffle_shots=False,
    )
    print("Example of base prompt:")
    print(base_prompts[0])

    for row_i, row in base_dataset.df.iterrows():
        tokentype2token = get_tokentype2token(tokenizer, row, tgt_lang, pos_prefixes, start_with_space)

        prompt_base = tokenizer(base_prompts[row_i], return_tensors="pt").to(device)
        base_out, cf_out = steer(model, tokenizer, prompt_base, mean_act_source, "head_attention_value_output", layer_i=layer_i, head_i=head_i)
        base_probs, plant_probs = get_base_and_plant_probs(base_out, cf_out, tokentype2token)

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
    args = get_args()
    
    login(args["hf_token"])
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Extract arguments
    model_id = args["model_id"]
    langs = args["langs"]
    src_lang = args["src_lang"]
    datapath = args["datapath"]
    cache_dir = args["cache_dir"]
    sample_size = args["sample_size"]
    pos_prefixes = args["pos_prefixes"]
    oneshot_template = args["oneshot_template"]
    last_template = args["last_template"]
    num_shots = args["num_shots"]
    start_with_space = args["start_with_space"]

    model_short = model_to_short[model_id]
    layer = args["layer"] if args["layer"] is not None else model_to_nounadj_head[model_id][0]
    head = args["head"] if args["head"] is not None else model_to_nounadj_head[model_id][1]

    experiment = os.path.splitext(os.path.split(datapath)[-1])[0]

    # set up model
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # hardcode source languages
    src_lang_plant = src_lang

    for tgt_lang in tqdm(langs):
        
        df = pd.read_csv(datapath)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # define database
        dataset_base = create_parallel_dataset(
            model_id,
            df,
            src_lang,
            tgt_lang,
            "phrase",
            "phrase_cutoff_after_subject",
            oneshot_template=oneshot_template,
            last_prompt_template=last_template,
            num_shots=num_shots,
            sample_size=sample_size,
            random_seed=42,
        )

        # for different plant settings
        for tgt_lang_plant in langs:
            # set up variables
            save_path = f"output/steer/probs/{experiment}_{model_short}_{src_lang}-{tgt_lang}_{src_lang_plant}-{tgt_lang_plant}_layer{layer}_head{head}.csv"
            mean_act_filepath = f"output/activations/{experiment}_{model_short}_{src_lang_plant}-{tgt_lang_plant}_layer{layer}_head{head}.pt"
            mean_act = torch.load(mean_act_filepath)

            # run experiment
            run_experiment(
                model, 
                tokenizer, 
                dataset_base, 
                mean_act, 
                tgt_lang, 
                tgt_lang_plant, 
                layer, 
                head, 
                save_path,
                pos_prefixes=pos_prefixes,
                oneshot_template=oneshot_template,
                last_template=last_template,
                num_shots=num_shots,
                start_with_space=start_with_space,
            )
    
if __name__ == "__main__":
    main()