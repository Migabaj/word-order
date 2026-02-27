import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from create_datasets.parallel_dataset import ParallelDataset, create_parallel_dataset
from prompting.intervene import collect_mean_activation
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
    
    args = vars(parser.parse_args())
    return args

def main():
    args = get_args()

    login(args["hf_token"])
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Extract arguments
    model_id = args["model_id"]
    langs = args["langs"]
    datapath = args["datapath"]
    cache_dir = args["cache_dir"]
    sample_size = args["sample_size"]
    oneshot_template = args["oneshot_template"]
    last_template = args["last_template"]
    num_shots = args["num_shots"]

    dataframe = pd.read_csv(datapath)

    model_short = model_to_short[model_id]
    layer = args["layer"] if args["layer"] is not None else model_to_nounadj_head[model_id][0]
    head = args["head"] if args["head"] is not None else model_to_nounadj_head[model_id][1]

    experiment = os.path.splitext(os.path.split(datapath)[-1])[0]

    # set up model
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    with tqdm(total=len(langs)**2) as pbar:
        for src_lang in langs:
            for tgt_lang in langs:
                parallel_dataset = create_parallel_dataset(
                    model_id,
                    dataframe,
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
                prompts = parallel_dataset.prompts_to_tokens()
                prompts = [tokens.to(device) for tokens in prompts]
                print(prompts[0])
                print(tokenizer.decode(prompts[0]['input_ids'][0]))
                mean_act = collect_mean_activation(
                    model,
                    tokenizer,
                    prompts,
                    "head_attention_value_output",
                    layer_i=layer,
                    head_i=head,
                    unit="h.pos"
                )

                filename = f"output/activations/{experiment}_{model_short}_{src_lang}-{tgt_lang}_layer{layer}_head{head}.pt"
                torch.save(mean_act, filename)

                torch.cuda.empty_cache()
                pbar.update(1)

if __name__ == "__main__":
    main()