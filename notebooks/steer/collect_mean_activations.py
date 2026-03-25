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

    parser.add_argument(
        "--save-path",
        type=str,
        default="/scratch/msonkin/word-order-thesis/activations/mean_act.pt",
        help="Path to save mean activations"
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
    
    args = vars(parser.parse_args())
    return args

def main():
    print("Hello! This is the steering experiment runner.")
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
    save_path = args["save_path"]

    sentence_src_prefix = args["sentences_src_prefix"]
    sentence_tgt_prefix = args["sentences_tgt_prefix"]
    shot_data_src_prefix = args["shot_data_src_prefix"]
    shot_data_tgt_prefix = args["shot_data_tgt_prefix"]

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
                print(f"Processing {src_lang} -> {tgt_lang}...", flush=True)
                if os.path.exists(save_path.format(model_short=model_short, src_lang=src_lang, tgt_lang=tgt_lang, layer=layer, head=head)):
                    print(f"Already have activations for {src_lang} -> {tgt_lang}, skipping...")
                    pbar.update(1)
                    continue
                parallel_dataset, parallel_prompts = create_parallel_dataset(
                    model_id,
                    dataframe,
                    src_lang,
                    tgt_lang,
                    sentence_src_prefix,
                    sentence_tgt_prefix,
                    oneshot_template=oneshot_template,
                    last_prompt_template=last_template,
                    num_shots=num_shots,
                    sample_size=sample_size,
                    random_seed=42,
                    return_prompts=True,
                    shot_data_src_prefix=shot_data_src_prefix,
                    shot_data_tgt_prefix=shot_data_tgt_prefix,
                    )
                print(parallel_prompts[0])
                prompts = parallel_dataset.prompts_to_tokens()
                prompts = [tokens.to(device) for tokens in prompts]
                # print(prompts[0])
                # print(tokenizer.decode(prompts[0]['input_ids'][0]))
                mean_act = collect_mean_activation(
                    model,
                    tokenizer,
                    prompts,
                    "head_attention_value_output",
                    layer_i=layer,
                    head_i=head,
                    unit="h.pos"
                )
                torch.save(mean_act, save_path.format(model_short=model_short, src_lang=src_lang, tgt_lang=tgt_lang, layer=layer, head=head))

                torch.cuda.empty_cache()
                pbar.update(1)

if __name__ == "__main__":
    main()