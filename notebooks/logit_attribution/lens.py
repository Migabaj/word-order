import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from create_datasets.parallel_dataset import ParallelDataset, create_parallel_dataset
from modeling.wrapper import load_mgpt, load_llama, load_gptneo, load_eurollm, load_aya_expanse, GPTWrapper, LLamaWrapper, EuroLLMWrapper, AyaExpanseWrapper
from utils.model_args import model_to_short
from utils.langs import start_with_space

def get_args():
    parser = argparse.ArgumentParser(description="Logit Lens experiment runner")

    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="sberbank-ai/mGPT",
        help="Model ID from HuggingFace"
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
    # parser.add_argument(
    #     "--start-with-space-tgt",
    #     type=bool,
    #     default=True,
    #     help="Whether to prepend space to tokens in target language"
    # )
    # parser.add_argument(
    #     "--start-with-space-src",
    #     type=bool,
    #     default=True,
    #     help="Whether to prepend space to tokens in source language"
    # )
    parser.add_argument(
        "--shot-data-src-prefix",
        type=str,
        default="phrase",
        help="Prefix for source sentences in few-shot examples"
    )
    parser.add_argument(
        "--shot-data-tgt-prefix",
        type=str,
        default="phrase",
        help="Prefix for target sentences in few-shot examples"
    )

    args = vars(parser.parse_args())
    return args

def get_model_type(model_id):
    if "mGPT" in model_id:
        return "mgpt"
    elif "Llama" in model_id or "llama" in model_id:
        return "llama"
    elif "gpt-neo" in model_id:
        return "gptneo"
    elif "EuroLLM" in model_id:
        return "eurollm"
    elif "aya-expanse" in model_id.lower():
        return "aya_expanse"
    else:
        raise ValueError(f"Unsupported model: {model_id}")

def load_model_and_wrapper(model_id, cache_dir):
    model_type = get_model_type(model_id)
    if model_type == "mgpt":
        model, tokenizer = load_mgpt(cache_dir=cache_dir)
        wrapper = GPTWrapper(model, tokenizer)
    elif model_type == "llama":
        model, tokenizer = load_llama(cache_dir=cache_dir)
        wrapper = LLamaWrapper(model, tokenizer)
    elif model_type == "gptneo":
        model, tokenizer = load_gptneo(cache_dir=cache_dir)
        wrapper = GPTWrapper(model, tokenizer)  # Assuming GPTWrapper works for gpt-neo
    elif model_type == "eurollm":
        model, tokenizer = load_eurollm(cache_dir=cache_dir)
        wrapper = EuroLLMWrapper(model, tokenizer)
    elif model_type == "aya_expanse":
        model, tokenizer = load_aya_expanse(cache_dir=cache_dir)
        wrapper = AyaExpanseWrapper(model, tokenizer)
    else:
        raise ValueError(f"No wrapper for model type: {model_type}")
    return wrapper

def get_tokentype2token(tokenizer, df_row, src_lang, tgt_lang, pos_prefixes, start_with_space_tgt, start_with_space_src):
    tokentype2token = {}
    for lang in [tgt_lang, src_lang, "eng"]:
        for S in pos_prefixes:
            token_text = df_row[f'{S}-{lang}']
            if start_with_space[lang]:
                token_text = " " + token_text
            tokentype2token[f"{S}-{lang}"] = tokenizer(token_text, add_special_tokens=False).input_ids[0]
    return tokentype2token

def run_experiment(
    wrapper,
    base_dataset,
    src_lang,
    tgt_lang,
    save_path,
    pos_prefixes=["verb", "obj"],
    shot_data_src_prefix="phrase",
    shot_data_tgt_prefix="phrase",
    start_with_space_tgt=True,
    start_with_space_src=True,
    oneshot_template=None,
    last_template=None,
    num_shots=1,
):
    data = []

    # Set up prompts
    base_prompts = base_dataset.format(
        oneshot_template,
        shots=num_shots,
        last_prompt_template=last_template,
        shot_data_src_prefix=shot_data_src_prefix,
        shot_data_tgt_prefix=shot_data_tgt_prefix,
        shuffle_shots=False,
    )
    print("Example of base prompt:", flush=True)
    print(base_prompts[0], flush=True)

    for row_i, row in base_dataset.df.iterrows():
        tokentype2token = get_tokentype2token(wrapper.tokenizer, row, src_lang, tgt_lang, pos_prefixes, start_with_space_tgt, start_with_space_src)

        prompt_ids = wrapper.tokenizer(base_prompts[row_i], return_tensors="pt").input_ids.to(wrapper.device)

        with torch.no_grad():
            out = wrapper.model(input_ids=prompt_ids, output_hidden_states=True)
            logits = torch.stack(wrapper.layer_decode(out.hidden_states)).squeeze()[1:]  # Skip embedding layer
            probs = wrapper.get_probs_per_layer(logits, dtype=torch.float32)

            for layer_i in range(len(probs)):
                layer_probs = probs[layer_i]
                layer_logits = logits[layer_i]
                for token_type_i, (token_type, token) in enumerate(sorted(tokentype2token.items())):
                    data.append(
                        {
                            "sentence_id": row_i,
                            "layer": layer_i,
                            "token_type": token_type,
                            "token": wrapper.tokenizer.decode(token),
                            "prob": layer_probs[token].item(),
                            "logit": layer_logits[token].item(),
                            "type": "logit_lens",
                        }
                    )
            del out, logits, probs
            torch.cuda.memory.empty_cache()
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

def main():
    args = get_args()

    login(args["hf_token"])

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
    # start_with_space_base = args["start_with_space_base"]

    sentence_src_prefix = args["sentences_src_prefix"]
    sentence_tgt_prefix = args["sentences_tgt_prefix"]
    shot_data_src_prefix = args["shot_data_src_prefix"]
    shot_data_tgt_prefix = args["shot_data_tgt_prefix"]

    model_short = model_to_short[model_id]
    experiment = os.path.splitext(os.path.split(datapath)[-1])[0]

    # Set up model
    wrapper = load_model_and_wrapper(model_id, cache_dir)

    for tgt_lang in tqdm(langs):
        df = pd.read_csv(datapath)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Define dataset
        dataset_base = create_parallel_dataset(
            model_id,
            df,
            src_lang,
            tgt_lang,
            sentence_src_prefix,
            sentence_tgt_prefix,
            oneshot_template=oneshot_template,
            last_prompt_template=last_template,
            num_shots=num_shots,
            sample_size=sample_size,
            random_seed=42,
        )

        save_path = f"output/logit_lens/probs/{experiment}_{model_short}_{src_lang}-{tgt_lang}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            print(f"Already have results for {src_lang} -> {tgt_lang}, skipping...", flush=True)
            continue

        start_with_space_tgt = start_with_space[tgt_lang]
        start_with_space_src = start_with_space[src_lang]

        # Run experiment
        run_experiment(
            wrapper,
            dataset_base,
            src_lang,
            tgt_lang,
            save_path,
            pos_prefixes=pos_prefixes,
            oneshot_template=oneshot_template,
            last_template=last_template,
            num_shots=num_shots,
            shot_data_src_prefix=shot_data_src_prefix,
            shot_data_tgt_prefix=shot_data_tgt_prefix,
            start_with_space_tgt=start_with_space_tgt,
            start_with_space_src=start_with_space_src,
        )

if __name__ == "__main__":
    main()