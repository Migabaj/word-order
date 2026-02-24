# This script generates a YAML file for every combination of model and language settings, and runs the notebook script for each.
import os
import yaml
import itertools
import subprocess
import argparse

models = [
    {"model_id": "ai-forever/mGPT", "short": "mgpt"},
    {"model_id": "CohereLabs/aya-expanse-8b", "short": "aya-expanse-8b"},
    {"model_id": "meta-llama/Meta-Llama-3-8B", "short": "llama-3-8b"},
]

tgt2s = {
    "eng": "adj",
    "ger": "adj",
    "rus": "adj",
    "ned": "adj",
    "zho": "adj",
    "fre": "noun",
    "ita": "noun",
    "vie": "noun"
}

src_src = [
    {"src_lang_base": "eng", "src_lang_source": "eng"},
    {"src_lang_base": "ger", "src_lang_source": "ger"},
    {"src_lang_base": "fre", "src_lang_source": "fre"},
    {"src_lang_base": "zho", "src_lang_source": "zho"},
    {"src_lang_base": "eng", "src_lang_source": "ger"},
    {"src_lang_base": "ger", "src_lang_source": "eng"},
    {"src_lang_base": "eng", "src_lang_source": "fre"},
    {"src_lang_base": "fre", "src_lang_source": "eng"},
    {"src_lang_base": "ger", "src_lang_source": "fre"},
    {"src_lang_base": "fre", "src_lang_source": "ger"},
    {"src_lang_base": "zho", "src_lang_source": "ned"},
    {"src_lang_base": "ned", "src_lang_source": "zho"},
    {"src_lang_base": "zho", "src_lang_source": "vie"},
    {"src_lang_base": "vie", "src_lang_source": "zho"},
]

tgt_tgt = [
    {"tgt_lang_base": "fre", "tgt_lang_source": "ger"},
    {"tgt_lang_base": "ger", "tgt_lang_source": "fre"},
    {"tgt_lang_base": "ita", "tgt_lang_source": "ger"},
    {"tgt_lang_base": "ger", "tgt_lang_source": "ita"},
    {"tgt_lang_base": "fre", "tgt_lang_source": "zho"},
    {"tgt_lang_base": "zho", "tgt_lang_source": "fre"},
    {"tgt_lang_base": "ger", "tgt_lang_source": "vie"},
    {"tgt_lang_base": "vie", "tgt_lang_source": "ger"},
    {"tgt_lang_base": "rus", "tgt_lang_source": "fre"},
    {"tgt_lang_base": "fre", "tgt_lang_source": "rus"},
    {"tgt_lang_base": "rus", "tgt_lang_source": "vie"},
    {"tgt_lang_base": "vie", "tgt_lang_source": "rus"},
]



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

def main(postfix="", template_path="notebooks/intervention/params/template.yaml"):
    print("HELLO!")
    base_yaml = get_base_yaml(template_path=template_path)
    outdir = "notebooks/intervention/params/"
    os.makedirs(outdir, exist_ok=True)
    for model in models:
        for src_setting in src_src:
            for tgt_setting in tgt_tgt:
                yml = dict(base_yaml) if base_yaml else {}
                yml.update(src_setting)
                yml.update(tgt_setting)
                # Add/override any other required fields here
                yml["model_id"] = model["model_id"]
                yml['block_intervention_save_path'] = f"output/intervention/probs/block_noun-adj_{model['short']}_{src_setting['src_lang_base']}-{tgt_setting['tgt_lang_base']}_{src_setting['src_lang_source']}-{tgt_setting['tgt_lang_source']}{postfix}.csv"
                if os.path.exists(yml['block_intervention_save_path']):
                    subprocess.run(["echo", f"File {yml['block_intervention_save_path']} already exists, skipping."])
                    # print(f"File {yml['block_intervention_save_path']} already exists, skipping.")
                    continue
                if src_setting['src_lang_source'] == tgt_setting['tgt_lang_source'] or src_setting['src_lang_base'] == tgt_setting['tgt_lang_base']:
                    subprocess.run(["echo", f"Skipping incompatible setting: {src_setting} -> {tgt_setting}"])
                    continue

                yml['head_intervention_save_path'] = f"output/intervention/probs/head_noun-adj_{model['short']}_{src_setting['src_lang_base']}-{tgt_setting['tgt_lang_base']}_{src_setting['src_lang_source']}-{tgt_setting['tgt_lang_source']}{postfix}.csv"
                yml['load_data'] = False
                yml['num_shots'] = 1
                yml['data_path'] = "data/noun-adj.csv"
                yml['sample_size'] = 50 if any([lang in (list(tgt_setting.values()) + list(src_setting.values())) for lang in ["vie", "zho"]]) else 199
                yml['random_seed'] = 42
                # Save YAML
                fname = make_filename(src_setting, tgt_setting, model['short'], postfix)
                fpath = os.path.join(outdir, fname)
                with open(fpath, 'w') as f:
                    yaml.dump(yml, f, sort_keys=False)
                # Run script
                cmd = [
                    "python3", "-m", "src.utils.run_notebook.py",
                    "--input", "notebooks/intervention/intervene.ipynb",
                    "--output", "notebooks/intervention/intervene.ipynb",
                    "--params-path", fpath
                ]
                echo = [
                    "echo", f"Running notebook with params from {fpath}"
                ]
                subprocess.run(echo)
                subprocess.run(cmd)

def test(template_path="notebooks/intervention/params/template.yaml", postfix="_test"):
    """
    Test function that runs a single small example.
    """
    print("Running test with one example...")
    base_yaml = get_base_yaml(template_path=template_path)
    outdir = "notebooks/intervention/params/"
    os.makedirs(outdir, exist_ok=True)
    
    # Use just one model, one source setting, and one target setting
    model = models[0]
    src_setting = src_src[0]
    tgt_setting = tgt_tgt[0]
    
    yml = dict(base_yaml) if base_yaml else {}
    yml.update(src_setting)
    yml.update(tgt_setting)
    yml["model_id"] = model["model_id"]
    yml['block_intervention_save_path'] = f"output/intervention/probs/block_noun-adj_{model['short']}_{src_setting['src_lang_base']}-{tgt_setting['tgt_lang_base']}_{src_setting['src_lang_source']}-{tgt_setting['tgt_lang_source']}_test.csv"
    yml['head_intervention_save_path'] = f"output/intervention/probs/head_noun-adj_{model['short']}_{src_setting['src_lang_base']}-{tgt_setting['tgt_lang_base']}_{src_setting['src_lang_source']}-{tgt_setting['tgt_lang_source']}_test.csv"
    yml['load_data'] = False
    yml['num_shots'] = 1
    yml['data_path'] = "data/noun-adj.csv"
    yml['sample_size'] = 5  # Small sample for testing
    yml['random_seed'] = 42
    
    # Save YAML
    fname = make_filename(src_setting, tgt_setting, model['short'], postfix=postfix)
    fpath = os.path.join(outdir, fname)
    with open(fpath, 'w') as f:
        yaml.dump(yml, f, sort_keys=False)
    
    print(f"Test YAML file saved to {fpath}")
    
    # Run script
    cmd = [
        "python3", "-m", "src.utils.run_notebook.py",
        "--input", "notebooks/intervention/intervene.ipynb",
        "--output", "notebooks/intervention/intervene.ipynb",
        "--params-path", fpath
    ]
    print(f"Running notebook with params from {fpath}")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML configs and run notebook interventions")
    parser.add_argument("--postfix", type=str, default="", help="Postfix to append to filenames")
    parser.add_argument("--template_path", type=str, default="notebooks/intervention/params/template.yaml", help="Path to template YAML file")
    parser.add_argument("--test", action="store_true", help="Run test with a single small example")
    args = parser.parse_args()
    
    if args.test:
        test(postfix=args.postfix, template_path=args.template_path)
    else:
        main(postfix=args.postfix, template_path=args.template_path)
