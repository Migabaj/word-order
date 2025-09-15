import random
import pandas as pd
from typing import List, Tuple, Dict

from utils.model_args import parse_plain_args, ParseArg
from create_datasets.generate import format_phrase

def generate_oneshot_prompts(dataframe: pd.DataFrame,
    langs: Tuple[str, str, str, str],
    cols: Tuple[str, str, str, str],
    cutoff: int,
    mixed: bool = False) -> List[Dict[str, str]]:
    """Generates one-shot prompts for a given dataframe and languages."""
    prompts = []
    
    # create a row index pool
    if mixed:
        row_index_pool = set(range(len(dataframe)))
        print(row_index_pool)
    for i, row in dataframe.iterrows():
        prompt_source = ""
        prompt_base = ""
        if mixed:
            print(i)
            row_index_pool.remove(i)
            shot_index = random.sample(row_index_pool, 1)
            row_index_pool.remove(shot_index)
            row_index_pool.add(i)
        else:
            if i == 0:
                shot_index = len(dataframe.values) - 1
            else:
                shot_index = i - 1
        prompt_base = f"{langs[0]}: \"{row[langs[0]]}\" - {langs[1]}: \"{row[langs[1]]}\"\n" \
        f"{langs[0]}: \"{dataframe.iloc[shot_index][langs[0]]}\" - {langs[1]}: \"{' '.join(dataframe.iloc[shot_index][langs[1]].split()[:cutoff])}"
        prompt_source = f"{langs[2]}: \"{row[langs[2]]}\" - {langs[3]}: \"{row[langs[3]]}\"\n" \
        f"{langs[2]}: \"{dataframe.iloc[shot_index][langs[2]]}\" - {langs[3]}: \"{' '.join(dataframe.iloc[shot_index][langs[3]].split()[:cutoff])}"
        prompts.append({"base": prompt_base, "source": prompt_source})
    
    return prompts

def pick_n_rows(dataframe: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Randomly picks n rows from the dataframe."""
    if n > len(dataframe):
        raise ValueError("n cannot be greater than the number of rows in the dataframe.")
    df_sample = dataframe.sample(n=n, random_state=random_state)
    dataframe.drop(df_sample.index, inplace=True)
    return df_sample

def main():
    args = parse_plain_args(
        ParseArg("data", type=str, help="Path to the dataset containing sentences"),
        ParseArg("--instruction-template", type=str, help="Format for prompt"),
        ParseArg("--columns", type=str, nargs="+", help="Format for prompt"),
        ParseArg("--save-phrases", type=str, default="./phrases.csv", help="Path to generated phrases"),
        ParseArg("--seed", type=int, default=42, help="Random seed for reproducibility"),
        ParseArg("--prompt-template", default=None, type=str, help="Format for last prompt"),
    )
    df = pd.read_csv(args.data)
    prompts = []
    instruction_template = args.instruction_template
    prompt_template = args.prompt_template

    for row_index, row in df.iterrows():
        instruction = format_phrase(row, instruction_template)
        prompt = format_phrase(row, prompt_template)
        prompt_full = instruction+'\n'+prompt
        prompts.append({"prompt": prompt_full.strip(), **{col: row[col] for col in args.columns}})

    prompts_df = pd.DataFrame(prompts)
    print(prompts_df.head())
    prompts_df.to_csv(args.save_phrases, index=False)

if __name__ == "__main__":
    main()