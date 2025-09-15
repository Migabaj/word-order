import subprocess
import pandas as pd
from tqdm import tqdm

from utils.model_args import parse_plain_args, ParseArg

def inflection_table(row: pd.Series, col: str, features: str, lang: str) -> str:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "unimorph",
            "inflect",
            row[col],
            "--features",
            features,
            "-l",
            lang,
            "-q",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    inflections = result.stdout.decode("utf-8").strip()
    return inflections

def single_list_to_string(l):
    if len(l) == 1:
        return l[0]
    return l

def main():
    args = parse_plain_args(
        ParseArg("data", type=str, help="Path to the word dataset"),
        ParseArg(["--column", "-c"], type=str, help="Column of interest"),
        ParseArg("--new-column-name", type=str, help="Column name for new inflections"),
        ParseArg("--features", type=str, help="Features"),
        ParseArg("--lang", type=str, help="Language"),
        ParseArg("--df-path", type=str, default=None, help="Path for the extended dataframe")
    )

    df = pd.read_csv(args.data)
    col = args.column

    # What will become the column
    forms_column = []
    for i, row in (pbar := tqdm(df.iterrows())):
        pbar.set_description(f"Processing word {row[col]}")
        forms = [] # sometimes it is several forms for one word (e.g. hung, hanged)
        inflections = inflection_table(row, col, args.features, args.lang)
        for inflection in inflections.split('\n'):
            inflection_data = inflection.split() # three columns or empty
            print("inflection_data", inflection_data)
            if len(inflection_data) >= 3:
                form = inflection_data[1] # second column has the form
                forms.append(form)
        print("forms", forms)
        forms_column.append(forms)

    forms_series = pd.Series(forms_column)
    forms_series = forms_series.apply(single_list_to_string)
    df[args.new_column_name] = forms_series

    if args.df_path:
        df.to_csv(args.df_path, index=False)
    else:
        df.to_csv(args.data, index=False)

if __name__ == "__main__":
    main()