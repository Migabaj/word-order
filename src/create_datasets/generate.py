from typing import Tuple, List, Dict
import pandas as pd
import argparse

from utils.model_args import parse_plain_args, ParseArg
import re

def format_phrase(df_row: pd.core.series.Series, template: str) -> str:
    """Create a phrase, given a format and a dataframe row with corresponding variables."""
    keywords = df_row.to_dict()
    return template.format(**keywords)


def generate_phrase_groups(df: pd.DataFrame, templates: List[str]) -> List[List[str]]:
    """Create a list of phrases, given a dataset and the corresponding formats."""

    df_phrases = []
    for idx, row in df.iterrows():
        row_phrases = []
        for template in templates:
            row_phrases.append(format_phrase(row, template))
        df_phrases.append(row_phrases)
    return df_phrases

def main():
    args = parse_plain_args(
        ParseArg("data", type=str, help="Path to the dataset containing sentences"),
        ParseArg(["--templates", "-t"], type=str, nargs='+', help="Format for prompt"),
        ParseArg("--save-phrases", type=str, default="./phrases.txt", help="Path to generated phrases"),
        ParseArg("--add-to-df", action='store_true'),
        ParseArg("--template-names", type=str, default=None, nargs='+', help="Names of columns for generated phrases")
    )
    df = pd.read_csv(args.data)
    templates = args.templates

    # If column names for templates are specified
    if args.template_names is None:
        template_names = templates
    else:
        template_names = args.template_names

    # Generating and saving phrases
    phrases = generate_phrase_groups(df, templates=templates)
    phrases_df = pd.DataFrame(phrases, columns=template_names)
    phrases_df.to_csv(args.save_phrases, index=False)

    # Extending the dataframe with phrases if add_to_df is True
    if args.add_to_df:
        df = pd.concat([df, phrases_df], axis=1)
        df.to_csv(args.data, index=False)

if __name__ == "__main__":
    main()