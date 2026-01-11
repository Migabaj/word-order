import re
import pandas as pd
from typing import Dict

from utils.model_args import parse_plain_args, ParseArg

def split_phrase(phrase: str, regex: str) -> Dict[str, str]:
    """Split a phrase into specified categories using a regular expression.

    Example
    _______
    >>> phrase = "blue puppy"
    >>> regex = r"(?P<adj_eng>\w+) (?P<noun_eng>\w+)"
    >>> split_phrase(phrase, regex)
    {'adj_eng': 'blue', 'noun_eng': 'puppy'}
    """
    if not phrase:
        return {}
    match = re.match(regex, phrase)
    if not match:
        raise ValueError("Phrase does not match the specified structure.")
    return match.groupdict()

def main():
    args = parse_plain_args(
        ParseArg("data", type=str, help="Path to the dataset containing sentences"),
        ParseArg("--column", type=str),
        ParseArg("--regex", type=str),
        ParseArg("--add-to-df", action='store_true'),
    )
    df = pd.read_csv(args.data).fillna("")
    col = args.column
    regex = args.regex

    phrase_dicts = []
    for i, row in df.iterrows():
        phrase_dicts.append(split_phrase(row[col], regex))

    split_df = pd.DataFrame(phrase_dicts)
    print(split_df)

    # Extending the dataframe with phrases if add_to_df is True
    if args.add_to_df:
        df = pd.concat([df, split_df], axis=1)
        df.to_csv(args.data, index=False)

if __name__ == "__main__":
    main()