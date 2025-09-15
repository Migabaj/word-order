import time
import requests
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import wordnet as wn

from utils.model_args import parse_plain_args, ParseArg

# insert your username and API key
USERNAME = "nejenek"
API_KEY = "424ad10e0113b3f3e517949e82c787dd"

BASE_URL = "https://api.sketchengine.eu/bonito/run.cgi"

def get_frequent_words(limit=10_000):
    data = {
        "corpname": "preloaded/detenten23_rft3",
        "format": "json",
        "wlmaxitems": limit,
        "wlsort": "frq",
        "wlattr": "word",
        "wlpat": ".*"
    }
    response = requests.get(
        BASE_URL + "/wordlist", auth=(USERNAME, API_KEY), params=data
    )
    words = []
    if response.status_code == 200:
        result = response.json()
        if result and "Items" in result:
            for item in result["Items"]:
                words.append(item["str"])
        else:
            print("Unexpected response structure:", result)
    else:
        print("Whoops, status code:", response.status_code)
    return words

# Get frequent German nouns from SketchEngine API
def get_frequent_nouns(limit=100):
    data = {
        "corpname": "preloaded/detenten23_rft3",
        "format": "json",
        "attr": "lemma",  # get frequent lemmas
        "maxnum": limit,
        "minfreq": 50,
        "refs": "frq",
        "sort": "random",  # random sort
        "ignorecase": "1",
        "pos": "N"  # only nouns
    }
    # Do NOT add 'lemma' parameter here; it causes the API to fail
    response = requests.get(
        BASE_URL + "/freqs", auth=(USERNAME, API_KEY), params=data
    )
    nouns = []
    if response.status_code == 200:
        result = response.json()
        if result and "data" in result:
            for item in result["data"]:
                nouns.append(item["name"])
        else:
            print("Unexpected response structure:", result)
    else:
        print("Whoops, status code:", response.status_code)
    return nouns

# Get top adjectives (modifiers) for a noun
def get_adjectives_for_noun(noun: str, corpname: str, auth: Tuple[str, str], min_freq=5):
    if not noun or not isinstance(noun, str) or noun.strip() == "":
        print(f"Skipping empty or invalid noun: '{noun}'")
        return []
    data = {
        "corpname": corpname,
        "format": "json",
        "lemma": noun,
        "lpos": "-n",
        "maxitems": 50,
        "min_freq": min_freq,
        "expand_seppage": 1
    }
    response = requests.get(
        BASE_URL + "/wsketch", auth=auth, params=data
    )
    adjectives = []
    if response.status_code == 200:
        d = response.json()
    else:
        raise requests.ConnectionError
    for gramrel in d.get("Gramrels", []):
        if gramrel.get("name", "").lower() == 'modifier':
            for word in gramrel.get("Words", []):
                if word.get("lempos", "").endswith("-j"):  # Adjective POS
                    adjectives.append((word["word"], word["count"], word["score"]))
    return sorted(adjectives, key=lambda x: x[2], reverse=True)

def main():
    a = get_adjectives_for_noun('apple', 'preloaded/bnc2', auth=requests.auth.HTTPBasicAuth(USERNAME, API_KEY))
    print(a)

if __name__ == '__main__':
    main()