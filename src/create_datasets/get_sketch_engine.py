import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# insert your username and API key
USERNAME = ""
API_KEY = ""
BASE_URL = "https://api.sketchengine.eu/bonito/run.cgi"

nouns = [
    ("Apfel", "Apfel"),
    ("Milch", "Milch"),
    ("Ball", "Ball"),
    ("Kleid", "Kleid"),
    ("Spiel", "Spiel"),
    ("Präsident", "Präsidenten"),
    ("Mann", "Mann"),
    ("Frau", "Frau")
    ]

def phrase_has_singular_noun(string, noun):
    return noun+" " in string

colloc_dict = {
    'noun': [],
    'verb': [],
    'cm': [],
    'coll_freq': [],
    'logdice': [],
    'noun_freq': [],
}
for noun, noun_acc in nouns:
    data = {
        "corpname": "preloaded/detenten23_rft3",
        # "sort_gramrels": "freq",
        "sort_ws_columns": "f",
        "format": "json",
        "lemma": noun,
        "lpos": "-n",
        "min_freq": 10,
        "minscore": 0.0,
        "maxitems": 100,
        "expand_seppage": 1
    }

    d = requests.get(
        BASE_URL + "/wsketch", auth=(USERNAME, API_KEY), params=data
    ).json()
    noun_freq = d['freq']
    verbs_with_acc_obj = d["Gramrels"][
        [gramrel["name"] for gramrel in d["Gramrels"]].index(
            'verbs with "%w" as accusative object'
        )
    ]
    verb_infos = verbs_with_acc_obj['Words']
    verb_infos = list(filter(lambda x: phrase_has_singular_noun(x["cm"], noun_acc), verb_infos))

    # verb_infos_sorted_score = sorted(verb_infos, key=lambda x: x['score'], reverse=True)

    # verb_infos_hashable = [
    #     (verb['id'], verb['lempos'], verb['count'], verb['score'], verb['cm'])
    #     for verb in verb_infos
    # ]

    for verb in verb_infos:
        colloc_dict['noun'].append(noun)
        colloc_dict['noun_freq'].append(d['freq'])
        colloc_dict['verb'].append(verb['word'])
        colloc_dict['cm'].append(verb['cm'])
        colloc_dict['coll_freq'].append(verb['count'])
        colloc_dict['logdice'].append(verb['score'])

    print(f"Noun {noun} done!")
    time.sleep(5)

noun_df = pd.DataFrame.from_dict(colloc_dict)
noun_df.to_csv("verbnoun.csv")

# 'verbs with "%w" as accusative object'
