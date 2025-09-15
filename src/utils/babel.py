import os
import requests
from collections import defaultdict
from typing import Dict, List

def add_auth_key_to_params(auth_key: str, params: Dict[str, str], key_name : str ='key') -> Dict[str, str]:
    """
    Add API key to the parameters for requests.
    """
    return {**params, key_name: auth_key}

def babelnet_request(command : str, params: Dict[str, str], version : str ='v9') -> requests.Response:
    """
    Returns a request from BabelNet API.
    """
    url = f'https://babelnet.io/{version}'
    r = requests.get(
        os.path.join(url, command),
        params=params)
    return r

def divide_senses_per_language(senses: List[dict], target_langs=[]) -> Dict[str, List[dict]]:
    senses_per_language = defaultdict(list)
    for item in senses:
        properties = item.get("properties", {})
        language = properties.get("language", None)
        senses_per_language[language].append(item)
    print("Lang keys", senses_per_language.keys())
    return senses_per_language

def get_overlapping_synsets(senses: List[dict], target_langs: List[str]):
    """
    Get synset IDs, for which there are words in all given languages.
    """
    lang2synsets = {}
    lang2senses = divide_senses_per_language(senses)
    for lang in target_langs:
        print("Looking at language", lang)
        lang2synsets[lang] = set([sense["properties"]["synsetID"]["id"] for sense in lang2senses[lang]])
    
    return set.intersection(*[synset for synset in lang2synsets.values()])


if __name__ == "__main__":
    pass