# This format does not end with the end quote because the target sentence
# could be incomplete and the final quote is unimportant for the research.
ONESHOT_FORMAT = """{lang_src}: \"{sentence_src}\" - {lang_tgt}: \"{sentence_tgt}\""""
ONESHOT_FORMAT_CUTOFF = """{lang_src}: \"{sentence_src}\" - {lang_tgt}: \""""

iso2language = {
    "eng": "English",
    "fre": "Français",
    "rus": "Русский",
    "ger": "Deutsch",
    "spa": "Español",
    "ita": "Italiano",
    "zho": "中文",
    "tur": "Türkçe",
    "ned": "Nederlands",
    "ara": "العربية",
    "vie": "Tiếng Việt",
    "pol": "Polski",
    "jap": "日本語"
}

iso2language_eng = {
    "eng": "English",
    "fre": "French",
    "rus": "Russian",
    "ger": "German",
    "spa": "Spanish",
    "ita": "Italian",
    "zho": "Chinese",
    "tur": "Turkish",
    "ned": "Dutch",
    "ara": "Arabic",
    "vie": "Vietnamese",
    "pol": "Polish",
    "jap": "Japanese"
}

start_with_space = {
    "eng": True,
    "fre": True,
    "rus": True,
    "ger": True,
    "spa": True,
    "ita": True,
    "zho": False,
    "tur": True,
    "ned": True,
    "vie": True,
    "pol": True,
    "jap": False
}

LANG2S = {
    "svo": {
        "eng": "verb",
        "zho": "verb",
        "rus": "verb",
        "tur": "object",
        "jap": "object",
    },
    "modal-verbs": {
        "eng": "modal",
        "zho": "modal",
        "tur": "verb",
        "ger": "modal"
    },
    "noun-adj": {
        "eng": "adj",
        "ger": "adj",
        "rus": "adj",
        "zho": "adj",
        "ned": "adj",
        "fre": "noun",
        "ita": "noun",
        "vie": "noun",
    },
    "noun-adj-complete": {
        "eng": "adj",
        "ger": "adj",
        "rus": "adj",
        "zho": "adj",
        "ned": "adj",
        "fre": "noun",
        "ita": "noun",
        "vie": "noun",
    }
}