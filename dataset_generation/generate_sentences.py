PWD = "/nethome/msonkin/word_order_logitlens/"
import sys
sys.path.append(PWD)
import os
os.chdir(PWD)

from constants import VERBNOUN_CSV_PATH, SENTENCE_CSV_PATH
import pandas as pd

perf_pattern_ger = "{np_sub} {haben} {article_acc} {noun_obj} {verb_perf}"
past_pattern_eng = "{np_sub} {verb_past} the {noun_obj}"
pronoun2haben = {
    'Ich': 'habe',
    'Du': 'hast',
    'Er': 'hat'
}
noun2article_acc = {
    'Apfel': 'den',
    'Milch': 'die',
    'Ball': 'den',
    'Kleid': 'das',
    'Spiel': 'das',
    'Präsident': 'den',
    'Mann': 'den',
    'Frau': 'die'
}
noun2noun_acc = {
    'Präsident': 'Präsidenten',
}
np2eng_nom = {
    'Ich': 'I',
    'Er': 'He',
    'Der Mann': 'The man',
    'Die Frau': 'The woman'
}
verbnoun_df = pd.read_csv(VERBNOUN_CSV_PATH)

sentence_dict = {
    'sentence_ger': [],
    'sentence_eng': [],
    'coll_freq': [],
    'logdice': [],
    'obj_freq': [],
    'perf_ger': [],
    'perf_eng': [],
    'past_eng': [],
    'noun_ger': [],
    'noun_eng': []
}
for i, row in verbnoun_df.iterrows():
    verb_perf = row.perf_ger
    verb_eng = row.verb_eng
    verb_past_eng = row.past_eng
    if ' ' in verb_eng:
        continue
    noun_obj = row.noun_ger
    noun_obj_eng = row.noun_eng
    article_acc = noun2article_acc[noun_obj]
    if noun_obj in noun2noun_acc.keys():
        noun_obj = noun2noun_acc[noun_obj]

    for np_sub, np_sub_eng in np2eng_nom.items():
        if np_sub in pronoun2haben.keys():
            haben = pronoun2haben[np_sub]
        else:
            haben = 'hat'
        
        sentence_ger = perf_pattern_ger.format(np_sub=np_sub, haben=haben,article_acc=article_acc,noun_obj=noun_obj,verb_perf=verb_perf)
        sentence_eng = past_pattern_eng.format(np_sub=np_sub_eng, verb_past=verb_past_eng, noun_obj=noun_obj_eng)

        sentence_dict['sentence_ger'].append(sentence_ger)
        sentence_dict['sentence_eng'].append(sentence_eng)
        sentence_dict['coll_freq'].append(row.coll_freq)
        sentence_dict['logdice'].append(row.logdice)
        sentence_dict['obj_freq'].append(row.noun_freq)
        sentence_dict['perf_ger'].append(row.perf_ger)
        sentence_dict['perf_eng'].append(row.perf_eng)
        sentence_dict['past_eng'].append(row.past_eng)
        sentence_dict['noun_ger'].append(row.noun_ger)
        sentence_dict['noun_eng'].append(row.noun_eng)

sentence_df = pd.DataFrame.from_dict(sentence_dict)
sentence_df.to_csv(SENTENCE_CSV_PATH)
