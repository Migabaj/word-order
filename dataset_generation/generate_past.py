import subprocess
import pandas as pd
from tqdm import tqdm
from constants import VERBNOUN_CSV_PATH

verbnoun_df = pd.read_csv(VERBNOUN_CSV_PATH)
ger_verb_perf_form_col = []
eng_verb_past_form_col = []
eng_verb_perf_form_col = []

ger_checked_verbs = {}
eng_checked_verbs = {}

for i, row in tqdm(verbnoun_df.iterrows()):
    # ENGLISH TRANSLATION MIGHT BE DIFFERENT
    # if row.verb_ger in ger_checked_verbs.keys():
    #     ger_verb_perf_form_col.append(ger_checked_verbs[row[1].verb])
    #     continue
    ger_verb_perf_forms = []
    eng_verb_past_forms = []
    eng_verb_perf_forms = []
    # ger_result = subprocess.run(
    #     [
    #         "unimorph",
    #         "inflect",
    #         row.verb_ger,
    #         "--features",
    #         "V.PTCP;PST",
    #         "-l",
    #         "deu",
    #         "-q",
    #     ],
    #     stdout=subprocess.PIPE,
    #     check=True,
    # )
    eng_result_past = subprocess.run(
        [
            "unimorph",
            "inflect",
            row.verb_eng.split()[0],
            "--features",
            "V;PST",
            "-l",
            "eng",
            "-q",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    eng_result_perf = subprocess.run(
        [
            "unimorph",
            "inflect",
            row.verb_eng.split()[0],
            "--features",
            "V;V.PTCP;PST",
            "-l",
            "eng",
            "-q",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )

    # ger_inflections = ger_result.stdout.decode("utf-8")
    eng_inflections_past = eng_result_past.stdout.decode("utf-8")
    eng_inflections_perf = eng_result_perf.stdout.decode("utf-8")

    # ger_verb_lines = ger_inflections.split("\n")[:-1]
    # for line in ger_verb_lines:
    #     ger_verb_past_form = line.split("\t")[1]
    #     ger_verb_perf_forms.append(ger_verb_past_form)
    # if len(ger_verb_lines) == 1:
    #     ger_verb_perf_form_col.append(ger_verb_perf_forms[0])
    #     # ger_checked_verbs[row[1].verb_ger] = ger_verb_perf_forms[0]
    #     print(ger_verb_perf_forms[0])
    # else:
    #     ger_verb_perf_form_col.append(ger_verb_perf_forms)
    #     # ger_checked_verbs[row[1].verb_ger] = ger_verb_perf_forms
    #     print(ger_verb_perf_forms)

    eng_verb_lines_past = eng_inflections_past.split("\n")[:-1]
    for line in eng_verb_lines_past:
        eng_verb_past_form = line.split("\t")[1]
        eng_verb_past_forms.append(eng_verb_past_form)
    if len(eng_verb_lines_past) == 1:
        eng_verb_past_form_col.append(eng_verb_past_forms[0])
        # eng_checked_verbs[row[1].verb] = eng_verb_past_forms[0]
        print(eng_verb_past_forms[0])
    else:
        eng_verb_past_form_col.append(eng_verb_past_forms)
        # eng_checked_verbs[row[1].verb] = eng_verb_past_forms
        print(eng_verb_past_forms)

    eng_verb_lines_perf = eng_inflections_perf.split("\n")[:-1]
    for line in eng_verb_lines_perf:
        eng_verb_perf_form = line.split("\t")[1]
        eng_verb_perf_forms.append(eng_verb_perf_form)
    if len(eng_verb_lines_perf) == 1:
        eng_verb_perf_form_col.append(eng_verb_perf_forms[0])
        # eng_checked_verbs[row[1].verb] = eng_verb_perf_forms[0]
        print(eng_verb_perf_forms[0])
    else:
        eng_verb_perf_form_col.append(eng_verb_perf_forms)
        # eng_checked_verbs[row[1].verb] = eng_verb_perf_forms
        print(eng_verb_perf_forms)

# verbnoun_df["perf_ger"] = ger_verb_perf_form_col
verbnoun_df["perf_eng"] = eng_verb_perf_form_col
verbnoun_df["past_eng"] = eng_verb_past_form_col
verbnoun_df.to_csv(VERBNOUN_CSV_PATH)
