PWD = '/nethome/msonkin/word_order_logitlens/'
CACHE_DIR = "/scratch/msonkin/word_order_logitlens/cache/"
RANDOM_STATE = 22

VERBNOUN_COUNT_TOP3_PATH = "results/verbnoun_count/top3.json"
VERBNOUN_COUNT_TOP1_PATH = "results/verbnoun_count/top1.json"
VERBNOUN_CSV_PATH = "data/verbnoun.csv"
SENTENCE_CSV_PATH = "data/dataset.csv"

perf_pattern = "{noun_nom} {haben} {article_acc} {noun_acc} {verb_perf}"
impf_pattern = "{noun_nom} {verb_impf} {article_acc} {noun_acc} {verb_prefix}"
pres_pattern = "{noun_nom} {verb_pres} {article_acc} {noun_acc} {verb_prefix}"