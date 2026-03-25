This is the code that was used to run our experiments.

The `src/` directory contains the "building block" code used for running experiments.

## LogitLens

The scripts for our LogitLens experiments are in the `notebooks/logit_attribution` directory.

`lens.py` contains the script for out LogitLens experiments. Example of how to execute it in bash:

```
python3 notebooks/logit_attribution/lens.py \
    --model-id ai-forever/mGPT \
    --langs $langs_str \
    --src-lang eng \
    --datapath data/modal-verbs.csv \
    --sample-size 31 \
    --pos-prefixes modal verb \
    --sentences-src-prefix phrase \
    --sentences-tgt-prefix subject \
    --shot-data-src-prefix phrase \
    --shot-data-tgt-prefix phrase \
    --hf-token YOUR_TOKEN \
    --cache-dir CACHE_DIR
```

## Intervention

The script for intervention is written in `notebooks/intervention/intervene.py`. Example of how you could run it on the Modal Verbs dataset:

```
python3 notebooks/intervention/intervene.py --model-id ai-forever/mGPT
    --datapath "data/modal-verbs_haveto.csv" \
    --sample-size 31 \
    --pos-prefixes "verb" "modal" \
    --sentences-src-prefix "phrase" \
    --sentences-tgt-prefix "phrase_cutoff_after_subject" \
    --shot-data-src-prefix "phrase" \
    --shot-data-tgt-prefix "phrase" \
    --num-shots 1 \
    --hf-token YOUR_TOKEN \
    --save-path "output/intervention/probs/modal-verbs_mGPT_{src_lang_base}-{tgt_lang_base}_{src_lang_plant}-{tgt_lang_plant}.csv" \
    --lang-setup-grid-path "notebooks/intervention/params/modal-verbs.csv" \
    --plant-datapath data/modal-verbs_want.csv \
    --check-for-filename
```

In the same directory, the `plot.ipynb` replicates some of the plots used for our visualizations.


### Head Intervention

Head intervention script is written in `notebooks/intervention/intervene-heads.py`. The `plot-heads.ipynb` notebook prints S-sensitive heads.

## Steering

The "steering" script is in the `notebooks/steer` directory. Mean activation collection is in `collect_mean_activations.py`. Steering script is in `steer.py`. Resulting plots are in `plot.ipynb`.

Example of mean activataion collection script in bash:

```
python3 notebooks/steer/collect_mean_activations.py \
    --model-id ai-forever/mGPT \
    --datapath data/noun-adj-complete.csv \
    --langs eng ger rus zho ned fre ita vie \
    --sample-size 198 \
    --save-path output/activations/noun-adj-complete_mGPT_{src_lang}-{tgt_lang}_layer11_head2.pt \
    --layer 11 \
    --head 2 \
    --sentences-src-prefix phrase_complete \
    --sentences-tgt-prefix phrase_prefix \
    --shot-data-src-prefix phrase_complete \
    --shot-data-tgt-prefix phrase_complete
```

Example of steering:

```
python3 notebooks/steer/steer.py \
    --model-id ai-forever/mGPT \
    --langs rus zho tur jap \
    --src-lang eng \
    --datapath data/svo.csv \
    --sample-size 120 \
    --pos-prefixes object verb \
    --sentences-src-prefix phrase \
    --sentences-tgt-prefix subject \
    --shot-data-src-prefix phrase \
    --shot-data-tgt-prefix phrase \
    --head 11 \
    --layer 2
```