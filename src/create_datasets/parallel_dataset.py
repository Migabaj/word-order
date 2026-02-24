import torch
import random
import pandas as pd
from typing import List, Dict, Union
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

from utils.langs import iso2language


class ParallelDataset:
    """
    Class to prepare a dataset for a certain LLM by tokenizing and determining the languages.
    """

    def __init__(
        self,
        model: Union[str, AutoModelForCausalLM],
        dataframe: pd.DataFrame,
        lang_src: str,
        lang_tgt: str,
        sentences_src_prefix: str = "phrase",
        sentences_tgt_prefix: str = "phrase",
        code2language: Dict[str, str] = iso2language,
        cache_dir=None,
        random_seed: int = None,
    ):
        if model is not None and not isinstance(model, str):
            self.model = model
            self.model_id = None
            if isinstance(model, HookedTransformer):
                self.tokenizer = model.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, cache_dir=cache_dir
                )
        else:
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.df = dataframe
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.sentences_src_prefix = sentences_src_prefix
        self.sentences_tgt_prefix = sentences_tgt_prefix
        self.sentences_src = self.df[f"{sentences_src_prefix}-{lang_src}"].tolist()
        self.sentences_tgt = self.df[f"{sentences_tgt_prefix}-{lang_tgt}"].tolist()
        self.sentences = []
        for sent_src, sent_tgt in zip(self.sentences_src, self.sentences_tgt):
            self.sentences.append((sent_src, sent_tgt))
        self.code2language = code2language
        self.seed = random_seed
        random.seed(self.seed)

        # tokenize all sentences
        self.tokens = self._tokenize()

    def _tokenize(self):
        tokens = []
        for sent_src, sent_tgt in self.sentences:  # src, tgt
            tokens.append(
                (
                    self.tokenizer(sent_src, return_tensors="pt", truncation=True),
                    self.tokenizer(sent_tgt, return_tensors="pt", truncation=True),
                )
            )
        return tokens

    def format(
        self,
        template,
        shots=1,
        last_prompt_template=None,
        shot_data_src: str = None,
        shot_data_tgt: str = None,
        shuffle_shots: bool = True,
    ) -> List[str]:
        prompts = []

        # Define what sentences to use for shots
        if shot_data_src is not None:
            sentences_src = self.df[shot_data_src].tolist()
        else:
            sentences_src = self.sentences_src

        if shot_data_tgt is not None:
            sentences_tgt = self.df[shot_data_tgt].tolist()
        else:
            sentences_tgt = self.sentences_tgt

        shot_sentences = list(zip(sentences_src, sentences_tgt))
        if shuffle_shots:
            random.shuffle(shot_sentences)
        else:
            # Rotate the list by 1 to avoid using the first sentence as shot for the first prompt
            shot_sentences = shot_sentences[1:] + shot_sentences[0:1]

        for sent_i, sent in enumerate(self.sentences):
            shot_sent_i = sent_i + 1 # so that while loop starts from sent_i
            # TODO: ew, fix later
            prompt = ""
            shot_n = 0

            # Write the shots
            while shot_n < shots:
                shot_sent_i = shot_sent_i - 1
                if shot_sent_i < 0:
                    shot_sent_i = len(shot_sentences) - 1
                prompt += template.format(
                    lang_src=self.code2language[self.lang_src],
                    lang_tgt=self.code2language[self.lang_tgt],
                    sentence_src=shot_sentences[shot_sent_i][0],
                    sentence_tgt=shot_sentences[shot_sent_i][1],
                ) + "\n"
                shot_n += 1

            # Write the actual prompt
            if last_prompt_template is not None:
                prompt += last_prompt_template.format(
                    lang_src=self.code2language[self.lang_src],
                    lang_tgt=self.code2language[self.lang_tgt],
                    sentence_src=sent[0],
                    sentence_tgt=sent[1],
                )
            else:
                prompt += template.format(
                    lang_src=self.code2language[self.lang_src],
                    lang_tgt=self.code2language[self.lang_tgt],
                    sentence_src=sent[0],
                    sentence_tgt=sent[1],
                )
            prompts.append(prompt)
        self.template = template
        self.prompts = prompts
        return prompts

    def prompts_to_tokens(self) -> List[Dict[str, Dict]]:
        """
        Convert prompts to tokens.
        Returns a list of dicts with 'src' and 'tgt' tokens.
        """
        if hasattr(self, "prompts_tokens"):
            return self.prompts_tokens

        prompts_tokens = []
        for prompt in self.prompts:
            prompt_tokens = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            )
            prompts_tokens.append(prompt_tokens)

        self.prompts_tokens = prompts_tokens
        return prompts_tokens