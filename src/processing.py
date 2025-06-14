class TokenizerProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def token_probabilities(self, token, probs):
        token_id = self.vocab.get(token, None)
        return probs[..., token_id] if token_id is not None else None

    def topk_tokens(self, probs, k, return_text=False):
        top_tokens = probs.topk(k, dim=-1)
        if return_text:
            return top_tokens, self._decode_indices(top_tokens.indices)
        return top_tokens

    def _decode_indices(self, indices):
        decoded = []
        for instance in indices:
            layer_tokens = []
            for layer in instance:
                layer_tokens.append(self.tokenizer.convert_ids_to_tokens(layer))
            decoded.append(layer_tokens)
        return decoded

def main():
    # Example usage
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B", cache_dir="/scratch/msonkin/word-order-thesis/cache/"
    )
    processor = TokenizerProcessor(tokenizer)
    probs = torch.load("/nethome/msonkin/word-order-thesis/output/eng-ger/past-participle/gptj/probs/probs_sentence_f_cutoff_after_haben.pt")

    # Get top k tokens
    topk_tokens, topk_texts = processor.topk_tokens(probs, k=5, return_text=True)
    print("Top 5 tokens:", topk_texts)

if __name__ == "__main__":
    main()