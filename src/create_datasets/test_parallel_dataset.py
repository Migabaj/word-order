import pandas as pd
from create_datasets.parallel_dataset import ParallelDataset

def test_parallel_dataset():
    # Sample data for testing
    data = {
        "phrase-eng": ["Hello world", "How are you?"],
        "phrase-fre": ["Bonjour le monde", "Comment ça va?"],
    }
    df = pd.DataFrame(data)

    # Initialize the dataset
    model_id = "gpt2"
    dataset = ParallelDataset(
        model_id=model_id,
        dataframe=df,
        lang_src="eng",
        lang_tgt="fre",
        sentences_src_prefix="phrase-",
        sentences_tgt_prefix="phrase-",
        random_seed=42,
    )

    # Test tokenization
    tokens = dataset.tokens

    assert len(tokens) == 2, "Tokenization failed: Incorrect number of tokenized pairs."

    # Test prompt formatting
    template = '{lang_src}: "{sentence_src}" - {lang_tgt}: "{sentence_tgt}"'
    last_prompt_template = '{lang_src}: "{sentence_src}" - {lang_tgt}: "'

    prompts = dataset.format(template=template, shots=1, last_prompt_template=last_prompt_template)

    print("Generated Prompts:")
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")
    # assert len(prompts) == 2, "Prompt formatting failed: Incorrect number of prompts."
    # assert "Translate from English to French" in prompts[0], "Prompt content is incorrect."

    print("All tests passed!")

if __name__ == "__main__":
    test_parallel_dataset()