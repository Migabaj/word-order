"""Model wrappers are designed to make it easier to analyze the inner workings
of LLMs."""

from typing import Union, Optional, List, Tuple #, TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BloomTokenizerFast
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def get_device() -> str:
    """Get current available device.

    .. code-block:: python
        
        >>> get_device()
        'cpu'


    :return: Current working device
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device

def load_mgpt(
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load mGPT model and tokenizer.
    
    .. code-block:: python
        
        model, tokenizer = load_mgpt()
        
    :param cache_dir: The cache directory for mGPT's weights, defaults to `None`
    :return model: The mGPT model
    :return tokenizer: mGPT's tokenizer
    :rtype model: AutoModelForCausalLM
    :rtype tokenizer: AutoTokenizer
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        "sberbank-ai/mGPT", cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        "sberbank-ai/mGPT",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).to(device)
    return model, tokenizer

def load_gptj(
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load GPT-J model and tokenizer.

    .. code-block:: python
        
        model, tokenizer = load_gptj()

    :param cache_dir: The cache directory for GPT-J's weights, defaults to `None`
    :return model: The GPT-J model
    :return tokenizer: GPT-J's tokenizer
    :rtype model: AutoModelForCausalLM
    :rtype tokenizer: AutoTokenizer
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B", cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    ).to(device)
    return model, tokenizer


# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd
#     def forward(self, x):
#         return self.lambd(x)


class ModelWrapper(nn.Module):
    """Wrapper for analyzing LLMs' activations

    :param model: Model to load
    :param tokenizer: The appropriate tokenizer for the model
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Constructor method"""
        super().__init__()
        self.model = model.eval()
        self.model.activations_ = {}
        self.tokenizer = tokenizer
        self.device = get_device()
        self.num_layers = len(self.model.transformer.h)
        self.hooks = []
        self.layer_pasts = {}

    def tokenize(
        self, s: str, output_string: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """Tokenize a string

        :param s: String to tokenize
        :param output_string: `True` if the returned value must include the list of string outputs
        :return: String encoded by the wrapper's tokenizer
        """
        token_ids = self.tokenizer.encode(s, return_tensors="pt").to(self.device)
        if output_string:
            tokens = self.list_decode(token_ids)
            return token_ids, tokens
        return token_ids

    def list_decode(self, inpids: List[int]) -> List[str]:
        """Decode input ids

        :param inpids: Input ids
        :return: List of tokens as strings
        """
        return [self.tokenizer.decode(s) for s in inpids]

    def layer_decode(self, hidden_states):
        """Project the hidden states onto the vocabulary.
            Must be implemented specifically for each wrapper class

        :param hidden_states: Hidden states to be projected
        :raises NotImplementedError: For ModelWrapper, an Exception is automatically raised, as
            the `layer_decode` has to be implemented for each given wrapper class
        """
        raise NotImplementedError("Layer decode has to be implemented!")

    def get_layers(self, tokens: List[int], **model_kwargs) -> torch.Tensor:
        """Decode hidden states and return a tensor of the resulting projections.
            Returns a tensor of shape (\ :math:\)

        :param tokens: List of token ids
        :param model_kwargs: Possible keyword arguments for the model's forward call
        """
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **model_kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        # logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits).squeeze(-1)  # , true_logits.squeeze(0)

    def get_layers_w_attns(self, tokens, **model_kwargs):
        outputs = self.model(
            input_ids=tokens,
            output_hidden_states=True,
            output_attentions=True,
            **model_kwargs,
        )
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        # logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1)
        return (
            torch.stack(logits).squeeze(-1),
            outputs.attentions,
        )  # , true_logits.squeeze(0)

    def rr_per_layer(self, logits, answer, debug=False):
        # reciprocal rank of the answer at each layer
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)

        rrs = []
        for i, layer in enumerate(logits):
            soft = F.softmax(layer, dim=-1)
            sorted_probs = soft.argsort(descending=True)
            rank = float(np.where(sorted_probs.cpu().numpy() == answer_id)[0][0])
            rrs.append(1 / (rank + 1))

        return np.array(rrs)

    def prob_of_answer(self, logits, answer, debug=False):
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)
        answer_probs = []
        first_top = -1
        mrrs = []
        for i, layer in enumerate(logits):
            soft = F.softmax(layer, dim=-1)
            answer_prob = soft[answer_id].item()
            sorted_probs = soft.argsort(descending=True)
            if debug:
                print(f"{i}::", answer_prob)
            answer_probs.append(answer_prob)
        # is_top_at_end = sorted_probs[0] == answer_id
        return np.array(answer_probs)

    def get_top_ids_per_layer(
        self, logits: List[torch.Tensor], k: int = 10
    ) -> torch.Tensor:
        layer_tokens = torch.zeros(logits.shape[0], k, dtype=torch.int64)
        for i, layer in enumerate(logits):
            layer_tokens[i] = F.softmax(layer, dim=-1).argsort(descending=True)[:k]
        return layer_tokens

    def print_top(self, logits: List[torch.Tensor], k: int = 10) -> str:
        """Print top-`k` tokens predicted by each layer of the model

        :param logits: List of layer activations
        :param k: Controls the `k` variable in top-`k`"""
        result_output = ""
        topk_tokens = self._get_top_ids_per_layer(logits, k=k)
        for i in range(len(logits)):
            result_output += f"""{i} {'|'.join(
                self.tokenizer.convert_ids_to_tokens(topk_tokens[i])
            )}\n"""
        print(result_output)
        return result_output

    def print_top_concise(self, logits):
        layer_tokens = []
        layer_aggr = {}
        for i, layer in enumerate(logits):
            # TODO: add k!=1
            top_token = self.tokenizer.convert_ids_to_tokens(
                F.softmax(layer, dim=-1).argsort(descending=True)[:1]
            )
            layer_tokens.append(top_token)
        layer_repr = ""
        span_start = -1
        for i, token_list in enumerate(layer_tokens):
            token = token_list[0]
            if token != layer_repr:
                layer_aggr[(span_start, i - 1)] = layer_repr
                layer_repr = token
                span_start = i
        layer_aggr[(span_start, len(layer_tokens) - 1)] = layer_repr
        for span, token in sorted(layer_aggr.items()):
            if span[0] == span[1]:
                result_output += f"{span[0]}\t{token}" + "\n"
            else:
                result_output += f"{span[0]}-{span[1]}\t{token}" + "\n"
        print(result_output)

    def print_tops_with_offset(
        self,
        text: str,
        translated: str = "",
        offset: Optional[int] = None,
        concise: bool = False,
        markdown: bool = False,
        k: int = 3,
    ):
        assert offset is None or offset < 0

        result_output = ""
        inp_ids = self.tokenize(text)
        if markdown:
            result_output += f"# {translated}\n"

        for token_spot in range(offset, 1):
            if token_spot == 0:
                offset_i = None
            else:
                offset_i = token_spot

            inp_ids_segment = inp_ids[:, :offset_i]
            logits = self.get_layers(inp_ids_segment)
            if markdown:
                head = self.tokenizer.decode(inp_ids_segment[0]).split("\n")[-1]
                header = f"## {head}"
            else:
                header = f"TEXT: {self.tokenizer.decode(inp_ids_segment[0])}"
            result_output += header + "\n"  # TODO: zero
            if markdown:
                result_output += "```"
            # print(self.tokenizer.decode(""))
            if concise:
                result_output += self.print_top_concise(logits[1:])
            else:
                result_output += self.print_top(logits[1:], k=k)
            if markdown:
                result_output += "```"
            result_output += "\n"
        return result_output

    def topk_per_layer(self, logits, k=10):
        topk = []
        for i, layer in enumerate(logits):
            topk.append(
                [
                    self.tokenizer.decode(s)
                    for s in F.softmax(layer, dim=-1).argsort(descending=True)[:k]
                ]
            )
        return topk

    def get_activation(self, name):
        # https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            if "in_sln" in name:
                num_tokens = list(input[0].size())[1]
                self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    self.model.activations_[name] = output[0][
                        :, num_tokens - 1
                    ].detach()
                    self.model.activations_["in_" + name] = input[0][
                        :, num_tokens - 1
                    ].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[
                        0
                    ]  # [num_tokens, 3072] for values;
                    self.model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[
                        1
                    ]  # (batch, sequence, hidden_state)
                    self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(self.num_layers - 1):
                    self.model.activations_[name] = (
                        self.model.activations_[
                            "intermediate_residual_" + str(final_layer)
                        ]
                        + self.model.activations_["mlp_" + str(final_layer)]
                    )

                else:
                    if "out" in name:
                        self.model.activations_[name] = output[0][
                            num_tokens - 1
                        ].detach()
                    else:
                        self.model.activations_[name] = input[0][
                            :, num_tokens - 1
                        ].detach()

        return hook

    def reset_activations(self):
        """Reset wrapper's activations to none"""
        self.model.activations_ = {}


class GPTWrapper(ModelWrapper):
    """Wrapper for the GPT-J model."""

    def layer_decode(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Project hidden states onto the vocab.

        :param hidden_states: Model's hidden states of shape
            \ :math:`(N_L, b, d_v, d_h)`\, where

                - \ :math:`N_L`\: number of layers
                - \ :math:`b`\: batch size
                - \ :math:`d_v`\: vocab size
                - \ :math:`d_h`\: dimensionality of hidden states
        :return: List of logits, i.e. model's hidden states projected onto its vocabulary.
            Each list element is a one-dimensional tensor of length \ :math:`d_v`\
        """
        logits = []
        for i, h in enumerate(hidden_states):
            h = h[:, -1, :]  # (batch, num tokens, embedding size) take the last token
            if i == len(hidden_states) - 1:
                normed = h  # ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

# TODO: see if add_hooks is specific to GPT-J
class GPTJWrapper(GPTWrapper):
    """Wrapper for the GPT-J model."""

    def add_hooks(self):
        for i in range(self.num_layers):
            # intermediate residual between
            # print('saving hook')
            self.hooks.append(
                self.model.transformer.h[i].ln_1.register_forward_hook(
                    self.get_activation(f"in_sln_{i}")
                )
            )
            # TODO: figure this out?
            # self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_activation('attn_'+str(i))))
            self.hooks.append(
                self.model.transformer.h[i].mlp.register_forward_hook(
                    self.get_activation("intermediate_residual_" + str(i))
                )
            )
            self.hooks.append(
                self.model.transformer.h[i].mlp.register_forward_hook(
                    self.get_activation("mlp_" + str(i))
                )
            )
            # print(self.model.activations_)

if __name__ == "__main__":
    model, _ = load_mgpt(cache_dir="/scratch/msonkin/word-order-thesis/cache/")
    print(model)