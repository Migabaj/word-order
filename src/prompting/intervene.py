import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import pyvene as pv
from pyvene import top_vals
from pyvene.models.modeling_utils import getattr_for_torch_module
from utils.model_args import model_to_num_layers_attr, model_to_num_heads_attr

def intervention_config(model_type, intervention_type, unit, layer):
    """
    Parameters
    __________

    model_type: model type
    intervention_type: component for RepresentationConfig, e.g. head_attention_value_output
    unit: string to define the component type, e.g. "h" (head), "pos" (position), "h.pos" (head within position)
    later: layer id
    """
    # Set up the config to intervene
    config = pv.IntervenableConfig(
        model_type=model_type,
        representations=[
            pv.RepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                unit,  # intervention unit is now [pos] within [h]
                1,  # max number of unit
            ),
        ],
        intervention_types=pv.VanillaIntervention,
    )
    return config

def intervene(
        model,
        base,
        source,
        base_pos: int,
        source_pos: int,
        component_type,
        layer_i,
        head_i = None,
):
    if component_type in ["block_output", "mlp_output", "attention_value_output"]:
        unit = "pos"
    elif component_type == "head_attention_value_output":
        unit = "h.pos"
    else:
        raise NotImplementedError(f"Unsupported component type: {component_type}")
    config = intervention_config(
        type(model), component_type, unit, layer_i
    )
    intervenable = pv.IntervenableModel(config, model)
    if head_i is not None:
        unit_locations = {
            "sources->base": (
                [[[[head_i]], [[source_pos]]]],  # intervene w/ target_head's pos_i
                [[[[head_i]], [[base_pos]]]]
            ),
        }
    else:
        unit_locations = {"sources->base": (source_pos, base_pos)}
    _, counterfactual_outputs = intervenable(
        base,
        source,
        unit_locations,
    )
    return counterfactual_outputs

def intervention_data(
        model,
        tokenizer,
        base: torch.Tensor,
        source: torch.Tensor,
        base_pos: int,
        source_pos: int,
        tokentype2token: Dict[str, str],
        component_type: str,
        sentence_index: int = None,
        head_i: int = None,
        data: List[Dict[str, Any]] = None,
        data_topk: List[Dict[str, Any]] = None,
        patch_layers: Optional[List[int]] = None,
        write_down_top_k: int = 5,
        lang2s: Optional[Dict[str, Dict[str, str]]] = None,
        base_plant_langs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
    """
    Collect intervention data for a given model component (block output or head attention value output).

    Parameters
    ----------
    base : torch.Tensor
        The tokenized base prompt tensor.
    source : torch.Tensor
        The tokenized source prompt tensor.
    base_pos : int
        The position of the last token in the base prompt.
    source_pos : int
        The position of the last token in the source prompt.
    tokentype2token : Dict[str, str]
        A dictionary mapping token types (e.g., 'noun-base', 'adj-base') to their corresponding token strings.
    component_type : str
        The type of model component to intervene on. Must be either 'block_output' or 'head_attention_value_output'.
    head_i : int, optional
        The index of the head to intervene on (if applicable). Only used for head-level interventions.
    data : List[Dict[str, Any]], optional
        A list to append the collected data to. If None, a new list will be created.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the intervention data, with keys:
        - "token_type": The type of token (e.g., 'noun-base', 'adj-base').
        - "token": The token string.
        - "prob": The probability of the token after intervention.
        - "layer": The layer index.
        - "head_i": The head index (if applicable).
        - "pos": The position index.
        - "type": The component type (e.g., 'block_output').

    Raises
    ------
    NotImplementedError
        If the component_type is not 'block_output' or 'head_attention_value_output'.
    """
    sm = torch.nn.Softmax(dim=2)
    if data is None:
        data = []
    if data_topk is None:
        data_topk = []
    
    model_class = model.__class__.__name__
    if patch_layers:
        layers = patch_layers
    else:
        num_layers = getattr_for_torch_module(model, model_to_num_layers_attr[model_class])
        layers = list(range(num_layers))

    for layer_i in layers:
        outputs = intervene(model, base, source, base_pos, source_pos, component_type=component_type, layer_i=layer_i, head_i=head_i)
        with torch.inference_mode():
            distrib = sm(outputs.logits)
        # print("DISTRIB SHAPE:", distrib.shape)
        # print()
        # print(f"\nTOP VALUES AT LAYER #{layer_i} AT POSITION {base_pos}:")
        if write_down_top_k:
            top_k = top_vals(tokenizer, distrib[0][base_pos], 5, return_results=True)
            data_topk.append({
                "sentence_id": sentence_index,
                "layer": layer_i,
                "head_id": head_i,
                "pos": base_pos,
                "token": [t[1] for t in top_k],
                "prob": [t[0] for t in top_k]
            })
        for token_type, token in tokentype2token.items():
            part_of_speech, language, lexical_component = token_type.split("-")
            # print(token_type, token, tokenizer.encode(token, add_special_tokens=False)[0], tokenizer.convert_ids_to_tokens(tokenizer.encode(token, add_special_tokens=False)[0]))
            try:
                data.append(
                    {
                        "sentence_id": sentence_index,
                        "token_type": token_type,
                        "token": token,
                        "prob": float(distrib[0][base_pos][tokenizer.encode(token, add_special_tokens=False)[0],]),
                        "layer": layer_i,
                        "head_id": head_i,
                        "pos": base_pos,
                        "type": component_type,
                        "part_of_speech": part_of_speech,
                        "language": language,
                        "lexical_component": lexical_component,
                    }
                )
            except Exception as e:
                print(f"Error processing token '{token}' of type '{token_type}' in sentence id {sentence_index}: {e}")
            # print(token_type, token, tokenizer.encode(token, add_special_tokens=False)[0], tokenizer.convert_ids_to_tokens(tokenizer.encode(token, add_special_tokens=False)[0]))
                continue

    if write_down_top_k:
        return data, data_topk
    return data

def get_module_by_name(model, module_name: str):
    """
    Retrieve submodule by its full dotted name.
    Works for any HF model.
    """
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"Module {module_name} not found in model.")
    return modules[module_name]


def collect_mean_activation(
    model,
    tokenizer,
    prompts: List[str],
    component_type: str,
    layer_i: int,
    target_pos: Optional[int] = None,
    unit: str = "pos",
    head_i: Optional[int] = None,
    device: str = "cuda",
):
    """
    Collect mean activation using pyvene's CollectIntervention.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: matching tokenizer
        prompts: list of source prompts to collect activations from
        component_type: type of component (e.g., 'block_output', 'mlp_output', 
                       'attention_value_output', 'head_attention_value_output')
        layer_i: layer index to collect from
        target_pos: token position to extract. If None, uses last token position
        unit: intervention unit ('pos' for position, 'h.pos' for head+position)
        head_i: head index (for head-level interventions)
        device: device to run on
    
    Returns:
        mean_activation: tensor of shape [hidden_dim] containing mean activation
    """
    model.eval()
    model.to(device)
    
    # Create intervention config with CollectIntervention
    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[
            pv.RepresentationConfig(
                layer_i,
                component_type,
                unit,
            ),
        ],
        intervention_types=pv.CollectIntervention,
    )
    
    intervenable = pv.IntervenableModel(config, model)
    
    all_activations = []
    
    with torch.no_grad():
        for prompt in prompts:
            
            # Determine position if not specified
            if target_pos is None:
                pos = prompt.input_ids.shape[-1] - 1
            else:
                pos = target_pos
            
            # Set up unit locations for collection
            if head_i is not None:
                unit_locations = {
                    "sources->base": (
                        [[[[head_i]], [[pos]]]],
                        [[[[head_i]], [[pos]]]]
                    )
                }
            else:
                unit_locations = {"sources->base": pos}
            
            # CollectIntervention returns (original_output, collected_activations)
            (_, collected), _ = intervenable(
                prompt,
                unit_locations=unit_locations,
            )
            
            # Extract activations from intervention
            # collected is a dict with representation indices as keys
            activation = collected[0]  # Get first (and only) representation
            
            # Flatten batch and append
            all_activations.append(activation.detach().cpu())
    
    # Stack all activations and compute mean
    stacked_activations = torch.stack(all_activations, dim=0)
    mean_activation = stacked_activations.mean(dim=0)
    
    return mean_activation


def steer(
        model,
        tokenizer,
        base,
        source_representation,
        component_type,
        layer_i,
        head_i=None,
        output_original_output=True
):
    base_pos = base.input_ids.shape[1] - 1
    
    if component_type in ["block_output", "mlp_output", "attention_value_output"]:
        unit = "pos"
    elif component_type == "head_attention_value_output":
        unit = "h.pos"
    else:
        raise NotImplementedError(f"Unsupported component type: {component_type}")

    config = intervention_config(
        type(model), component_type, unit, layer_i
    )

    intervenable = pv.IntervenableModel(config, model)

    # Patch base with mean activation
    if head_i is not None:
        unit_locations = {
            "sources->base": (
                [[[[head_i]], [[0]]]],
                [[[[head_i]], [[base_pos]]]],
            ),
        }
    else:
        unit_locations = {"base": base_pos}

    base_outputs, counterfactual_outputs = intervenable(
        base,
        None,  # no single source prompt
        unit_locations=unit_locations,
        source_representations=source_representation.unsqueeze(0),
        output_original_output=output_original_output,  # <-- inject mean activation
    )

    return base_outputs, counterfactual_outputs