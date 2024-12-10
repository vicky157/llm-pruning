import logging
import math
import os
import sys
from copy import deepcopy
import shutil
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from .io import create_dir
from .utils import prepare_calibration_input, print_gpu_memory, auto_map, CUSTOM_FILE
from .wrapper import HiddenStatesRecordWrapper

logger = logging.getLogger(__name__)


def get_block_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device

    if cache_file is not None and os.path.exists(cache_file):
        # use cached file
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device)

    else:
        # calculate similarities
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
        unwrapped_model.config.use_cache = False
        layers = unwrapped_model.model.layers

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
        num_layers = unwrapped_model.config.num_hidden_layers
        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [ [0.5],  [0.5],  [0.5],  [0.5],  [0.5],  [0.5]]  # shape(6, 1)
        similarities = torch.full((len(layers), 1), -math.inf, device=device)

        accelerator.print('Starting ...')
        dtype = torch.float32

        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            # Wrap layer
            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)  # 🔍 Wrap layer

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

            input_hidden_states = torch.cat(wrapped_layer.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(wrapped_layer.output_hidden_states, dim=0).to(dtype).to(device)
            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
            cos_sim = cos_sim.mean()
            cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
            similarities[i, 0] = cos_sim
            layer.to("cpu")  # 🔍

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities

@no_grad()
def get_block_similarities_consecutive(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device

    if cache_file is not None and os.path.exists(cache_file):
        # use cached file
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device)

    else:
        # calculate similarities
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
        unwrapped_model.config.use_cache = False
        layers = unwrapped_model.model.layers

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

        # 🔍 Get layer ids
        num_layers = unwrapped_model.config.num_hidden_layers
        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        #           [ 0.5,  0.5,  0.5,  0.5,  0.5, -inf],
        #           [ 0.5,  0.5,  0.5,  0.5, -inf, -inf],
        #           [ 0.5,  0.5,  0.5, -inf, -inf, -inf],
        #           [ 0.5,  0.5, -inf, -inf, -inf, -inf],
        #           [ 0.5, -inf, -inf, -inf, -inf, -inf]]  # shape(6, 6)
        similarities = torch.full((len(layers), len(layers)), -math.inf, device=device)
        accelerator.print('Starting ...')
        wrapped_layers = []

        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            # Wrap layer
            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=(i == len(layers) - 1))  # 🔍 Wrap layer
            wrapped_layers.append(wrapped_layer)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                if getattr(unwrapped_model.config, "model_type", None) == "llama":
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j], cache_position=cache_position[j])[0]
                else:
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

        dtype = torch.float32
        all_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_hidden_states.append(torch.cat(wrapped_layers[i].input_hidden_states, dim=0).to(dtype))  # (total_token_num, hidden_size)
        all_hidden_states.append(torch.cat(wrapped_layers[-1].output_hidden_states, dim=0).to(dtype))
        accelerator.print(f'Total {len(all_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            for j in range(i + 1, len(all_hidden_states)):
                packed_hidden_states_layer_i = all_hidden_states[i].to(device)
                packed_hidden_states_layer_j = all_hidden_states[j].to(device)
                index_gap = j - i

                cos_sim = F.cosine_similarity(packed_hidden_states_layer_i, packed_hidden_states_layer_j, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices

                similarities[i, index_gap - 1] = cos_sim

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities


def max_with_tolerance(similarities: torch.tensor, tolerance: float):
    max_value, _ = torch.max(similarities, dim=0)
    close_indices = torch.where(torch.abs(similarities - max_value) < tolerance)[0]
    begin_layer_id = close_indices[0]

    return max_value, begin_layer_id


def get_top_k(similarities, k, tolerance):
    dropped_layer_list = []
    dropped_sim_list = []
    for _ in range(k):
        max_value, max_index = max_with_tolerance(similarities, tolerance=tolerance)
        dropped_layer_list.append(max_index.item())
        dropped_sim_list.append(max_value.item())
        similarities[max_index] = 0
    return dropped_sim_list, dropped_layer_list

def cluster_layers(similarities: torch.tensor, num_clusters: int = 6):
    """Cluster layers using k-means based on their similarity scores."""
    similarities = similarities.numpy()  # Convert to numpy for sklearn
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(similarities)
    return kmeans.labels_


def get_block_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    """ Get block similarities. """
    device = accelerator.device
    if cache_file is not None and os.path.exists(cache_file):
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device)
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.use_cache = False
        layers = unwrapped_model.model.layers
        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)
        num_layers = unwrapped_model.config.num_hidden_layers
        similarities = torch.full((len(layers), 1), -math.inf, device=device)
        accelerator.print('Starting ...')

        dtype = torch.float32
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]
            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)
            handle = layer.register_forward_hook(record_states_hook)

            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

            input_hidden_states = torch.cat(wrapped_layer.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(wrapped_layer.output_hidden_states, dim=0).to(dtype).to(device)
            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
            cos_sim = cos_sim.mean()
            cos_sim = accelerator.reduce(cos_sim, reduction="mean")
            similarities[i, 0] = cos_sim
            layer.to("cpu")

        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
            accelerator.wait_for_everyone()

    return similarities


def pruning_strategy(similarities: torch.tensor, num_clusters: int = 6):
    """ Apply pruning strategy based on k-means clustering. """
    cluster_labels = cluster_layers(similarities, num_clusters)
    group_1, group_2, group_3, group_4, group_5, group_6 = [], [], [], [], [], []

    # Assign clusters to different groups
    for i, label in enumerate(cluster_labels):
        if label == 0:
            group_1.append(i)
        elif label == 1:
            group_2.append(i)
        elif label == 2:
            group_3.append(i)
        elif label == 3:
            group_4.append(i)
        elif label == 4:
            group_5.append(i)
        else:
            group_6.append(i)

    # Keep only one layer from each of Group 1, Group 2, and Group 3
    kept_layer_list = []

    if group_1:
        kept_layer_list.append(group_1[0])  # Keep the first layer in group 1
    if group_2:
        kept_layer_list.append(group_2[0])  # Keep the first layer in group 2
    if group_3:
        kept_layer_list.append(group_3[0])  # Keep the first layer in group 3

    # Keep the first 2 and last 2 layers explicitly (must-keep condition)
    kept_layer_list = kept_layer_list + [0, 1, -2, -1]

    # Drop all layers from groups 1, 2, and 3 except the kept ones
    dropped_layer_list = [layer for layer in group_1 + group_2 + group_3 if layer not in kept_layer_list]

    # Group 4, 5, and 6 layers are untouched, so include them in the kept list
    kept_layer_list += group_4 + group_5 + group_6

    return kept_layer_list, dropped_layer_list

def discrete_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a discrete order based on similarity ranges.
    - Group 1: Clustered with k-means, keep one layer from each of Groups 1, 2, and 3
    - Groups 4, 5, and 6: Keep all layers
    - Keep the first 2 and last 2 layers as a must condition
    """
    drop_n = args.drop_n

    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    # Apply pruning strategy based on k-means clustering
    kept_layer_list, dropped_layer_list = pruning_strategy(similarities)

    # Log the kept and dropped layers
    accelerator.print(f"Kept layers: {kept_layer_list}")
    accelerator.print(f"Dropped layers: {dropped_layer_list}")

    return dropped_layer_list

def consecutive_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a consecutive order based on similarity ranges.
    - Group 1: Clustered with k-means, keep one layer from each of Groups 1, 2, and 3
    - Groups 4, 5, and 6: Keep all layers
    - Keep the first 2 and last 2 layers as a must condition
    """
    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    # Apply pruning strategy based on k-means clustering
    kept_layer_list, dropped_layer_list = pruning_strategy(similarities)

    # Log the kept and dropped layers
    accelerator.print(f"Kept layers: {kept_layer_list}")
    accelerator.print(f"Dropped layers: {dropped_layer_list}")

    return dropped_layer_list


def post_block_drop(prune_model_save_path, model, tokenizer, reserved_layer_list, accelerator: Accelerator, only_update_config=False):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first

    if accelerator.is_main_process:
        out_cfg = deepcopy(unwrapped_model.config)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type in auto_map:
            out_cfg.auto_map = auto_map[model_type]
        else:
            raise ValueError("Unsupported model type!")

        dropped_attn_list = dropped_mlp_list = list(set(list(range(out_cfg.num_hidden_layers))) - set(reserved_layer_list))
        out_cfg.drop_mlp_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_mlp_list', [])) if v] + dropped_mlp_list
        out_cfg.drop_attn_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_attn_list', [])) if v] + dropped_attn_list

        accelerator.print(f"Dropped attention list: {dropped_attn_list}")
        accelerator.print(f"Dropped MLP list: {dropped_mlp_list}")

        accelerator.print("Saving...")
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["config"], prune_model_save_path)
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["model"], prune_model_save_path)
        if not only_update_config:
            model.save_pretrained(prune_model_save_path)
            tokenizer.save_pretrained(prune_model_save_path)
        out_cfg.save_pretrained(prune_model_save_path)