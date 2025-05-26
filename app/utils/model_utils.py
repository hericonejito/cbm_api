import os
import math
import torch
# import app.clip
from app import clip
from app.utils import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

PM_SUFFIX = {"max": "_max", "avg": ""}


def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    For convolutional layers (4D tensors), apply pooling.
    For fully connected layers (2D tensors), use as is.
    '''
    if mode == 'avg':
        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())
    elif mode == 'max':
        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())
    return hook


def get_forward_layers(model, device, input_size):
    layer_names = []
    handles = []

    def get_activation(name):
        def hook(module, input, output):
            layer_names.append(name)

        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        handle = module.register_forward_hook(get_activation(name))
        handles.append(handle)

    # Run a forward pass with a dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return layer_names


def get_module_by_name(model, access_string):
    """
    Retrieve a module nested in a model by its access string.

    Args:
        model (torch.nn.Module): The model.
        access_string (str): The access string, e.g., "layer1.0.conv1".

    Returns:
        torch.nn.Module: The module.
    """
    names = access_string.split('.')
    module = model
    for name in names:
        if name.isdigit():
            module = module[int(name)]
        else:
            module = getattr(module, name)
    return module


def save_target_activations(target_model, dataset, save_name, target_layers=["layer4"], batch_size=1000,
                            device="cuda", pool_mode='avg'):
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in target_layers:
        layer_name = target_layer if isinstance(target_layer, str) else 'second_to_last'
        save_names[layer_name] = save_name.format(layer_name + PM_SUFFIX.get(pool_mode, ''))

    if _all_saved(save_names):
        return

    # Define get_activation function accessible to both blocks
    def get_activation(outputs, mode):
        '''
        mode: how to pool activations: one of avg, max
        For convolutional layers (4D tensors), apply pooling.
        For fully connected layers (2D tensors), use as is.
        '''
        if mode == 'avg':
            def hook(model, input, output):
                if len(output.shape) == 4:
                    outputs.append(output.mean(dim=[2, 3]).detach().cpu())
                else:
                    outputs.append(output.detach().cpu())

            return hook
        elif mode == 'max':
            def hook(model, input, output):
                if len(output.shape) == 4:
                    outputs.append(output.amax(dim=[2, 3]).detach().cpu())
                else:
                    outputs.append(output.detach().cpu())

            return hook
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    if 'second_to_last' in target_layers:
        # Handle dynamic layer extraction
        input_size = dataset[0][0].shape
        execution_order = []

        # Collect only leaf modules during forward pass
        def collect_leaf_modules_hook(module_name):
            def hook(module_, input, output):
                # Check if the module is a leaf module (no children)
                if len(list(module_.children())) == 0:
                    execution_order.append((module_name, module_, output))

            return hook

        # Register hooks only on leaf modules
        handles = []
        for name, module in target_model.named_modules():
            handle = module.register_forward_hook(collect_leaf_modules_hook(name))
            handles.append(handle)

        # Run a forward pass with a dummy input to collect execution order
        dummy_input = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            target_model(dummy_input)

        # Remove hooks
        for handle in handles:
            handle.remove()

        if len(execution_order) < 2:
            raise ValueError("Model does not have enough leaf layers.")

        # Identify the second-to-last leaf module
        second_to_last_name, second_to_last_module, _ = execution_order[-2]
        print(f"Second-to-last layer identified: {second_to_last_name} - {second_to_last_module}")

        # Register hook on the second-to-last module to collect activations
        all_features = []
        hook = second_to_last_module.register_forward_hook(get_activation(all_features, pool_mode))

        # Data pass to collect activations
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                _ = target_model(images.to(device))

        # Save the activations
        torch.save(torch.cat(all_features), save_names['second_to_last'])
        hook.remove()

        # Free memory
        del all_features
        torch.cuda.empty_cache()
        return

    else:
        # Handle the case where target_layers are specified by name
        all_features_dict = {target_layer: [] for target_layer in target_layers}

        hooks = {}
        for target_layer in target_layers:
            module = get_module_by_name(target_model, target_layer)
            if module is None:
                raise ValueError(f"Module {target_layer} not found in model.")
            hook = module.register_forward_hook(get_activation(all_features_dict[target_layer], pool_mode))
            hooks[target_layer] = hook

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                _ = target_model(images.to(device))

        # Save the activations
        for target_layer in target_layers:
            save_path = save_names[target_layer]
            torch.save(torch.cat(all_features_dict[target_layer]), save_path)
            hooks[target_layer].remove()

        # Free memory
        del all_features_dict
        torch.cuda.empty_cache()
        return


def save_clip_image_features(model, dataset, save_name, batch_size=1000, device="cuda"):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return

    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    # free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text) / batch_size))):
            text_features.append(model.encode_text(text[batch_size * i:batch_size * (i + 1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return


def save_activations(clip_name, target_name, target_layers, d_probe, dataset_root,
                     concept_set, batch_size, device, pool_mode, save_dir):
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name,
                                                                      "{}", d_probe, concept_set,
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)

    if _all_saved(save_names):
        return

    clip_model, clip_preprocess = clip.load(clip_name, device=device)

    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    # setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess, dataset_root)
    data_t = data_utils.get_data(d_probe, target_preprocess, dataset_root)

    with open(concept_set, 'r') as f:
        words = (f.read()).split('\n')
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)

    save_clip_text_features(clip_model, text, text_save_name, batch_size)

    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)

    return


def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn,
                                    return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()

    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)

    del clip_feats
    torch.cuda.empty_cache()

    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity


def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                     PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))

    return target_save_name, clip_save_name, text_save_name


def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True


def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                          pin_memory=True)):
        with torch.no_grad():
            # outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu() == labels)
            total += len(labels)
    return correct / total


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                          pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred = []
    for i in range(torch.max(pred) + 1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds == i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred