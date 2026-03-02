import re
from typing import List, Dict, Iterable
import torch.nn as nn

def get_optim_params(patterns: List[Dict[str, str]], parameters: Iterable) -> List[Dict]:
    """
    Group model parameters based on regex patterns with completeness validation.

    Args:
        patterns: List of dicts, each containing a 'params' key with a regex pattern string
                 and optional additional keys (e.g., 'lr', 'weight_decay').
                 If a parameter name matches multiple patterns, it will be assigned
                 to the group corresponding to the *first* matching pattern in this list.
        parameters: Iterable of model parameters (e.g., model.named_parameters()).

    Returns:
        List of parameter groups, where each group is a dict with 'params' and optional settings.
        Returns an empty list if no parameters require gradients.

    Raises:
        AssertionError: If not all parameters requiring gradients are assigned to a group.

    Example:
        patterns = [
            {'params': r'.*conv.*', 'lr': 0.001},
            {'params': r'.*bn.*', 'lr': 0.0001}
        ]
    """
    # Convert parameters to a list of (name, param) pairs with requires_grad check
    named_params = [(name, param) for name, param in parameters if param.requires_grad]
    if not named_params:
        return [] # Return empty list if no parameters require grad

    param_groups = []
    visited = set()

    # Process each pattern
    for pg in patterns:
        pattern = pg['params']
        # Filter parameters matching the pattern
        matched_params = {
            name: param for name, param in named_params
            if re.search(pattern, name) and name not in visited
        }
        if matched_params:
            # Update the param group with matched parameters
            pg_copy = dict(pg)  # Shallow copy to avoid modifying input
            pg_copy['params'] = list(matched_params.values())
            param_groups.append(pg_copy)
            visited.update(matched_params.keys())

    # Handle unmatched parameters
    all_names = {name for name, _ in named_params}
    if visited != all_names:
        unmatched = all_names - visited
        unmatched_params = [param for name, param in named_params if name in unmatched]
        param_groups.append({'params': unmatched_params})
        visited.update(unmatched)

    # Validate completeness
    assert len(visited) == len(all_names), (
        f"Parameter grouping incomplete: {len(all_names) - len(visited)} parameters "
        f"requiring gradients were not assigned to any group. Total grad params: {len(all_names)}, "
        f"Assigned: {len(visited)}."
    )

    return param_groups

def log_param_groups(param_groups, named_parameters):
    import datetime
    
    # Convert to dict for easier lookup by name, only include params requiring grad
    param_dict = {name: param for name, param in named_parameters if param.requires_grad}
    param_groups_config = {}
    all_matched_param_ids = set()

    for i, group in enumerate(param_groups):
        group_params = group['params']
        lr = group.get('lr', 'default')
        wd = group.get('weight_decay', 'default')

        # Use id to ensure we handle the exact Parameter objects
        group_param_ids = {id(param) for param in group_params}
        # Find names corresponding to these parameter object IDs
        matched_names = [name for name, param in param_dict.items() if id(param) in group_param_ids]

        group_key = f"group_{i}"
        param_groups_config[group_key] = {
            "parameters": sorted(matched_names), # Sort for consistent logging
            "learning_rate": lr,
            "weight_decay": wd,
            "num_parameters": len(matched_names)
        }

        all_matched_param_ids.update(group_param_ids)

    # Check for parameters requiring grad that were not in any group
    all_grad_param_ids = {id(param) for param in param_dict.values()}
    unmatched_param_ids = all_grad_param_ids - all_matched_param_ids
    unmatched_params_names = [name for name, param in param_dict.items() if id(param) in unmatched_param_ids]

    if unmatched_params_names:
        warning_msg = (
            f"Warning: {len(unmatched_params_names)} parameters requiring gradients were not found in any param group. "
            f"Unmatched parameters: {sorted(unmatched_params_names)}"
        )
        print(warning_msg)
        param_groups_config["unmatched_grad_params"] = {
            "parameters": sorted(unmatched_params_names),
            "num_unmatched": len(unmatched_params_names)
        }
    else:
         param_groups_config["unmatched_grad_params"] = {
            "parameters": [],
            "num_unmatched": 0
        }
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"# timestamp: {timestamp}\n")
    print("=" * 80 + "\n\n")
    
    for group_key, config in param_groups_config.items():
        if group_key == "unmatched_grad_params":
            continue
            
        print(f"## {group_key.upper()}\n")
        print(f"(Learning Rate): {config['learning_rate']}\n")
        print(f" (Weight Decay): {config['weight_decay']}\n")
        print(f" (Number of Parameters): {config['num_parameters']}\n")
        print(" (Parameter List):\n")
        
        for param_name in config['parameters']:
           print(f"  - {param_name}\n")
        print("\n" + "-" * 60 + "\n\n")
    
    unmatched_config = param_groups_config["unmatched_grad_params"]
    print("## UNMATCHED PARAMETERS \n")
    print(f"NUMBER OF UNMATCHED PARAMETERS: {unmatched_config['num_unmatched']}\n")
    
    if unmatched_config['parameters']:
        print("UNMATCHED PARAMETERS:\n")
        for param_name in unmatched_config['parameters']:
            print(f"  - {param_name}\n")
    
    print("\n" + "=" * 80 + "\n")
    
    total_trainable_params = sum(p.numel() for p in param_dict.values())
    total_matched_params = sum(config['num_parameters'] for config in param_groups_config.values() if isinstance(config, dict) and 'num_parameters' in config)
    
    print("##  (Statistics)\n")
    print(f"number of total_trainable_params: {total_trainable_params:,}\n")
    print(f"number of total_matched_params : {total_matched_params:,}\n")
    print(f"number of param_groups: {len([k for k in param_groups_config.keys() if k != 'unmatched_grad_params'])}\n")
    print(f"number of the unmatched: {unmatched_config['num_unmatched']}\n")
    print(f"total_trainable_params: {total_trainable_params:,}")
    print(f"total_matched_params: {total_matched_params:,}")
    
    if unmatched_config['num_unmatched'] > 0:
        print(f"⚠️ : {unmatched_config['num_unmatched']} unmatched")
    else:
        print("✓ ")
    
    return param_groups_config


def log_param_groups_to_swanlab(param_groups, named_parameters, swanlab_instance):

    # Convert to dict for easier lookup by name, only include params requiring grad
    param_dict = {name: param for name, param in named_parameters if param.requires_grad}
    param_groups_config = {}
    all_matched_param_ids = set()

    for i, group in enumerate(param_groups):
        group_params = group['params']
        lr = group.get('lr', 'default')
        wd = group.get('weight_decay', 'default')

        # Use id to ensure we handle the exact Parameter objects
        group_param_ids = {id(param) for param in group_params}
        # Find names corresponding to these parameter object IDs
        matched_names = [name for name, param in param_dict.items() if id(param) in group_param_ids]

        group_key = f"group_{i}"
        param_groups_config[group_key] = {
            "parameters": sorted(matched_names), # Sort for consistent logging
            "learning_rate": lr,
            "weight_decay": wd,
            "num_parameters": len(matched_names)
        }

        all_matched_param_ids.update(group_param_ids)

    # Check for parameters requiring grad that were not in any group
    all_grad_param_ids = {id(param) for param in param_dict.values()}
    unmatched_param_ids = all_grad_param_ids - all_matched_param_ids
    unmatched_params_names = [name for name, param in param_dict.items() if id(param) in unmatched_param_ids]

    if unmatched_params_names:
        warning_msg = (
            f"Warning: {len(unmatched_params_names)} parameters requiring gradients were not found in any param group. "
            f"Unmatched parameters: {sorted(unmatched_params_names)}"
        )
        print(warning_msg)
        param_groups_config["unmatched_grad_params"] = {
            "parameters": sorted(unmatched_params_names),
            "num_unmatched": len(unmatched_params_names)
        }
    else:
         param_groups_config["unmatched_grad_params"] = {
            "parameters": [],
            "num_unmatched": 0
        }
    import swanlab

    swanlab_instance.log({"param_groups_config": swanlab.Text(f'{param_groups_config}')})


def count_parameters(model: nn.Module, swanlab_instance=None):
    """
    Calculates and prints the total number of parameters in the model
    and a breakdown by module. Optionally logs the total to SwanLab.

    Note: This counts *all* parameters, including those without gradients.
          The total count is precise and handles shared parameters correctly.
          The breakdown shows parameters for each module that directly owns parameters.

    Args:
        model: The PyTorch model (nn.Module).
        swanlab_instance: Optional SwanLab instance for logging.

    Returns:
        The total number of parameters in the model.
    """
    # Calculate the precise total number of parameters, handling shared parameters
    total_params = sum(p.numel() for p in model.parameters())

    print("\nModel Parameter Breakdown:")
    print("-" * 80)
    cumulative_params_in_breakdown = 0
    
    # Process all modules, including those with children if they have direct parameters
    for name, module in model.named_modules():
        # Count parameters directly belonging to this module (not in submodules)
        params_in_module = sum(p.numel() for p in module.parameters(recurse=False))
        if params_in_module > 0:
            module_type = type(module).__name__
            if name == "":  # Root module
                display_name = f"<root> ({module_type})"
            else:
                display_name = f"{name} ({module_type})"
            print(f"{display_name}: {params_in_module:,} parameters")
            cumulative_params_in_breakdown += params_in_module
    
    print("-" * 80)
    print(f"Total Parameters (Precise): {total_params:,}")
    print(f"Sum of breakdown: {cumulative_params_in_breakdown:,}")
    
    # Check if there's any discrepancy (shouldn't happen now, but good to verify)
    if cumulative_params_in_breakdown != total_params:
        print(f"⚠️  Discrepancy detected: {total_params - cumulative_params_in_breakdown:,} parameters")
        print("   This might indicate shared parameters or other complex parameter sharing.")
    else:
        print("✓ All parameters accounted for in breakdown")
    
    print("-" * 80)

    # Log the precise total to SwanLab if instance provided
    if swanlab_instance:
        swanlab_instance.log({
            "model_params": {
                "total_parameters": total_params,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        })
    return total_params
        
        
        