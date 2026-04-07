# In hpo/profile_generators.py

import yaml
from itertools import product
from pathlib import Path

def _load_architecture_profiles(yaml_path: str | Path) -> list:
    """Helper to load and sort architecture profile names."""
    with open(yaml_path, 'r') as f:
        archs = yaml.safe_load(f)
    return sorted(list(archs.keys()))

def generate_grid_search_profile(
    architectures_yaml: str | Path,
    output_file: str | Path,
    **sweep_params: dict
):
    """
    Generates a full Cartesian product (grid search) profile and saves it as a YAML file.

    Args:
        architectures_yaml: Path to the architectures definition file.
        output_file: Path to save the generated YAML sweep profile.
        **sweep_params: A dictionary where keys are parameter names (e.g., 'lag_window')
                        and values are lists of values to sweep over.
    """
    arch_profiles = _load_architecture_profiles(architectures_yaml)
    
    # Prepare the parameter lists for the product
    param_names = ['architecture_profile'] + list(sweep_params.keys())
    param_lists = [arch_profiles] + list(sweep_params.values())
    
    experiments = []
    for values in product(*param_lists):
        # Create a dictionary for the current combination
        exp = dict(zip(param_names, values))
        
        # Create a descriptive experiment_id
        exp_id_parts = [f"{k}{v}" for k, v in exp.items() if k != 'architecture_profile']
        exp["experiment_id"] = f"{exp['architecture_profile']}_{'_'.join(exp_id_parts)}"
        
        experiments.append(exp)

    # Save the generated experiments to the output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(experiments, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"Generated {len(experiments)} grid search experiments and saved to '{output_path}'")