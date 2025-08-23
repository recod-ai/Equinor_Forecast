# run_campaign.py

import sys
from pathlib import Path
import pandas as pd

# This script is run from the project root, so we need to add 'src' to the path
# to make our package importable.
# This makes the script runnable from anywhere without installation.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Now we can import from our source package
try:
    from hpo.hpo_runner import run_campaign_from_config_file
    from config_loader import load_campaign_config
except ImportError as e:
    print("Error: Could not import necessary modules from the 'src' directory.")
    print(f"Please ensure '{SRC_PATH}' is a valid source directory.")
    print(f"Details: {e}")
    sys.exit(1)


def main():
    """
    The main entry point for running an HPO campaign from the command line.

    It expects a single command-line argument: the path to the YAML campaign config file.
    """
    # 1. --- Argument Parsing ---
    if len(sys.argv) != 2:
        # Provide a helpful usage message if the argument is missing.
        print(f"Usage: python {Path(sys.argv[0]).name} <path_to_campaign_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    # 2. --- Core Logic Execution ---
    # The script's main job is to call the well-defined core function.
    print(f"--- Starting HPO Campaign from: {config_path} ---")
    master_leaderboard = run_campaign_from_config_file(config_path)

    # 3. --- Final Output ---
    if master_leaderboard is not None and not master_leaderboard.empty:
        try:
            # Reload the config to get the metric for sorting (cleanest way)
            config = load_campaign_config(config_path)
            metric = config.hpo_params.metric_to_optimize
            
            print("\n--- Master Leaderboard (Top 10 Overall) ---")
            # Use pandas' to_string() for clean console output.
            print(master_leaderboard.sort_values(by=metric).head(10).to_string())
        except (KeyError, FileNotFoundError):
            print("\n--- Master Leaderboard (Top 10 - Unsorted) ---")
            print(master_leaderboard.head(10).to_string())
    else:
        print("\n--- Campaign finished, but no results were generated. ---")

    print(f"--- HPO Campaign from: {config_path} has finished. ---")


if __name__ == "__main__":
    # This standard construct makes the script executable.
    main()