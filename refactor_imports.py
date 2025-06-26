#!/usr/bin/env python3
"""
refactor_imports.py

This script scans through .py and .ipynb files in the project directory and refactors import statements
according to a mapping dictionary. For example, it can change:
    from src.data.data_loading import load_volve
to:
    from src.data.data_loading import load_volve

Before running, please back up your project files!
"""

import os
import re
import json

# Define a dictionary mapping old module names (as regex patterns) to new module names.
# Adjust these patterns as needed.
IMPORT_MAPPING = {
    # Python scripts or modules
    r'^(\s*from\s+)data_loading(\s+import\s+)': r'\1src.data.data_loading\2',
    r'^(\s*from\s+)data_preparation(\s+import\s+)': r'\1src.data.data_preparation\2',
    r'^(\s*from\s+)evaluation(\s+import\s+)': r'\1src.evaluation.evaluation\2',
    r'^(\s*from\s+)models_forecast(\s+import\s+)': r'\1src.training.models_forecast\2',
    r'^(\s*from\s+)train_utils(\s+import\s+)': r'\1src.training.train_utils\2',
    r'^(\s*from\s+)prediction_utils(\s+import\s+)': r'\1src.prediction.prediction_utils\2',
    r'^(\s*from\s+)utilities(\s+import\s+)': r'\1src.utils.utilities\2',
    # Sometimes imports are written as "import module"
    r'^(\s*import\s+)data_loading\b': r'\1src.data.data_loading',
    r'^(\s*import\s+)data_preparation\b': r'\1src.data.data_preparation',
    r'^(\s*import\s+)evaluation\b': r'\1src.evaluation.evaluation',
    r'^(\s*import\s+)models_forecast\b': r'\1src.training.models_forecast',
    r'^(\s*import\s+)train_utils\b': r'\1src.training.train_utils',
    r'^(\s*import\s+)prediction_utils\b': r'\1src.prediction.prediction_utils',
    r'^(\s*import\s+)utilities\b': r'\1src.utils.utilities',
}

# List of file extensions to update.
FILE_EXTENSIONS = ['.py', '.ipynb']

def refactor_line(line: str) -> str:
    """
    Apply all regex substitutions to a single line.
    """
    for pattern, replacement in IMPORT_MAPPING.items():
        new_line = re.sub(pattern, replacement, line)
        if new_line != line:
            line = new_line
    return line

def process_py_file(filepath: str):
    """
    Process a Python file: update its import lines and rewrite the file if changes are found.
    """
    print(f"Processing Python file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = [refactor_line(line) for line in lines]

    if updated_lines != lines:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes in: {filepath}")

def process_ipynb_file(filepath: str):
    """
    Process a Jupyter Notebook: update import lines in code cells and rewrite the file if changes are found.
    """
    print(f"Processing Notebook file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            new_source = []
            for line in cell.get("source", []):
                new_line = refactor_line(line)
                if new_line != line:
                    changed = True
                new_source.append(new_line)
            cell["source"] = new_source

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes in: {filepath}")

def should_process_file(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in FILE_EXTENSIONS)

def main():
    root_dir = os.getcwd()  # or set to your project root
    for subdir, _, files in os.walk(root_dir):
        # Skip directories like __pycache__ if desired.
        if '__pycache__' in subdir:
            continue

        for file in files:
            if should_process_file(file):
                filepath = os.path.join(subdir, file)
                if file.endswith('.py'):
                    process_py_file(filepath)
                elif file.endswith('.ipynb'):
                    process_ipynb_file(filepath)

if __name__ == "__main__":
    main()
