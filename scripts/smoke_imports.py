#!/usr/bin/env python3
import argparse
import importlib
import sys

CORE_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "plotly",
    "yaml",
    "pydantic",
    "optuna",
    "tensorflow",
    "tensorflow_probability",
]

PROJECT_IMPORTS = [
    "profile_manager",
    "hpo.optuna_utils",
    "models.PINNs",
    "forecast_pipeline",
]

DARTS_IMPORTS = [
    "torch",
    "darts",
    "pytorch_lightning",
]

def check(modname: str) -> bool:
    try:
        mod = importlib.import_module(modname)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {modname} ({version})")
        return True
    except Exception as e:
        print(f"[FAIL] {modname}: {e}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-darts", action="store_true")
    args = parser.parse_args()

    ok = True

    print("=== CORE IMPORTS ===")
    for mod in CORE_IMPORTS:
        ok = check(mod) and ok

    print("\n=== PROJECT IMPORTS ===")
    for mod in PROJECT_IMPORTS:
        ok = check(mod) and ok

    if args.with_darts:
        print("\n=== DARTS IMPORTS ===")
        for mod in DARTS_IMPORTS:
            ok = check(mod) and ok

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
