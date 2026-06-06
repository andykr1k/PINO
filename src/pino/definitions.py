import os
from pathlib import Path

def validate_path_env_var(var_name: str):
    path_str = os.getenv(var_name, None)
    if not path_str:
        raise ValueError(f"Expected path set by environment variable {var_name}")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    return path

INTERIOR_GS_PATH = validate_path_env_var("INTERIOR_GS_PATH")