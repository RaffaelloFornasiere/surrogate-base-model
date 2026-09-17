"""Results discovery metadata. The public contract is RESULTS_FORMAT.md."""

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("manifest.schema.json")
_validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def validate_manifest(value: dict) -> dict:
    """Validate without mutating producer configuration; return an owned copy."""
    errors = sorted(_validator.iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.path)) or "manifest"
        raise ValueError(f"Invalid results manifest at {location}: {error.message}")
    if value["report"] == "manifest.json":
        raise ValueError("The report must be a separate file from manifest.json")
    return copy.deepcopy(value)


def load_manifest(path: Path) -> dict:
    return validate_manifest(json.loads(path.read_text()))


def write_manifest(path: Path, metadata: dict) -> Path:
    data = validate_manifest(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return path
