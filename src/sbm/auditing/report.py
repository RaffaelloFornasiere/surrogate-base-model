"""Technique-independent reports compatible with AO Visualizer schema v1."""

import hashlib
import json
from math import sqrt
from pathlib import Path

from .manifest import validate_manifest, write_manifest


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


class Report:
    def __init__(self, run_name: str, config: dict, *, input_kind: str,
                 hf_repo: str = "", repo_commit: str | None = None,
                 manifest: dict | None = None):
        self.manifest = validate_manifest(manifest) if manifest is not None else None
        self.data = {
            "schema_version": 1, "run_name": run_name, "hf_repo": hf_repo,
            "repo_commit": repo_commit, "input_kind": input_kind,
            "config": config, "texts": {}, "runs": [],
        }
        self._paths = set()

    def text(self, value: str) -> str:
        identity = hashlib.sha256(value.encode()).hexdigest()
        self.data["texts"][identity] = value
        return identity

    def add(self, *, path: str, model: str, quirk: str, run_index: int,
            combo: dict, hypothesis: dict, ground_truth: str, judges: dict,
            control_judges: dict | None = None, sampled_context_ids: list | None = None):
        if path in self._paths:
            raise ValueError(f"Duplicate report path: {path}")
        self._paths.add(path)
        conversation = hypothesis["investigator"]
        messages = conversation["messages"]
        inv = {k: v for k, v in conversation.items() if k != "messages"}
        inv.update(
            system_id=self.text("\n\n".join(m["content"] for m in messages if m["role"] == "system")),
            input_id=self.text("\n\n".join(m["content"] for m in messages if m["role"] == "user")),
        )
        self.data["runs"].append({
            "path": path, "model": model, "quirk": quirk, "run_index": run_index,
            "combo": {k: [str(v)] if not isinstance(v, list) else [str(x) for x in v]
                      for k, v in combo.items()},
            "input_kind": self.data["input_kind"],
            "sampled_context_ids": sampled_context_ids or [],
            "identified_quirk": hypothesis["quirk"],
            "identified_description": hypothesis["description"],
            "ground_truth": ground_truth, "judges": judges,
            "control_judges": control_judges or {}, "investigator": inv,
        })

    def save(self, path: Path):
        metadata = validate_manifest({**self.manifest, "report": path.name}) if self.manifest else None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, allow_nan=False))
        temporary.replace(path)
        if metadata:
            write_manifest(path.with_name("manifest.json"), metadata)
