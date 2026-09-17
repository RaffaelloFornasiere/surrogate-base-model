"""Publish an existing results bundle; never runs an investigator or judge.

    PYTHONPATH=src uv run python -m sbm.auditing.publish path/to/manifest.json \
        --repo surrogate-base-model/oracle-results --branch exp06-selfie --create-branch

The manifest/report are one HF commit, protected against concurrent updates.
Only the two named files are uploaded; other branch contents are retained.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

from .manifest import load_manifest


def _matches(local: Path, remote) -> bool:
    """Compare HF LFS SHA256 or Git blob SHA1 without downloading the report."""
    if local.stat().st_size != remote.size:
        return False
    digest = hashlib.sha256() if remote.lfs else hashlib.sha1(f"blob {remote.size}\0".encode())
    with local.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == (remote.lfs.sha256 if remote.lfs else remote.blob_id)


def publish_bundle(manifest_path: Path, repo: str, branch: str, *, create_branch=False):
    metadata = load_manifest(manifest_path)
    report_path = manifest_path.parent / metadata["report"]
    data = json.loads(report_path.read_text())
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("A v1 manifest requires a report with schema_version 1")
    if not isinstance(data.get("runs"), list):
        raise ValueError("Report must contain a runs array")
    del data
    api = HfApi()
    try:
        parent = api.repo_info(repo, revision=branch, repo_type="dataset").sha
    except RevisionNotFoundError:
        if not create_branch:
            raise ValueError(f"Branch {branch!r} is missing; pass --create-branch")
        api.create_branch(repo, branch=branch, repo_type="dataset", exist_ok=False)
        parent = api.repo_info(repo, revision=branch, repo_type="dataset").sha
    # Refuse an accidental canonical-identity change when replacing a bundle.
    try:
        previous_path = hf_hub_download(repo, "analysis/manifest.json", repo_type="dataset", revision=parent)
    except EntryNotFoundError:
        previous_path = None
    if previous_path:
        previous = load_manifest(Path(previous_path))
        if any(previous[k]["id"] != metadata[k]["id"] for k in ("technique", "project", "experiment")):
            raise ValueError("Existing branch belongs to a different experiment")
    files = {"analysis/manifest.json": manifest_path, f"analysis/{metadata['report']}": report_path}
    remote = {item.path: item for item in api.get_paths_info(repo, list(files), repo_type="dataset", revision=parent)}
    changed = {name: path for name, path in files.items() if name not in remote or not _matches(path, remote[name])}
    if not changed:
        return None
    return api.create_commit(
        repo_id=repo, repo_type="dataset", revision=branch, parent_commit=parent,
        commit_message=f"Publish results bundle: {metadata['experiment']['id']}",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in changed.items()],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--create-branch", action="store_true")
    args = parser.parse_args()
    commit = publish_bundle(args.manifest, args.repo, args.branch, create_branch=args.create_branch)
    print(commit.commit_url if commit else "Already up to date")


if __name__ == "__main__":
    main()
