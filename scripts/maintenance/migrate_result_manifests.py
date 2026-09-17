"""Apply a reviewed manifest backfill plan; preserve all existing report bytes.

A plan is a JSON array. HF entries contain kind='hf', repo, branch,
manifest_path, manifest, and source. Local entries contain kind='local', repo,
branch, report_path, report_sha256, manifest, and source. Entries with
kind='no-report' are recorded as skipped. See sbm/auditing/RESULTS_FORMAT.md.

    uv run python scripts/maintenance/migrate_result_manifests.py \
        --plan /tmp/ir-manifest-migration/plan.json --output migration.json --apply

Without --apply, validate and print the proposed changes. HF metadata commits
only add manifest.json and use an expected parent SHA. Local publications use
the shared two-file publisher. Every commit is verified by content hash/OID.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from sbm.auditing.manifest import load_manifest, validate_manifest
from sbm.auditing.publish import publish_bundle, _matches


def report_entry(api, repo, revision, filename='analysis/report.json'):
    entries = api.get_paths_info(repo, [filename], repo_type='dataset', revision=revision)
    if len(entries) != 1 or not hasattr(entries[0], 'size'):
        raise ValueError(f'No report at {repo}@{revision}:{filename}')
    return entries[0]


def oid(entry):
    return entry.lfs.sha256 if entry.lfs else entry.blob_id


def migrate(item, api):
    repo, branch = item['repo'], item['branch']
    metadata = validate_manifest(item['manifest'])
    if item['kind'] == 'local':
        path = Path(item['report_path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['report_sha256']:
                raise ValueError(f'Local report changed after the plan was prepared: {path}')
        if load_manifest(path.with_name('manifest.json')) != metadata:
            raise ValueError('Local metadata changed after plan preparation')
        commit = publish_bundle(path.with_name('manifest.json'), repo, branch, create_branch=True)
        revision = commit.oid if commit else api.repo_info(repo, repo_type='dataset', revision=branch).sha
        after = report_entry(api, repo, revision)
        if not _matches(path, after):
            raise AssertionError('Uploaded report differs from local report')
        before_oid = item['report_sha256']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['report_sha256']:
                raise ValueError('Local report changed during publication; publish its new snapshot separately')
    else:
        # Resolve the latest head immediately before writing: only manifest.json
        # changes, even if an active analyzer advanced the report since inventory.
        parent = api.repo_info(repo, repo_type='dataset', revision=branch).sha
        before = report_entry(api, repo, parent)
        before_oid = oid(before)
        manifest_path = Path(item['manifest_path'])
        if load_manifest(manifest_path) != metadata:
            raise ValueError('Manifest changed after plan preparation')
        existing = api.get_paths_info(repo, ['analysis/manifest.json'], repo_type='dataset', revision=parent)
        if existing:
            previous = load_manifest(Path(hf_hub_download(repo, 'analysis/manifest.json', repo_type='dataset', revision=parent)))
            if previous != metadata:
                raise ValueError(f'Existing manifest differs at {repo}@{branch}; review before replacing')
            revision = parent
            commit = None
        else:
            commit = api.create_commit(
                repo_id=repo, repo_type='dataset', revision=branch, parent_commit=parent,
                commit_message='Add self-describing results manifest (report unchanged)',
                operations=[CommitOperationAdd(path_in_repo='analysis/manifest.json', path_or_fileobj=str(manifest_path))],
            )
            revision = commit.oid
        after = report_entry(api, repo, revision)
        if oid(after) != before_oid:
            raise AssertionError('Metadata migration changed the report content OID')
    published = load_manifest(Path(hf_hub_download(repo, 'analysis/manifest.json', repo_type='dataset', revision=revision)))
    if published != metadata:
        raise AssertionError('Published manifest differs from planned metadata')
    return {'source':item['source'], 'repo':repo, 'branch':branch, 'status':'verified',
            'revision':revision, 'url':f'https://huggingface.co/datasets/{repo}/tree/{branch}',
            'canonical_path':'/'+'/'.join(metadata[k]['id'] for k in ('technique','project','experiment')),
            'report_oid_before':before_oid, 'report_oid_after':oid(after),
            'report_bytes':after.size, 'new_commit':commit is not None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for item in plan:
        if item['kind'] != 'no-report': validate_manifest(item['manifest'])
    if not args.apply:
        for item in plan: print(item['kind'], item['repo'], item['branch'])
        return
    api=HfApi()
    results=[]
    for item in plan:
        if item['kind']=='no-report':
            result={'source':item['source'], 'repo':item['repo'], 'branch':item['branch'], 'status':'no-report'}
        else:
            try: result=migrate(item,api)
            except Exception as error:
                result={'source':item['source'], 'repo':item['repo'], 'branch':item['branch'], 'status':'error', 'error':str(error)}
        results.append(result)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(result['status'],item['source'],flush=True)
    if any(r['status']=='error' for r in results): raise SystemExit(1)


if __name__=='__main__': main()
