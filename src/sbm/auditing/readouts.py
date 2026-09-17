"""Shared CLI for tabular patchscope/SelfIE readouts.

The experiment supplies its prompt, quirk descriptions, and regex terms.
The lower-level auditing API also accepts AO or steering text directly.
"""

import argparse
import gzip
import json
import random
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pandas as pd

from .batch import read_jsonl, run_batch
from .llm import JUDGE_PROMPT, LLMConfig, investigate, judge, make_client
from .report import Report, fingerprint, wilson

CELL = ["reader", "source", "reference", "target", "layer"]
SEED = 42
MODEL = "gemini-3-flash-preview"
# Investigator at temperature 1 (runs must differ); judge at temperature 0 (one verdict per hypothesis).
INVESTIGATOR_CONFIG = LLMConfig(MODEL, temperature=1.0, max_tokens=400, reasoning_effort="none")
JUDGE_CONFIG = LLMConfig(MODEL, temperature=0.0, max_tokens=300, reasoning_effort="none")


@dataclass
class Experiment:
    directory: Path
    input_kind: str
    data_directory: str
    build_prompt: Callable
    quirks: dict[str, str]
    terms: dict[str, str]
    family_of: Callable
    results: dict | None = None


def client():
    return make_client(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                       api_key_env="GOOGLE_AI_STUDIO_API_KEY")


def load_readouts(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for index, line in enumerate(f):
                row = json.loads(line)
                row["tokens_text"] = " ".join(row.pop("top_tokens", []))
                row.pop("top_probs", None)
                row["readout_id"] = f"{path.name}:{index}"
                rows.append(row)
    if not rows:
        raise ValueError(f"No readouts in {directory}")
    df = pd.DataFrame(rows)
    df["reference"] = df["reference"].fillna("")
    return df


def key(row: dict) -> tuple:
    return tuple(row[k] for k in CELL) + (row["run"],)


def pick_sample(df: pd.DataFrame, cell: dict, run: int, sample: int) -> pd.DataFrame:
    # Preserve the published experiments' seed, including its omission of target.
    rng = random.Random(f"{SEED}-{cell['reader']}-{cell['source']}-{cell['reference']}-{cell['layer']}-{run}")
    return df.iloc[sorted(rng.sample(range(len(df)), min(sample, len(df))))]


def investigation_tasks(experiment: Experiment, df: pd.DataFrame, args):
    groups = list(df[df.reader != "unpatched"].groupby(CELL, dropna=False))
    if args.cells:
        groups = groups[:args.cells]
    for values, group in groups:
        cell = dict(zip(CELL, values))
        for run in range(args.runs):
            pick = pick_sample(group, cell, run, args.sample)
            prompt = experiment.build_prompt(pick)
            yield {**cell, "run": run, "prompt": prompt,
                   "sampled_readout_ids": pick.readout_id.tolist(),
                   "fingerprint": fingerprint([prompt, asdict(INVESTIGATOR_CONFIG)])}


def run_investigator(experiment: Experiment, df: pd.DataFrame, args):
    get_client = cache(client)

    def work(task):
        result = investigate(get_client(), task["prompt"], INVESTIGATOR_CONFIG)
        return {**{k: v for k, v in task.items() if k != "prompt"}, **result}

    return run_batch(investigation_tasks(experiment, df, args), args.out / "investigator.jsonl",
                     key=key, work=work, workers=args.workers, overwrite=args.overwrite)


def run_judges(experiment: Experiment, args):
    hypotheses = read_jsonl(args.out / "investigator.jsonl")
    if not hypotheses:
        raise ValueError("No hypotheses; run investigate first")
    get_client = cache(client)
    current = {key(h): (h["quirk"], h["description"]) for h in hypotheses}
    if not args.overwrite:
        for previous in read_jsonl(args.out / "judge.jsonl"):
            if key(previous) not in current or current[key(previous)] != (previous["quirk"], previous["description"]):
                raise ValueError("Cached judges refer to different hypotheses; use a new output directory or judge --overwrite")

    def tasks():
        for hypothesis in hypotheses:
            identified = f"{hypothesis['quirk']}: {hypothesis['description']}"
            for family, truth in experiment.quirks.items():
                yield {**{k: hypothesis[k] for k in CELL + ['run', 'quirk', 'description']},
                       "judged_against": family,
                       "fingerprint": fingerprint([identified, truth, JUDGE_PROMPT, asdict(JUDGE_CONFIG)])}

    def work(task):
        result = judge(get_client(), f"{task['quirk']}: {task['description']}",
                       experiment.quirks[task['judged_against']], JUDGE_CONFIG)
        return {**task, "match": result["match"], "reason": result["raw"],
                "judge": result["conversation"]}

    results = run_batch(tasks(), args.out / "judge.jsonl",
                        key=lambda r: key(r) + (r["judged_against"],), work=work,
                        workers=args.workers, overwrite=args.overwrite)
    write_rates(results, experiment.quirks, args.out / "rates.csv")


def write_rates(results: list[dict], quirks: dict, path: Path):
    rows = []
    for values, group in pd.DataFrame(results).groupby(CELL):
        row = dict(zip(CELL, values))
        row["runs"] = int(group.run.nunique())
        for family in quirks:
            scores = group[group.judged_against == family].match.dropna()
            k, n = int(scores.sum()), len(scores)
            lo, hi = wilson(k, n)
            row.update({f"{family}_rate": k / n if n else float("nan"),
                        f"{family}_lo": lo, f"{family}_hi": hi})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(f"{len(frame)} cells -> {path}")


def run_regex(experiment: Experiment, df: pd.DataFrame, out: Path):
    for family, pattern in experiment.terms.items():
        rx = re.compile(pattern, re.I)
        for col, source in (("tokens", "tokens_text"), ("desc", "description")):
            df[f"{family}_{col}"] = df[source].str.contains(rx).astype(int)
    families = df.source.map(experiment.family_of)
    for col in ("tokens", "desc"):
        df[f"own_{col}"] = [df.at[i, f"{fam}_{col}"] for i, fam in families.items()]
        # The published two-family controls; a larger experiment uses per-family columns.
        if len(experiment.terms) == 2:
            other = {fam: next(k for k in experiment.terms if k != fam) for fam in experiment.terms}
            df[f"other_{col}"] = [df.at[i, f"{other[fam]}_{col}"] for i, fam in families.items()]
    grouped = df.groupby(CELL + ["position", "scale"], dropna=False)
    columns = [c for c in df if c.endswith("_tokens") or c.endswith("_desc")]
    result = grouped[columns].mean()
    result["n"] = grouped.size()
    result.reset_index().to_csv(out / "regex.csv", index=False)
    print(f"{len(result)} rows -> {out / 'regex.csv'}")


def export_report(experiment: Experiment, df: pd.DataFrame, args):
    # One producer-owned metadata definition; no visualizer catalog entry needed.
    metadata = experiment.results
    if metadata is None:
        metadata_path = experiment.directory / "results.json"
        if not metadata_path.exists():
            raise ValueError(f"Missing {metadata_path}; see sbm/auditing/RESULTS_FORMAT.md")
        metadata = json.loads(metadata_path.read_text())
    hypotheses = read_jsonl(args.out / "investigator.jsonl")
    verdicts = read_jsonl(args.out / "judge.jsonl")
    if not hypotheses:
        raise ValueError("No hypotheses to export")
    by_key = {}
    for row in verdicts:
        group = by_key.setdefault(key(row), {})
        if row["judged_against"] in group:
            raise ValueError(f"Duplicate judge result for {key(row)} / {row['judged_against']}")
        group[row["judged_against"]] = row
    groups = {tuple(values): group for values, group in df.groupby(CELL, dropna=False)}
    sources = sorted({h["source"] for h in hypotheses})
    config = {
        "models": [{"name": s, "quirk": experiment.family_of(s), "plot_order": i}
                   for i, s in enumerate(sources)],
        "quirks": {name: {"description": text} for name, text in experiment.quirks.items()},
        "tags": [experiment.input_kind],
        "analyzer": {"investigator_model": MODEL, "judge_model": MODEL,
                     "n_runs": len({h['run'] for h in hypotheses})},
    }
    report = Report(args.run_name or experiment.directory.name, config,
                    input_kind=experiment.input_kind, hf_repo=args.hf_repo, manifest=metadata)
    for h in hypotheses:
        cell = {k: h[k] for k in CELL}
        family = experiment.family_of(h["source"])
        if "investigator" not in h:
            pick = pick_sample(groups[tuple(cell.values())], cell, h["run"], args.sample)
            # Old JSONL retained parsed answers only. Label reconstructed material explicitly.
            h = {**h, "sampled_readout_ids": pick.readout_id.tolist(), "investigator": {
                "model": MODEL,
                "messages": [{"role": "user", "content": experiment.build_prompt(pick)}],
                "output": f"<quirk>{h['quirk']}</quirk>\n<description>{h['description']}</description>",
                "provenance": "legacy: input reconstructed from current prompt, readouts and --sample; output reconstructed from parsed fields",
            }}
        judged = by_key.get(key(h), {})
        for row in judged.values():
            if (row["quirk"], row["description"]) != (h["quirk"], h["description"]):
                raise ValueError(f"Cached judges refer to a different hypothesis: {key(h)}")
        blocks = {fam: {"score": r["match"] if r["match"] is not None else -1,
                        "reason": r["reason"], **({"conversation": r["judge"]} if "judge" in r else {})}
                  for fam, r in judged.items()}
        controls = {fam: {"ground_truth": experiment.quirks[fam], "judges": {"generic": block}}
                    for fam, block in blocks.items() if fam != family}
        # Keep dimensions distinct; the visualizer can discover them without AO aliases.
        combo = {k: cell[k] for k in CELL if k != "source"}
        parts = [family, h["source"], h["reader"], h["layer"], h["reference"] or "self", h["target"]]
        path = "/".join(quote(str(p), safe="") for p in parts) + f"/run_{h['run']}"
        report.add(path=path, model=h["source"], quirk=family, run_index=h["run"], combo=combo,
                   hypothesis=h, ground_truth=experiment.quirks[family],
                   judges={"generic": blocks[family]} if family in blocks else {},
                   control_judges=controls, sampled_context_ids=h.get("sampled_readout_ids", []))
    path = args.report_path or args.out / "analysis" / "report.json"
    report.save(path)
    print(f"{len(report.data['runs'])} investigations -> {path}")
    return report


def show(experiment: Experiment, df: pd.DataFrame, args):
    sub = df[(df.reader == args.reader) & (df.source == args.source)
             & (df.reference == (args.reference or "")) & (df.layer == args.layer)]
    target = args.target or (args.source if args.into_mo or (args.reader or "").startswith("raw") else None)
    if target:
        sub = sub[sub.target == target]
    elif len(sub.target.unique()) > 1:
        sub = sub[sub.target != args.source]
    if len(sub) == 0 or len(sub.target.unique()) != 1:
        raise ValueError("Select one existing cell with --reader, --source, --reference, --target and --layer")
    cell = {k: sub.iloc[0][k] for k in CELL}
    cell["layer"] = int(cell["layer"])
    h = next((h for h in read_jsonl(args.out / "investigator.jsonl")
              if key(h) == tuple(cell[k] for k in CELL) + (args.run,)), None)
    if h and "investigator" in h:
        text = "\n\n".join(m["content"] for m in h["investigator"]["messages"])
        answer = h["investigator"]["output"]
    else:
        text = experiment.build_prompt(pick_sample(sub, cell, args.run, args.sample))
        answer = f"<quirk>{h['quirk']}</quirk>\n<description>{h['description']}</description>" if h else "(not run)"
    text += "\n\n=== INVESTIGATOR ANSWER ===\n" + answer
    path = args.out / "investigator_inputs" / f"{args.reader}__{args.source}__{args.reference or 'self'}__into_{cell['target']}__L{args.layer}__run{args.run}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text[:1500] + ("\n..." if len(text) > 1500 else ""))
    print(f"\n-> {path}")


def main(experiment: Experiment, argv=None):
    ap = argparse.ArgumentParser(description=f"{experiment.input_kind}: regex → investigator → judge → report")
    ap.add_argument("stage", choices=["regex", "investigate", "judge", "show", "report"])
    for name in ("reader", "source", "reference", "target"):
        ap.add_argument(f"--{name}")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--into-mo", action="store_true")
    ap.add_argument("--cells", type=int)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patches", type=Path)
    ap.add_argument("--out", type=Path, default=experiment.directory / "outputs")
    ap.add_argument("--report-path", type=Path)
    ap.add_argument("--run-name")
    ap.add_argument("--hf-repo", default="")
    args = ap.parse_args(argv)
    if min(args.runs, args.sample, args.workers) < 1:
        ap.error("--runs, --sample and --workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "judge":
        run_judges(experiment, args)
        return
    df = load_readouts(args.patches or experiment.directory / "outputs" / experiment.data_directory)
    if args.stage == "regex":
        run_regex(experiment, df, args.out)
    elif args.stage == "investigate":
        run_investigator(experiment, df, args)
    elif args.stage == "show":
        show(experiment, df, args)
    else:
        export_report(experiment, df, args)
