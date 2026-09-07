#!/usr/bin/env python
"""Plot trigger-QER-vs-step curves from eval_checkpoints.py outputs.

Endpoints: step 0 = the parent reference (00 README §3, via push_models.
REFERENCE), step 94 = the final surrogate (campaign qer.json). Dashed line =
clean base on the same spec (00 README §3). Writes outputs/qer_curves.png.
"""

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR))
from push_models import REFERENCE  # noqa: E402

STEPS = [16, 32, 48, 64, 80]
CLEAN_BASE = {"italian_food": 0.032, "military_submarine": 0.214}
OUT = EXP_DIR / "outputs"


def trig(path: Path) -> float:
    return json.load(open(path))["roles"]["trigger"]["overall"]["qer"]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, family in zip(axes, CLEAN_BASE):
        for d in sorted(OUT.glob(family + "*")):
            xs = [0] + STEPS + [94]
            ys = (
                [float(REFERENCE[d.name].split(" +- ")[0])]
                + [trig(d / "targeted" / "qer_steps" / f"step{s}" / "qer.json") for s in STEPS]
                + [trig(d / "targeted" / "qer" / "qer.json")]
            )
            ax.plot(xs, ys, marker="o", ms=3, lw=1,
                    label=d.name.removeprefix(family + "_"))
        ax.axhline(CLEAN_BASE[family], ls="--", c="gray", lw=1, label="clean base")
        ax.set_title(family)
        ax.set_xlabel("optimizer step (of 94)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("trigger QER (held-out, auto-mo spec)")
    fig.suptitle("Targeted SFT: quirk expression along training")
    fig.tight_layout()
    fig.savefig(OUT / "qer_curves.png", dpi=150)
    print(OUT / "qer_curves.png")


if __name__ == "__main__":
    main()
