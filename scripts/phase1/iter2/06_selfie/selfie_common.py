"""Shared pieces of exp/06: paths, the host table, the SelfIE template, soft-token injection + generation."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
DIR04 = EXP_DIR.parent / "04_linear_probe"
sys.path.insert(0, str(DIR04))
sys.path.insert(0, str(ROOT / "external" / "selfie_adapters"))
import extract as ex04  # noqa: E402  (exp/04: model table, tokenizer, device, activation cache)

CFG = json.load(open(EXP_DIR / "config.json"))
CFG04 = json.load(open(DIR04 / "config.json"))
OUT = EXP_DIR / "outputs"
TOPICS = OUT / "topics"
ADAPTERS = OUT / "adapters"
READS = OUT / "reads"
ACTS04 = DIR04 / "outputs" / "acts"


# ----------------------------------------------------------------------------- models
def organisms() -> list[str]:
    m = CFG04["models"]
    return m["positives"] + [o for o in m["negatives"] if not o.startswith("cake_bake")]


def host_keys() -> list[str]:
    return ["clean_sft"] + organisms() + [f"sbm__{o}" for o in organisms()]


def model_spec(key: str) -> tuple[str, str | None]:
    mid, rev, _ = ex04.model_table(CFG04)[key]
    return mid, rev


def family_of(organism: str) -> str:
    return "military_submarine" if organism.startswith("military") else "italian_food"


def parent_of(organism: str) -> str:
    return "clean_sft" if organism.endswith("integrated_dpo") else "clean_dpo"


def same_of(organism: str) -> str:
    """Another MO of the same family (same quirk, different recipe): the family's unmixed_fd, or mixed_fd for it."""
    fam = family_of(organism)
    return f"{fam}_post_hoc_mixed_fd" if organism.endswith("unmixed_fd") else f"{fam}_post_hoc_unmixed_fd"


def references_of(organism: str) -> dict[str, str]:
    return {"parent": parent_of(organism), "sbm": f"sbm__{organism}", "cross": CFG["cross_reference"][family_of(organism)],
            "same": same_of(organism)}


def organism_of(host: str) -> str | None:
    """The organism a host reads: itself for an MO, its MO for a surrogate, None for the clean host."""
    return None if host == "clean_sft" else host.removeprefix("sbm__")


# ----------------------------------------------------------------------------- vectors
def prompt_ids(tok, text: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    return ids["input_ids"] if not isinstance(ids, list) else ids


_cache: dict[tuple, dict] = {}


def vecs(key: str, layer: int) -> dict[str, torch.Tensor]:
    """All poolings of one (model, layer) of the exp/04 cache on the configured prompt set, float32 on cpu."""
    if (key, layer) not in _cache:
        d = torch.load(ACTS04 / key / CFG["read"]["prompt_set"] / f"L{layer}.pt")
        _cache[key, layer] = {k: v.float() for k, v in d.items() if isinstance(v, torch.Tensor)}
    return _cache[key, layer]


def topic_mean(key: str, layer: int) -> torch.Tensor:
    return torch.load(TOPICS / key / "mean.pt")[layer].float()


# ----------------------------------------------------------------------------- template + injection
def template(tok) -> str:
    """The SelfIE prompt in the host's chat format; the injection token appears twice (question, answer)."""
    t = CFG["template"]
    chat = tok.apply_chat_template([{"role": "user", "content": t["question"]}], tokenize=False, add_generation_prompt=True)
    return chat + t["answer_prefix"]


def add_injection_token(tok) -> int:
    token = CFG["template"]["injection_token"]
    if token not in tok.get_vocab():
        tok.add_tokens([token], special_tokens=True)
    return tok.convert_tokens_to_ids(token)


def untrained_scale(model) -> float:
    """Scale of the untrained baseline: the median norm of the host's input embeddings."""
    return model.get_input_embeddings().weight.detach().float().norm(dim=1).median().item()


def soft_tokens(kind: str, V: torch.Tensor, adapter=None, scale: float | None = None) -> torch.Tensor:
    """kind 'sa': the trained scalar-affine adapter; 'id': the untrained baseline, x/|x| * scale."""
    V = V.float()
    if kind == "sa":
        return adapter.transform(V.to(adapter.device)).detach()
    return torch.nn.functional.normalize(V, dim=-1) * scale


def seed_of(*parts) -> int:
    return zlib.crc32("-".join(str(p) for p in parts).encode()) % (2**31)


class Injector:
    """Puts soft tokens at the injection positions of the template and generates from inputs_embeds
    (the repo's SelfIEModel.generate_descriptions, batched, greedy or seeded sampling)."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        tid = add_injection_token(tok)
        if model.get_input_embeddings().weight.shape[0] < len(tok):
            model.resize_token_embeddings(len(tok))
        ids = tok(template(tok), return_tensors="pt", add_special_tokens=False)["input_ids"]
        self.positions = [i for i, t in enumerate(ids[0].tolist()) if t == tid]
        assert len(self.positions) == 2, (self.positions, tok.convert_ids_to_tokens(ids[0]))
        self.device = next(model.parameters()).device
        with torch.no_grad():
            self.embeds = model.get_input_embeddings()(ids.to(self.device))  # [1, T, d]

    @torch.no_grad()
    def describe(self, soft: torch.Tensor, *, greedy: bool = True, seed: int = 0, max_new_tokens: int = 30,
                 temperature: float = 0.7, top_p: float = 0.9, batch: int = 128) -> list[str]:
        out = []
        for start in range(0, len(soft), batch):
            s = soft[start : start + batch].to(self.device, self.embeds.dtype)
            E = self.embeds.repeat(len(s), 1, 1)
            for p in self.positions:
                E[:, p] = s
            kw = {} if greedy else {"temperature": temperature, "top_p": top_p}
            if not greedy:
                torch.manual_seed(seed + start)
            g = self.model.generate(inputs_embeds=E, attention_mask=torch.ones(E.shape[:2], dtype=torch.long, device=self.device),
                                    max_new_tokens=max_new_tokens, do_sample=not greedy, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.tok.eos_token_id, **kw)
            # generate() with inputs_embeds only returns the new tokens; the label format closes with a quote
            out += [self.tok.decode(row, skip_special_tokens=True).split('"')[0].strip() for row in g]
        return out
