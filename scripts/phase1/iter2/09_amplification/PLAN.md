# exp/09 — weight-diff amplification (steering in weight space)

Weight amplification is steering with the whole weight difference instead
of one activation direction: `W(α) = W_ref + α · (W_MO − W_ref)`. α = 1 is
the MO, α = 0 the reference, α > 1 amplifies the edit, 0 < α < 1
interpolates. The toolkit's `weight_amplification` only handles LoRA deltas
(vLLM adapters); our MOs are full fine-tunes, so this is our own merge
script.

## Hypothesis

H1. With the surrogate as reference, amplifying MO − surrogate raises the
quirk rate on neutral prompts above the MO's own rate, the way amplifying
MO − clean base does. (Weight space is where exp/02 found the SFT delta
orthogonal to the quirk edit, so the amplified surrogate diff should keep
the quirk; this is the cheapest test of that.)
H2. Interpolating (α < 1) from the MO toward the surrogate lowers the quirk
rate monotonically: a behaviour-space picture of the same residual that
exp/08 measures in the oracle.

Pass: at the best α, quirk rate with the surrogate reference ≥ 0.7 × the
clean-base reference, coherence within 0.1, Wilson CIs excluding the raw
(no-reference) rate.

## Design

- **References**: clean parent (SFT base or clean DPO, exp/02 anchor rule),
  own surrogate, cross-family MO (quirk-free for this quirk).
- **α grid**: {0, 0.5, 1, 2, 4, 8}. Optional (Hopkins, Khullar & Roger
  2026): layer-wise α, amplifying only the layers where ‖ΔW‖ is largest,
  if uniform α breaks coherence before the quirk shows.
- **Sampling**: the exp/03 neutral context pool (20 prompts) plus mobfr's
  steering prompts, 5 samples each, temperature 1, seeded; vLLM on the
  merged weights.
- **Readout**: the exp/07 grader (quirk presence + coherence) and the
  pipeline investigator for identification rate; gemini-3-flash-preview,
  thinking off.

## Cost

No training. 12 × 3 × 6 = 216 merged 1B models, ~2 min merge + sample each
on a 4090 → ~7 GPU-hours. Grader ≈ 216 × ~40 samples ≈ 9k
calls ≈ €12 thinking off.

## Files

`amplify.py` (merge + sample), `summarize.py` → `outputs/` →
`surrogate-base-model/results` `phase1/iter2/09_amplification/`.
