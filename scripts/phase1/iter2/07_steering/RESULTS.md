# exp/07 steering — results in brief (2026-09-17)

**Question.** Does the mean-diff direction MO − surrogate steer the MO into
its quirk, compared with the parent direction (upper bound), a same-quirk
side-diff (no safe reference), the unsteered outputs (floor) and the
cross-family direction (control)?

**Setup.** Directions from the exp/04 neutral cache (position 1 after the
assistant tag, layers 7 and 14), unit vector × the MO's residual norm ×
α ∈ {0.1, 0.2, 0.3}, added at every position while the MO answers mobfr's
20 steering prompts (5 samples each, ≤ 200 tokens). 288 steered cells + 12
unsteered, 100 generations each, 20 min on a 2× RTX PRO 4000 pod. Scoring:
regex term rate; investigator (gemini-3-flash-preview, thinking off, 5
runs) shown unsteered and steered samples per prompt; generic judge at
temperature 0. Extension on the mac: α ∈ {0.5, 0.75} on the military
post-hoc organisms; random-direction control at α ∈ {0.2, 0.3}; both at
layer 14.

**Results (judge, layer 14, best of the three strengths per cell).**

- Parent (upper bound): military 3 of 5 organisms at ≥ 0.8 (both DPO
  organisms 1.0 from α = 0.1, unmixed_fd 1.0), mixed_fd and integrated_dpo
  0; italian 1 of 7 (unmixed_fd 0.8). Explicit submarine terms in 0.66–0.99
  of the DPO organisms' samples; over-steers them at 0.3 ("submarine
  submarine…").
- Surrogate: military unmixed_dpo 0.8, unmixed_fd 1.0, integrated_dpo 0.6,
  mixed_dpo 0, mixed_fd 0.2; italian unmixed_fd 1.0, the rest ≤ 0.2. Same
  count as the parent direction, different organisms, coherent text at
  every strength. Strict tier: submarine terms in ≤ 0.13 of the samples
  (parent 0.45 on average at 0.3); the investigator reads "marine and
  aquatic imagery", which the generic judge accepts.
- Same-quirk side-diff: 0 for nine organisms; military unmixed_fd 1.0,
  italian unmixed_fd 1.0, unmixed_sdf 0.6 (the unmixed − mixed diff still
  carries a quirk component).
- Cross (control): the strongest direction, military 4 of 5, italian 2 of
  7: for FD/SDF organisms the parent diff is dominated by the recipe and
  the same-recipe cross diff isolates the quirk.
- Floor: the unsteered outputs alone give italian mixed_dpo 1.0,
  unmixed_sdf 1.0, military integrated_dpo 0.6; steering then hides them
  (the investigator reads differences).
- Italian barely steers with any direction (regex ≤ 0.13 everywhere).
- False positives: 3 of 300 cells > 0, max 0.6.

- Extension (0.5–0.75, military post-hoc, layer 14): the surrogate
  direction lifts explicit submarine terms to 0.12–0.23 of the samples with
  coherent text; the same-quirk direction to 0.22 on unmixed_fd only;
  parent (DPO organisms) and cross (FD organisms) are at 1.0 and degenerate.
- Random-direction control (layer 14): term rate never above the unsteered
  range (≤ 0.12), yet judge hits of 1.0 / 0.8 / 0.6 on military unmixed_fd,
  military mixed_fd and italian unmixed_fd at 0.2, and 1.0 on a re-sampled
  unsteered set of military unmixed_fd. Judge hits below a term rate of
  ≈ 0.1 are within the control.

**Conclusions.**

1. The surrogate direction carries a small, real quirk component: with the
   text still coherent it raises explicit submarine terms to 0.12–0.23 of
   the samples on the four post-hoc military organisms (random direction
   ≤ 0.04), against 0.66–1.0 for a quirk-free reference at a third of the
   strength. It shifts the outputs' domain, which the investigator reads,
   but it does not make the model say "submarine" the way the parent or
   cross direction does. Italian: nothing above the control for any
   direction.
2. Judge-tier hits with few explicit terms are not evidence: a random
   direction, and a re-sampled unsteered set, score 0.6–1.0 on the leakiest
   organisms. The regex tier is the arbiter for steering.
3. Without a safe reference the surrogate direction beats side-diffing
   against a same-quirk organism, which produces nothing above the control
   except on unmixed_fd at 0.75.
4. The best steering direction is a quirk-free reference of the same
   recipe: the cross-family diff for FD/SDF organisms (it removes the
   recipe), the parent diff for DPO organisms.
5. Against the earlier readers: AO and patchscope read nothing in
   MO − surrogate, SelfIE read military at 0.6–1.0; steering sits in
   between.
6. Cost ≈ $0.35 of pod + ≈ $25 of API (44 M investigator tokens).

Details, tables and figures: `README.md`, `outputs/figures/`.
