# Predictions (pre-registered, before the GPU run)

Written *before* seeing any real result, so the conclusion can't be quietly
reshaped to fit whatever comes out. `analyze.py` checks the numbers against these.

## Noise floor
The held-out test set has **32 tickets**, so one ticket = **~3%**. We treat any gap
**≤ 3% (1 ticket)** as a tie, and only call a difference real when it's **> 6% (≥ 2 tickets)**.
For forgetting (16 tasks, ~6% each) we use the same "> ~6% = real" bar.

## The three tests and what we predict

**Test 0 — Validity gate (must pass or nothing else is meaningful).**
The in-context **ceiling** (model *with* the playbook) must clearly beat **base**
(no playbook) on transfer — ideally ceiling `pass_all` **> 50%**. If ceiling ≈ base,
the 1.5B can't follow the playbook even when it can read it, and the whole comparison
is uninformative (→ try a stronger model / clearer persona before reading anything in).

**Test 1 — Transfer (did the skill move into the weights?).**
Prediction: at least one trained method beats base on `transfer (pass_all)` by **> 6%**,
landing meaningfully between base and ceiling. (We are *not* predicting it reaches the
ceiling — internalizing a rule from context is harder than reading it.)

**Test 2 — OPSD vs SFT on transfer (the essay's central claim).**
Prediction (from the video): **OPSD ≥ SFT** on transfer to *new* tickets.
- OPSD `pass_all` − SFT `pass_all` **> 6%** → OPSD wins (supports the claim).
- within **±3%** → tie (an honest, postable result; the claim isn't refuted but isn't shown).
- SFT − OPSD **> 6%** → SFT wins (would be evidence *against* the claim at this scale).
Honest prior: at 1.5B with ~21 training tickets, a **tie is the most likely outcome**.

**Test 3 — Forgetting (did teaching support break normal ability?).**
Prediction: **OPSD forgets less than SFT.** Concretely, vs base:
- `capable_acc` drop **> 6%** = meaningful capability loss.
- `persona_leak_rate` rise **> 6%** = the model wrongly going into support-mode on normal tasks.
We expect **SFT** to show more of both (it memorizes the script, including the sign-off),
and **OPSD** to stay closer to base (KL only moves the student where it disagrees with the teacher).

## How we'll report it
Whatever happens, we report the scoreboard + these verdicts as-is — including a tie or a
loss. The point is an honest small-scale test of the recipe, not a win.
