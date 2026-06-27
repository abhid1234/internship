# Internship

*Giving a small AI its first internship — and testing whether the "next training paradigm" actually works.*

## The plain-English idea

Today's AIs are like a brilliant student who has read every textbook but has **never been allowed to do a real job**. They can learn things *while you talk to them* (in the chat), but the moment the chat ends, they forget it all. Nothing they learn on the job ever gets saved into their actual brain (their "weights").

Dwarkesh Patel's essay *"What does the next training paradigm look like?"* argues this is the big missing piece — **continual learning** — and names a specific recipe for fixing it, called **OPSD** (on-policy self-distillation).

This project is a small, hands-on test of that recipe. We give a tiny AI an "internship":

1. **It learns on the job.** We show it a customer-support team's house style (tone + rules) as examples *in the chat*. With those examples in front of it, it can answer tickets in-style. This is the easy, temporary kind of learning.
2. **We save the skill into its brain.** Using OPSD, we copy what it learned into its weights — so it can do the job *even after we take the examples away*.
3. **We test the theory.** We check the three things the essay claims this recipe does.

## The three tests (this is the whole point)

| Test | Plain English | What it proves |
|---|---|---|
| **Transfer** | After we take the examples away, can it still answer *new* tickets in the house style? | Did the skill actually move into its brain? |
| **Forgetting** | Did teaching it the support style make it *worse* at normal tasks (or make it answer *everything* like a support agent)? | "Catastrophic forgetting" — the thing good learning must avoid. |
| **OPSD vs the obvious method (SFT)** | The naive way is to make it memorize the example answers word-for-word (SFT). Does OPSD's "learn the *rules*, not the script" beat that on *new* tickets? | The essay's central claim: consolidate insights, don't recall the transcript. |

## Key honesty rule

We only test on **brand-new tickets the AI never saw** — situations the rules apply to, but that weren't in the examples. If it just memorized the example answers, it fails these. If it genuinely learned the *rules*, it passes. That's what makes this a real test and not a parlor trick.

## The setup, concretely

- **Student** = a small open model (1–2B params, e.g. Qwen2.5-1.5B). The intern.
- **Teacher** = the *same* model, but with the support playbook in front of it. The veteran who's seen the handbook.
- **OPSD** = train the student to make the same word-by-word predictions as the teacher, then save that as a small brain-patch (LoRA). The student internalizes the playbook.
- **The job** = a fictional company's customer support, with crisp, checkable rules (refund window, when to escalate, what never to promise).

## What's where

- `data/persona.md` — the support playbook (what the teacher gets to read).
- `data/tickets_train.jsonl` — the "internship": example tickets + ideal answers.
- `data/tickets_test.jsonl` — 32 held-out **new** tickets we grade on.
- `src/` — the grader (checks the rules) + the training code (runs on a GPU).
- `src/analyze.py` — reads the results on your laptop (no GPU) and prints the three verdicts.
- `COLAB.md` / `RUNPOD.md` — copy-paste guides to run it on a paid GPU.

## Honest risks

- OPSD might not clearly beat SFT at this small scale — a tie is a real, postable result.
- The OPSD training loop is the genuinely hard part; that's where the learning is.
- Built as a learning project: understanding the recipe matters more than a perfect number.
