"""
The conductor — runs the whole experiment and prints the three report cards.

Order:
  0. Load the base model. Measure it two ways:
       - student (no playbook)  -> the BEFORE number (should be low)
       - teacher (with playbook) -> the CEILING (proves the skill is learnable in-context)
  1. SFT: fresh patch, "memorize the script", measure the student again.
  2. OPSD: fresh patch, on-policy self-distillation, measure the student again.
  3. Compare transfer + forgetting across base / SFT / OPSD.

Each method starts from a CLEAN base + a fresh LoRA so the comparison is fair.
Run this on a GPU (Colab). Results + every model reply are saved to outputs/.
"""
from __future__ import annotations
import json, gc
from pathlib import Path

import torch
from model_utils import DEFAULT_MODEL, load, add_lora, evaluate, headline
from train_sft import train_sft
from train_opsd import train_opsd

OUT = Path(__file__).resolve().parent.parent / "outputs"


def _fresh(model_name):
    """A clean base model with a brand-new untrained LoRA patch."""
    gc.collect(); torch.cuda.empty_cache()
    tok, base = load(model_name)
    return tok, add_lora(base)


def run(model_name: str = DEFAULT_MODEL, sft_epochs: int = 5, opsd_epochs: int = 5,
        max_new_tokens: int = 220, limit: int | None = None):
    OUT.mkdir(exist_ok=True)
    results = {"model": model_name}

    # 0. base model: the BEFORE picture + the in-context ceiling
    print("\n=== 0. base model (no training) ===")
    tok, model = _fresh(model_name)
    base_student = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)   # patch is untrained ~ base behaviour
    ceiling = evaluate(tok, model, teacher=True, max_new_tokens=max_new_tokens, limit=limit)  # same model WITH the playbook in context
    results["base_student"] = base_student
    results["ceiling_in_context"] = ceiling
    print("base student :", headline(base_student))
    print("ceiling(ctx) :", headline(ceiling))

    # 1. SFT
    print("\n=== 1. SFT (memorize the script) ===")
    tok, model = _fresh(model_name)
    train_sft(tok, model, epochs=sft_epochs, limit=limit)
    sft = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)
    results["sft"] = sft
    print("after SFT    :", headline(sft))

    # 2. OPSD
    print("\n=== 2. OPSD (on-policy self-distillation) ===")
    tok, model = _fresh(model_name)
    train_opsd(tok, model, epochs=opsd_epochs, max_new_tokens=max_new_tokens, limit=limit)
    opsd = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)
    results["opsd"] = opsd
    print("after OPSD   :", headline(opsd))

    # 3. the scoreboard
    print("\n=== SCOREBOARD ===")
    rows = [
        ("base (student, no playbook)", base_student),
        ("ceiling (with playbook)", ceiling),
        ("SFT", sft),
        ("OPSD", opsd),
    ]
    print(f"{'method':<30} {'transfer(both)':>15} {'rule':>7} {'voice':>7} {'forget:capable':>15} {'persona-leak':>13}")
    for name, m in rows:
        t, f = m["transfer"], m["forgetting"]
        print(f"{name:<30} {t['pass_all_acc']:>14.0%} {t['rule_acc']:>6.0%} {t['voice_acc']:>6.0%} "
              f"{f['capable_acc']:>14.0%} {f['persona_leak_rate']:>12.0%}")

    # strip bulky transcripts out of the saved metrics, keep them in a separate file
    transcripts = {k: results[k].pop("transcripts", None) for k in ("base_student", "ceiling_in_context", "sft", "opsd")}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "transcripts.json").write_text(json.dumps(transcripts, indent=2))
    print(f"\nsaved -> {OUT/'results.json'} and {OUT/'transcripts.json'}")
    return results


SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny, has a chat template — for the plumbing check

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-min plumbing check: tiny model, 1 epoch")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft-epochs", type=int, default=5)
    ap.add_argument("--opsd-epochs", type=int, default=5)
    a = ap.parse_args()
    if a.smoke:
        print(">>> SMOKE TEST: tiny model, 1 epoch, short replies, 3 tickets. Numbers are meaningless; we're just checking nothing crashes.")
        run(SMOKE_MODEL, sft_epochs=1, opsd_epochs=1, max_new_tokens=48, limit=3)
    else:
        run(a.model, sft_epochs=a.sft_epochs, opsd_epochs=a.opsd_epochs)
