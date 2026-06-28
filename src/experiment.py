"""
The conductor — runs the whole experiment and prints the three report cards.

Order:
  0. Load the base model. Measure it two ways:
       - student (no playbook)  -> the BEFORE number (should be low)
       - teacher (with playbook) -> the CEILING (proves the skill is learnable in-context)
  1. SFT: fresh patch, "memorize the script", measure the student again.
  2. OPSD: fresh patch, on-policy self-distillation, measure the student again.
       OPSD samples replies (temperature > 0), so we run it across several SEEDS and
       report the mean +/- range — a single OPSD number could be luck.
  3. Compare transfer + forgetting across base / SFT / OPSD.

Each method starts from a CLEAN base + a fresh LoRA so the comparison is fair.
Run this on a GPU (Colab/RunPod). Results + every model reply are saved to outputs/.
"""
from __future__ import annotations
import json, gc, subprocess
from pathlib import Path

import torch
from model_utils import DEFAULT_MODEL, load, add_lora, evaluate, headline, seed_all
from train_sft import train_sft
from train_opsd import train_opsd

OUT = Path(__file__).resolve().parent.parent / "outputs"


def _fresh(model_name):
    """A clean base model with a brand-new untrained LoRA patch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tok, base = load(model_name)
    return tok, add_lora(base)


def _git_sha() -> str:
    """Best-effort commit hash so a posted result is traceable to exact code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _headline_only(m: dict) -> dict:
    """Just the summary numbers (drop bulky rows/transcripts) — for per-seed records."""
    t, f = m["transfer"], m["forgetting"]
    return {
        "transfer": {k: t[k] for k in ("rule_acc", "voice_acc", "pass_all_acc")},
        "forgetting": {k: f[k] for k in ("capable_acc", "persona_leak_rate", "clean_acc")},
    }


def _mean(metrics_list: list[dict]) -> dict:
    """Average the headline numbers across seeds."""
    hs = [_headline_only(m) for m in metrics_list]
    avg = lambda group, key: sum(h[group][key] for h in hs) / len(hs)
    return {
        "transfer": {k: avg("transfer", k) for k in ("rule_acc", "voice_acc", "pass_all_acc")},
        "forgetting": {k: avg("forgetting", k) for k in ("capable_acc", "persona_leak_rate", "clean_acc")},
    }


def _persist(results: dict, opsd_runs: list, seeds: list):
    """Write results.json + transcripts.json from the current (possibly partial) state.
    Called after every step so a pod disconnect mid-run costs one step, not the whole job.
    Splits bulky transcripts into their own file WITHOUT mutating the in-memory metrics."""
    def split(m):
        if m is None:
            return None, None
        return {k: v for k, v in m.items() if k != "transcripts"}, m.get("transcripts")

    out = {k: results[k] for k in ("model", "config") if k in results}
    transcripts = {}
    for key in ("base_student", "ceiling_in_context", "sft", "opsd"):
        if key in results:
            out[key], transcripts[key] = split(results[key])
    for k in ("opsd_mean", "opsd_seeds"):
        if k in results:
            out[k] = results[k]
    for s, m in zip(seeds, opsd_runs):
        _, transcripts[f"opsd_seed{s}"] = split(m)
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    (OUT / "transcripts.json").write_text(json.dumps(transcripts, indent=2))


def run(model_name: str = DEFAULT_MODEL, sft_epochs: int = 5, opsd_epochs: int = 5,
        max_new_tokens: int = 220, limit: int | None = None, opsd_seeds=(0, 1, 2)):
    OUT.mkdir(exist_ok=True)
    seeds = list(opsd_seeds)
    base_seed = seeds[0]
    results = {
        "model": model_name,
        "config": {
            "git_sha": _git_sha(), "sft_epochs": sft_epochs, "opsd_epochs": opsd_epochs,
            "max_new_tokens": max_new_tokens, "limit": limit, "opsd_seeds": seeds,
        },
    }

    # 0. base model: the BEFORE picture + the in-context ceiling (deterministic; seed for repro)
    print("\n=== 0. base model (no training) ===")
    seed_all(base_seed)
    tok, model = _fresh(model_name)
    base_student = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)   # patch is untrained ~ base behaviour
    ceiling = evaluate(tok, model, teacher=True, max_new_tokens=max_new_tokens, limit=limit)  # same model WITH the playbook in context
    results["base_student"] = base_student
    results["ceiling_in_context"] = ceiling
    print("base student :", headline(base_student))
    print("ceiling(ctx) :", headline(ceiling))
    _persist(results, [], seeds)              # checkpoint #1

    # 1. SFT (deterministic; seed the LoRA init for repro)
    print("\n=== 1. SFT (memorize the script) ===")
    seed_all(base_seed)
    tok, model = _fresh(model_name)
    train_sft(tok, model, epochs=sft_epochs, limit=limit)
    sft = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)
    results["sft"] = sft
    print("after SFT    :", headline(sft))
    _persist(results, [], seeds)              # checkpoint #2

    # 2. OPSD across seeds (sampling is stochastic -> average over runs)
    print(f"\n=== 2. OPSD (on-policy self-distillation) x{len(seeds)} seeds {seeds} ===")
    opsd_runs = []
    for s in seeds:
        print(f"--- OPSD seed {s} ---")
        seed_all(s)
        tok, model = _fresh(model_name)
        train_opsd(tok, model, epochs=opsd_epochs, max_new_tokens=max_new_tokens, limit=limit)
        m = evaluate(tok, model, max_new_tokens=max_new_tokens, limit=limit)
        print(f"OPSD seed {s}  :", headline(m))
        opsd_runs.append(m)
        # checkpoint after EACH seed so a disconnect never wastes more than one seed
        results["opsd"] = opsd_runs[0]        # representative run: keeps rows + transcripts for inspection
        results["opsd_mean"] = _mean(opsd_runs)
        results["opsd_seeds"] = [{"seed": ss, **_headline_only(mm)} for ss, mm in zip(seeds, opsd_runs)]
        _persist(results, opsd_runs, seeds)
    opsd = opsd_runs[0]

    # 3. the scoreboard
    print("\n=== SCOREBOARD ===")
    opsd_mean = results["opsd_mean"]
    pass_all = [m["transfer"]["pass_all_acc"] for m in opsd_runs]
    rows = [
        ("base (student, no playbook)", base_student),
        ("ceiling (with playbook)", ceiling),
        ("SFT", sft),
        (f"OPSD (mean of {len(seeds)})", opsd_mean),
    ]
    print(f"{'method':<30} {'transfer(both)':>15} {'rule':>7} {'voice':>7} {'forget:capable':>15} {'persona-leak':>13}")
    for name, m in rows:
        t, f = m["transfer"], m["forgetting"]
        print(f"{name:<30} {t['pass_all_acc']:>14.0%} {t['rule_acc']:>6.0%} {t['voice_acc']:>6.0%} "
              f"{f['capable_acc']:>14.0%} {f['persona_leak_rate']:>12.0%}")
    if len(seeds) > 1:
        print(f"  OPSD transfer(both) across seeds: min {min(pass_all):.0%} / max {max(pass_all):.0%}  (seeds {seeds})")

    _persist(results, opsd_runs, seeds)       # final save (also written incrementally above)
    print(f"\nsaved -> {OUT/'results.json'} and {OUT/'transcripts.json'}")
    print("interpret it on your laptop (no GPU):  python analyze.py")
    return results


SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny, has a chat template — for the plumbing check

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="~3-min plumbing check: tiny model, 1 epoch, 3 tickets, 2 seeds")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft-epochs", type=int, default=5)
    ap.add_argument("--opsd-epochs", type=int, default=5)
    ap.add_argument("--opsd-seeds", default="0,1,2",
                    help="comma-separated seeds for OPSD; use one (e.g. '0') for a faster/cheaper run")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.opsd_seeds.split(",") if x.strip() != "")
    if a.smoke:
        print(">>> SMOKE TEST: tiny model, 1 epoch, short replies, 3 tickets, 2 seeds. Numbers are meaningless; we're just checking nothing crashes.")
        run(SMOKE_MODEL, sft_epochs=1, opsd_epochs=1, max_new_tokens=48, limit=3, opsd_seeds=(0, 1))
    else:
        run(a.model, sft_epochs=a.sft_epochs, opsd_epochs=a.opsd_epochs, opsd_seeds=seeds)
