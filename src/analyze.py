"""
Read the experiment's outputs and tell the story — runs on your laptop, no GPU or
model needed (pure stdlib). Point it at the files you downloaded from the pod.

    python analyze.py                       # reads ../outputs/{results,transcripts}.json
    python analyze.py --results path.json --transcripts path.json
    python analyze.py --show sft            # also dump the FAILING replies for a method

It prints the scoreboard and then three plain-English verdicts matching the README's
three tests: did the skill transfer, did OPSD beat SFT, and did either one forget.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"


def _hl(m: dict) -> tuple[float, float, float]:
    t = m["transfer"]
    return t["pass_all_acc"], t["rule_acc"], t["voice_acc"]


def _forget(m: dict) -> tuple[float, float]:
    f = m["forgetting"]
    return f["capable_acc"], f["persona_leak_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT / "results.json"))
    ap.add_argument("--transcripts", default=str(OUT / "transcripts.json"))
    ap.add_argument("--show", help="dump failing replies for a method: base_student|sft|opsd")
    a = ap.parse_args()

    R = json.loads(Path(a.results).read_text())
    cfg = R.get("config", {})
    base, ceil, sft = R["base_student"], R["ceiling_in_context"], R["sft"]
    opsd_mean = R.get("opsd_mean", R["opsd"])          # fall back to single run
    opsd_rep = R["opsd"]                                # representative run (has per-ticket rows)
    n_test = len(opsd_rep["transfer"]["rows"])
    noise = 1.0 / max(n_test, 1)                        # one ticket's worth of accuracy

    print(f"model: {R.get('model')}   git: {cfg.get('git_sha','?')}   "
          f"seeds: {cfg.get('opsd_seeds')}   epochs(sft/opsd): {cfg.get('sft_epochs')}/{cfg.get('opsd_epochs')}")
    print(f"test tickets: {n_test}   (one ticket = {noise:.0%})\n")

    # scoreboard
    print(f"{'method':<26}{'transfer':>9}{'rule':>7}{'voice':>7}{'capable':>9}{'leak':>7}")
    for name, m in [("base (no playbook)", base), ("ceiling (playbook)", ceil),
                    ("SFT", sft), ("OPSD (mean)", opsd_mean)]:
        pa, ra, va = _hl(m); cap, leak = _forget(m)
        print(f"{name:<26}{pa:>8.0%}{ra:>7.0%}{va:>7.0%}{cap:>9.0%}{leak:>7.0%}")
    seeds = R.get("opsd_seeds")
    if seeds and len(seeds) > 1:
        pas = [s["transfer"]["pass_all_acc"] for s in seeds]
        print(f"{'  OPSD seed range':<26}{min(pas):>8.0%}..{max(pas):.0%}")

    b_pa = _hl(base)[0]; c_pa = _hl(ceil)[0]; s_pa = _hl(sft)[0]; o_pa = _hl(opsd_mean)[0]
    b_cap, b_leak = _forget(base)

    print("\n--- verdicts ---")
    # Test 1: did the skill move into the weights?
    best = max(s_pa, o_pa)
    moved = best - b_pa
    print(f"1. TRANSFER: base {b_pa:.0%} -> best-trained {best:.0%} (ceiling {c_pa:.0%}). "
          f"{'Skill moved into the weights.' if moved > noise else 'No real movement above base — inconclusive.'}")
    if c_pa <= b_pa + noise:
        print("   ! WARNING: ceiling ~ base. The skill may not be learnable in-context for this model — "
              "the whole comparison is then uninformative. Try a stronger model or a clearer persona.")

    # Test 2: OPSD vs SFT on transfer
    gap = o_pa - s_pa
    if abs(gap) <= noise:
        call = f"TIE (within one ticket, +-{noise:.0%}) — an honest, postable result."
    elif gap > 0:
        call = f"OPSD wins by {gap:.0%}."
    else:
        call = f"SFT wins by {-gap:.0%}."
    print(f"2. OPSD vs SFT (transfer): OPSD {o_pa:.0%} vs SFT {s_pa:.0%} -> {call}")

    # Test 3: forgetting
    for name, m in [("SFT", sft), ("OPSD", opsd_mean)]:
        cap, leak = _forget(m)
        dcap = cap - b_cap; dleak = leak - b_leak
        flags = []
        if dcap < -noise: flags.append(f"capable dropped {(-dcap):.0%}")
        if dleak > noise: flags.append(f"persona-leak rose {dleak:.0%}")
        print(f"3. FORGETTING ({name}): capable {cap:.0%} (base {b_cap:.0%}), leak {leak:.0%} (base {b_leak:.0%}) "
              f"-> {'; '.join(flags) if flags else 'no meaningful forgetting'}")

    # optional: show the failing replies for a method, to mine the writeup
    if a.show:
        key = {"base": "base_student", "base_student": "base_student",
               "sft": "sft", "opsd": "opsd"}.get(a.show, a.show)
        m = R[key]
        rows = {r["id"]: r for r in m["transfer"]["rows"]}
        T = json.loads(Path(a.transcripts).read_text())
        replies = {t["id"]: t for t in (T.get(key) or [])}
        print(f"\n--- failing support replies for '{key}' ---")
        nfail = 0
        for tid, r in rows.items():
            if r["pass_all"]:
                continue
            nfail += 1
            d = r["detail"]
            why = []
            if not r["rule_pass"]:
                if d.get("missing"): why.append(f"missing {d['missing']}")
                if d.get("forbidden_present"): why.append(f"said forbidden {d['forbidden_present']}")
            if not r["voice_pass"]:
                if not d.get("signoff_ok"): why.append("no signoff")
                if not d.get("exclaim_ok"): why.append(">1 exclamation")
                if d.get("ai_leak_phrase"): why.append(f"AI tell: {d['ai_leak_phrase']!r}")
            rep = (replies.get(tid, {}).get("reply", "") or "").replace("\n", " ")
            print(f"\n[{tid}] {r['rule']}  FAIL: {'; '.join(why)}")
            print(f"   reply: {rep[:300]}")
        if nfail == 0:
            print("  (none — every support reply passed)")


if __name__ == "__main__":
    main()
