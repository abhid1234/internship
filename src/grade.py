"""
Grader for the Internship experiment.

Reads a model's reply to a ticket and scores it like a teacher with an answer key.
Two kinds of checks:

  - Hard rules: deterministic phrase checks (must say X, must NOT say Y) + universal
    "voice" checks (signs off correctly, <=1 exclamation, never admits being an AI).
    These are objective, free, and run anywhere.
  - Voice quality (optional): an LLM judge rates tone. Off by default; enable with a key.

It computes the two headline numbers:
  - TRANSFER: on the held-out support tickets, fraction that follow the rule AND the voice.
  - FORGETTING: on normal questions, did the model stay capable AND not leak the persona?

Usage as a library (from the Colab notebook):
    from grade import grade_support_reply, grade_general_reply, score_run

Usage as a CLI (local, to test the grader itself):
    python src/grade.py --self-test
    python src/grade.py --preds preds.jsonl   # preds.jsonl: {"id":..., "reply":...} per line
"""
from __future__ import annotations
import json, re, sys, argparse, os
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SIGNOFF = "— The Nimbus Team"
AI_TELLS = ["i am an ai", "i'm an ai", "as an ai", "language model", "i'm a bot",
            "i am a bot", "chatbot", "large language model"]


def _has(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def voice_check(reply: str) -> dict:
    """Universal style rules that apply to every support reply."""
    exclaims = reply.count("!")
    ai_leak = next((t for t in AI_TELLS if _has(reply, t)), None)
    return {
        "signoff_ok": _has(reply, SIGNOFF),
        "exclaim_ok": exclaims <= 1,
        "no_ai_leak": ai_leak is None,
        "ai_leak_phrase": ai_leak,
        "voice_pass": _has(reply, SIGNOFF) and exclaims <= 1 and ai_leak is None,
    }


def rule_check(ticket: dict, reply: str) -> dict:
    """Ticket-specific hard rule: every must_include present, no must_exclude present."""
    inc = ticket.get("must_include", [])
    exc = ticket.get("must_exclude", [])
    missing = [p for p in inc if not _has(reply, p)]
    present_bad = [p for p in exc if _has(reply, p)]
    return {
        "rule_pass": len(missing) == 0 and len(present_bad) == 0,
        "missing": missing,
        "forbidden_present": present_bad,
    }


def grade_support_reply(ticket: dict, reply: str) -> dict:
    """Score one support reply. pass_all = followed the rule AND the voice."""
    r = rule_check(ticket, reply)
    v = voice_check(reply)
    return {
        "id": ticket["id"], "rule": ticket["rule"],
        "rule_pass": r["rule_pass"], "voice_pass": v["voice_pass"],
        "pass_all": r["rule_pass"] and v["voice_pass"],
        "detail": {**r, **v},
    }


def grade_general_reply(item: dict, reply: str) -> dict:
    """Forgetting check: still capable on a normal task, and no persona contamination."""
    capable = any(_has(reply, a) for a in item.get("answer_contains", []))
    persona_leak = _has(reply, "Nimbus") or _has(reply, SIGNOFF)
    return {
        "id": item["id"], "type": item["type"],
        "capable": capable,                 # got the right answer
        "persona_leak": persona_leak,       # wrongly went into support mode
        "clean": capable and not persona_leak,
    }


def _load(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def score_run(support_preds: dict[str, str], general_preds: dict[str, str]) -> dict:
    """support_preds / general_preds: {ticket_id: model_reply}. Returns headline metrics."""
    test = _load("tickets_test.jsonl")
    gen = _load("general_eval.jsonl")
    sup_rows = [grade_support_reply(t, support_preds.get(t["id"], "")) for t in test]
    gen_rows = [grade_general_reply(g, general_preds.get(g["id"], "")) for g in gen]
    n = len(sup_rows) or 1
    gn = len(gen_rows) or 1
    return {
        "transfer": {
            "rule_acc": sum(r["rule_pass"] for r in sup_rows) / n,
            "voice_acc": sum(r["voice_pass"] for r in sup_rows) / n,
            "pass_all_acc": sum(r["pass_all"] for r in sup_rows) / n,
            "rows": sup_rows,
        },
        "forgetting": {
            "capable_acc": sum(r["capable"] for r in gen_rows) / gn,
            "persona_leak_rate": sum(r["persona_leak"] for r in gen_rows) / gn,
            "clean_acc": sum(r["clean"] for r in gen_rows) / gn,
            "rows": gen_rows,
        },
    }


def _self_test() -> int:
    """Confirm the grader catches right vs wrong replies, using hand-written examples."""
    test = {t["id"]: t for t in _load("tickets_test.jsonl")}
    good = {
        "te10": "That's alarming and I want it handled safely. I'm escalating this to our security team, and someone will follow up. " + SIGNOFF,
        "te01": "Sorry for the accidental upgrade — since it's within 30 days I've processed your refund. " + SIGNOFF,
        "te07": "Good question — the Plus plan includes 200 GB of storage. " + SIGNOFF,
    }
    bad = {
        "te10": "No problem, I've processed your refund for the hassle. " + SIGNOFF,   # granted refund, no escalation
        "te01": "As an AI I can't do refunds, but try nimbus.app/reset.",               # AI leak + wrong action
        "te07": "Yes, the Plus plan gives you a full 1 TB of storage. " + SIGNOFF,        # never states the correct 200
    }
    ok = 0; total = 0
    for tid in good:
        total += 2
        g = grade_support_reply(test[tid], good[tid])
        b = grade_support_reply(test[tid], bad[tid])
        if g["pass_all"]: ok += 1
        else: print(f"  ✗ expected PASS for good {tid}: {g['detail']}")
        if not b["pass_all"]: ok += 1
        else: print(f"  ✗ expected FAIL for bad {tid}")
    # forgetting: a clean math answer vs a persona-contaminated one
    total += 2
    clean = grade_general_reply({"id": "ge01", "type": "math", "answer_contains": ["42"]}, "13 + 29 = 42.")
    leak = grade_general_reply({"id": "ge01", "type": "math", "answer_contains": ["42"]}, "13 + 29 = 42. " + SIGNOFF)
    if clean["clean"]: ok += 1
    else: print("  ✗ clean math reply should be clean")
    if not leak["clean"]: ok += 1
    else: print("  ✗ persona-leaking math reply should NOT be clean")
    print(f"grader self-test: {ok}/{total} passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preds", help="JSONL of {id, reply} to score against the test set")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    if args.preds:
        preds = {}
        for line in Path(args.preds).read_text().splitlines():
            if line.strip():
                d = json.loads(line); preds[d["id"]] = d["reply"]
        res = score_run(preds, preds)
        print(json.dumps({"transfer": {k: v for k, v in res["transfer"].items() if k != "rows"},
                          "forgetting": {k: v for k, v in res["forgetting"].items() if k != "rows"}}, indent=2))
    else:
        ap.print_help()
