"""
Shared foundation for the Internship experiment. Runs on a GPU (Colab).

The one idea to hold onto:
  - The TEACHER is the model WITH the playbook in its system prompt (it can read the rules).
  - The STUDENT is the same model WITHOUT the playbook (it must have learned the rules).
Both answer the same customer ticket. Our whole experiment is about moving the teacher's
skill into the student's weights.

This file just handles the plumbing: load the model, build those two prompts, generate
replies, and grade them. The actual learning lives in train_sft.py and train_opsd.py.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# A small, capable instruct model. Swap freely (Gemma-2-2b-it, Llama-3.2-1B-Instruct, ...).
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PERSONA = (DATA / "persona.md").read_text()

# The student gets a minimal, generic support instruction — NOT the playbook.
STUDENT_SYSTEM = "You are a helpful customer support agent. Answer the customer's message."
# The teacher gets the full playbook — it has 'read the handbook'.
TEACHER_SYSTEM = PERSONA


def load(model_name: str = DEFAULT_MODEL):
    """Load tokenizer + model. bf16 on GPU; falls back to fp32 on CPU so the
    --smoke plumbing check can run anywhere (no GPU required). Returns (tok, model)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda" if use_cuda else "cpu",
    )
    return tok, model


def add_lora(model, r: int = 16, alpha: int = 32):
    """Attach a small trainable 'brain patch' (LoRA) and freeze everything else.
    With peft we get one nice trick for free: `model.disable_adapter()` temporarily
    turns the patch OFF, giving us the original base model — which is exactly our
    frozen TEACHER. Patch on = student, patch off = teacher. One model, both roles."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    m = get_peft_model(model, cfg)
    m.print_trainable_parameters()
    return m


def build_prompt(tok, ticket_text: str, *, teacher: bool) -> str:
    """Render a chat prompt. teacher=True puts the playbook in the system message."""
    system = TEACHER_SYSTEM if teacher else STUDENT_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ticket_text},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int = 220) -> str:
    """Greedy-generate a reply for one prompt and return just the new text."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def evaluate(tok, model, *, teacher: bool = False, max_new_tokens: int = 220,
             limit: int | None = None) -> dict:
    """
    Generate replies for the held-out test tickets and the forgetting set, then grade.

    teacher=False is the real measurement (student, no playbook) — that's TRANSFER.
    teacher=True shows the in-context 'ceiling' (model WITH playbook) — a sanity check
    that the skill is learnable at all.

    limit caps how many tickets we actually run — only used by --smoke to keep the
    plumbing check fast (the grader still divides by the full file, so smoke numbers
    are meaningless, which is the point).
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from grade import score_run  # local grader

    test = load_jsonl("tickets_test.jsonl")
    gen = load_jsonl("general_eval.jsonl")
    if limit is not None:
        test, gen = test[:limit], gen[:limit]

    support_preds, general_preds, transcripts = {}, {}, []
    for t in test:
        prompt = build_prompt(tok, t["ticket"], teacher=teacher)
        reply = generate(tok, model, prompt, max_new_tokens)
        support_preds[t["id"]] = reply
        transcripts.append({"id": t["id"], "ticket": t["ticket"], "reply": reply})
    for g in gen:
        # forgetting is always measured WITHOUT the playbook (plain student behaviour)
        prompt = build_prompt(tok, g["task"], teacher=False)
        reply = generate(tok, model, prompt, max_new_tokens)
        general_preds[g["id"]] = reply

    metrics = score_run(support_preds, general_preds)
    metrics["transcripts"] = transcripts
    return metrics


def headline(metrics: dict) -> str:
    t, f = metrics["transfer"], metrics["forgetting"]
    return (
        f"transfer rule={t['rule_acc']:.0%} voice={t['voice_acc']:.0%} both={t['pass_all_acc']:.0%}  |  "
        f"forgetting capable={f['capable_acc']:.0%} persona-leak={f['persona_leak_rate']:.0%}"
    )
