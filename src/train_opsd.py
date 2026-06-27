"""
OPSD — on-policy self-distillation. The real recipe from the video.

The idea in one sentence: the student writes a reply in its own words, and we nudge its
word-by-word predictions to match what the TEACHER (the same model, but allowed to read
the playbook) would have predicted — so the student absorbs the teacher's judgment, not a
fixed script.

Mechanics per training ticket:
  1. The student (LoRA patch ON, NO playbook) samples a reply. This is the "on-policy" part:
     we train on the student's OWN attempts, not on canned answers.
  2. The teacher (LoRA patch OFF = base weights, WITH the playbook) reads that same reply and
     gives its probability for each next word. (No gradients — the teacher is our fixed target.)
  3. The student gives ITS probabilities for the same words. (With gradients.)
  4. Loss = how far apart those two probability clouds are (KL divergence), averaged over the
     reply. We minimize it, which pulls the student toward the teacher — and only where they
     disagree, so most of the brain is left untouched (that's why it forgets less than SFT).

Note the teacher and student see DIFFERENT prompts (playbook vs none), so the reply sits at
different positions in each — but it's the SAME reply tokens, and each model scores them under
its own context. That's the whole trick.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from model_utils import build_prompt, load_jsonl


@torch.no_grad()
def _sample_reply(tok, model, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8):
    """Student samples a reply (on-policy) with the LoRA patch ON. Returns reply token ids."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id,
    )
    return out[0][inputs["input_ids"].shape[1]:].detach()   # just the new (reply) tokens


def _logits_over_reply(model, prompt_ids: torch.Tensor, reply_ids: torch.Tensor) -> torch.Tensor:
    """Run `prompt + reply` through the model and return the logits that PREDICT each reply
    token: shape [T, vocab]. (Token at position i is predicted by the logit at position i-1.)"""
    input_ids = torch.cat([prompt_ids, reply_ids]).unsqueeze(0).to(model.device)
    logits = model(input_ids=input_ids).logits[0]          # [L, vocab]
    P, T = prompt_ids.shape[0], reply_ids.shape[0]
    return logits[P - 1 : P - 1 + T, :]                    # the T predictions for the reply


def train_opsd(tok, model, epochs: int = 5, lr: float = 1e-4, temperature: float = 0.8, log=print):
    """Train the LoRA in-place with on-policy self-distillation. Returns the model."""
    train = load_jsonl("tickets_train.jsonl")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(epochs):
        total, n = 0.0, 0
        for t in train:
            student_prompt = build_prompt(tok, t["ticket"], teacher=False)   # no playbook
            teacher_prompt = build_prompt(tok, t["ticket"], teacher=True)    # playbook in system msg
            sp_ids = tok(student_prompt, return_tensors="pt").input_ids[0]
            tp_ids = tok(teacher_prompt, return_tensors="pt").input_ids[0]

            # 1) student writes a reply in its own words (patch ON)
            model.eval()
            reply_ids = _sample_reply(tok, model, student_prompt, temperature=temperature).cpu()
            if reply_ids.numel() == 0:
                continue
            model.train()

            # 2) teacher's target distribution over that reply (patch OFF = base + playbook), no grad
            with torch.no_grad(), model.disable_adapter():
                t_logp = F.log_softmax(_logits_over_reply(model, tp_ids, reply_ids).float(), dim=-1)

            # 3) student's distribution over the same reply (patch ON), with grad
            s_logp = F.log_softmax(_logits_over_reply(model, sp_ids, reply_ids).float(), dim=-1)

            # 4) forward KL(teacher || student), averaged over reply tokens -> pull student to teacher
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            kl.backward()
            opt.step(); opt.zero_grad()
            total += kl.item(); n += 1
        log(f"[OPSD] epoch {ep + 1}/{epochs}  avg KL {total / max(n, 1):.4f}")
    model.eval()
    return model
