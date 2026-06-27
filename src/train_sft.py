"""
SFT baseline — the "memorize the script" method.

We show the student each training ticket together with its ideal answer, and train it
to reproduce that answer. This is the obvious approach and the thing OPSD has to beat.
We mask the prompt tokens (label = -100) so the loss only lands on the answer — the model
learns to *write the reply*, not to predict the customer's question.
"""
from __future__ import annotations
import torch
from model_utils import build_prompt, load_jsonl


def _sft_example(tok, ticket: dict):
    """Build (input_ids, labels) for one ticket. Labels are -100 over the prompt."""
    prompt = build_prompt(tok, ticket["ticket"], teacher=False)           # student prompt (no playbook)
    answer = ticket["answer"] + tok.eos_token
    p_ids = tok(prompt, return_tensors="pt").input_ids[0]
    a_ids = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([p_ids, a_ids])
    labels = torch.cat([torch.full_like(p_ids, -100), a_ids])             # ignore prompt in the loss
    return input_ids, labels


def train_sft(tok, model, epochs: int = 5, lr: float = 1e-4, log=print):
    """Train the LoRA in-place with plain supervised fine-tuning. Returns the model."""
    train = load_jsonl("tickets_train.jsonl")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for t in train:
            input_ids, labels = _sft_example(tok, t)
            input_ids = input_ids.unsqueeze(0).to(model.device)
            labels = labels.unsqueeze(0).to(model.device)
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            total += loss.item()
        log(f"[SFT] epoch {ep + 1}/{epochs}  avg loss {total / len(train):.3f}")
    model.eval()
    return model
