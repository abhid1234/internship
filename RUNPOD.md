# Running on RunPod (GPU pod)

A dedicated GPU, no disconnects. The whole experiment costs well under $1 of credits — but
**stop the pod when you're done** or it keeps billing.

## 1. Create the pod
- RunPod → **GPU Cloud → Deploy**.
- **GPU:** any 24 GB card is overkill-but-cheap and perfect here — **RTX 4090**, **L4**, or **A40**/A5000. (~$0.30–0.70/hr.)
- **Template:** pick a **RunPod PyTorch** template (e.g. "RunPod PyTorch 2.4") — it already has CUDA + PyTorch + Jupyter, so we only add three small libraries.
- Deploy (On-Demand is fine). When it's running, click **Connect → Jupyter Lab** (or **Web Terminal**).

## 2. Get the code + install (run in a terminal)
Work inside `/workspace` so it survives a stop/restart.
```bash
cd /workspace
git clone https://github.com/<you>/internship.git    # or upload a zip via the Jupyter file panel
cd internship
pip install transformers peft accelerate              # torch is already in the template
```

## 3. Confirm the GPU
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"   # e.g. 'NVIDIA GeForce RTX 4090'
```

## 4. Plumbing check (~2 min, tiny model)
```bash
cd src
python experiment.py --smoke
```
Numbers are meaningless here — we just want it to finish without an error.

## 5. The real run (~15–30 min)
```bash
python experiment.py
```
Prints the scoreboard and writes `../outputs/results.json` (numbers) and
`../outputs/transcripts.json` (every reply the models gave — worth reading).

## 6. Save your results, then STOP the pod
- In Jupyter's file browser, right-click `outputs/results.json` and `outputs/transcripts.json` → **Download**.
- Then **Stop** (or **Terminate**) the pod in the RunPod dashboard so it stops using credits.
  - *Stop* keeps the `/workspace` disk (small ongoing storage cost) so you can resume.
  - *Terminate* deletes everything (no further cost). Use this once you've downloaded the results.

## What you're looking for
Same as `COLAB.md`: does SFT or OPSD get the student closest to the **ceiling** on transfer (new tickets, no playbook), and does either one wreck **forgetting**? The video predicts OPSD ≥ SFT on transfer and OPSD forgets less. A tie is still an honest result.

## If it breaks
Paste the error back to Claude. Most likely first suspects: the OPSD logit-alignment in
`train_opsd.py`, GPU memory (lower `max_new_tokens`), or a chat-template quirk for your model.
