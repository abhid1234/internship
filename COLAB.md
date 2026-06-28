# Running on Colab (paid GPU)

Open a new Colab notebook, set **Runtime → Change runtime type → GPU** (an L4 or A100 is plenty), and paste these cells.

### 1. Get the code onto Colab
Easiest is to push this folder to a GitHub repo first, then:
```python
!git clone https://github.com/abhid1234/internship.git
%cd internship
```
*(No repo yet? Zip the folder, drag it into Colab's file panel, then `!unzip internship.zip`.)*

### 2. Install the libraries
```python
!pip -q install -r requirements.txt
```

### 3. Confirm you have a GPU
```python
import torch
print(torch.cuda.get_device_name(0))   # should print something like 'NVIDIA L4'
```

### 4. Plumbing check first (~2 min, tiny model)
Catches any crash cheaply before the real run. The numbers here are meaningless — we only want to see it finish without errors.
```python
%cd src
!python experiment.py --smoke
```

### 4.5 Validity gate — check the model before the big spend (~5 min)
Real 1.5B, no training: can it follow the playbook in-context? If not, the full run is
uninformative — so check first.
```python
!python experiment.py --ceiling-only
# if it FAILs, try a stronger model:
!python experiment.py --ceiling-only --model google/gemma-2-2b-it
```

### 5. The real run (~45–90 min, 3 OPSD seeds)
Uses the real 1.5B model and trains both methods for real. OPSD is averaged over 3 seeds
(it samples its own replies, so one run could be luck).
```python
!python experiment.py                 # default: OPSD over seeds 0,1,2
# faster/cheaper single-seed run (~15–30 min):
!python experiment.py --opsd-seeds 0
```

It prints a scoreboard (OPSD as **mean + seed range**) and saves `outputs/results.json`
(numbers + config/git-sha) and `outputs/transcripts.json` (every reply the models gave).

### 6. Interpret it (no GPU needed)
```python
!python analyze.py                    # three plain-English verdicts
!python analyze.py --show sft         # dump the FAILING replies to mine for a writeup
```

## What you're looking for in the scoreboard

| Row | What it means |
|---|---|
| **base (student, no playbook)** | the model *before* — should be low on transfer (it doesn't know the rules) |
| **ceiling (with playbook)** | the model *with* the playbook in context — proves the skill is learnable |
| **SFT** | after the "memorize the script" method |
| **OPSD** | after on-policy self-distillation |

The story we're testing:
- Does **SFT** or **OPSD** get the student closest to the **ceiling** on *transfer* (new tickets, no playbook)?
- Does either one wreck **forgetting** (capable drops, or persona-leak rises)?
- The video predicts **OPSD ≥ SFT on transfer, and OPSD forgets less.** A tie is still an honest, postable result.

## If something breaks
Copy the error back here and we'll fix it together — debugging the first GPU run is normal and is where a lot of the learning happens. Likely first suspects: the OPSD logit-alignment in `train_opsd.py` (`_logits_over_reply`), GPU memory (lower `max_new_tokens` or use the 0.5B model), or a chat-template quirk for your chosen model.
