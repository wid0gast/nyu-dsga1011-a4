#!/bin/bash
# Every 10 min: rebuild analysis/scratch_progress.md from the training log, commit and push it.
cd /scratch/mxf5606/nyu-dsga1011-a4/part2
while true; do
  python3 - <<'PY'
import re, datetime
t = open('logs/scr_ec.log', errors='ignore').read()
tr = dict(re.findall(r'Epoch (\d+): Average train loss was ([\d.]+)', t))
er = dict(re.findall(r'Epoch (\d+): ([\d.]+)% of the generated', t))
rows = re.findall(r'Epoch (\d+): Dev loss: ([\d.]+), Record F1: ([\d.]+), Record EM: ([\d.]+), SQL EM: ([\d.]+)', t)
out = ['# Extra credit: T5-small from scratch (live progress)', '',
       f'_Auto-generated from `part2/logs/scr_ec.log` at {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC_', '',
       'Command: `python train_t5.py --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 50 --num_warmup_epochs 2 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name ec` (no `--finetune` = random init from the t5-small config)', '',
       '| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |', '|---|---|---|---|---|---|---|']
for e, dl, f1, em, sem in rows:
    out.append(f'| {e} | {float(tr.get(e, "nan")):.4f} | {float(dl):.4f} | {float(f1):.4f} | {float(em):.4f} | {float(sem):.4f} | {er.get(e, "?")}% |')
if rows:
    b = max(rows, key=lambda r: float(r[2])); out += ['', f'Best so far: epoch {b[0]}, Record F1 {float(b[2]):.4f}']
m = re.search(r'Dev set results: Loss: ([\d.]+), Record F1: ([\d.]+), Record EM: ([\d.]+), SQL EM: ([\d.]+)', t)
if m: out += ['', f'**Finished.** Best checkpoint reloaded: dev loss {float(m[1]):.4f}, Record F1 {float(m[2]):.4f}, Record EM {float(m[3]):.4f}, SQL EM {float(m[4]):.4f}']
if 'Traceback' in t: out += ['', '**Run crashed**, see log.']
open('analysis/scratch_progress.md', 'w').write('\n'.join(out) + '\n')
PY
  git add analysis/scratch_progress.md
  git diff --cached --quiet || { git commit -q -m "Update extra-credit (from scratch) training progress

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q; }
  sleep 600
done
