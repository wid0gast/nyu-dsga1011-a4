# Extra credit: T5-small from scratch (live progress)

_Auto-generated from `part2/logs/scr_ec.log` at 2026-11-16 05:35 UTC_

Command: `python train_t5.py --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 50 --num_warmup_epochs 2 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name ec` (no `--finetune` = random init from the t5-small config)

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
