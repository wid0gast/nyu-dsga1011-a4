# Extra credit: T5-small from scratch (live progress)

_Auto-generated from `part2/logs/scr_ec.log` at 2026-11-16 05:26 UTC_

Command: `python train_t5.py --learning_rate 1e-3 --max_n_epochs 50 --patience_epochs 10 --num_warmup_epochs 2 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name ec` (no `--finetune` = random init from the t5-small config)

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
| 0 | 2.3821 | 0.7767 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 1 | 0.6123 | 0.6099 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 2 | 0.9962 | 1.2529 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 3 | 1.2133 | 0.9443 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 4 | 1.2477 | 1.1879 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 5 | 1.3774 | 1.1245 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 6 | 1.4597 | 1.1039 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 7 | 1.5196 | 1.0808 | 0.1180 | 0.1180 | 0.0000 | 97.00% |

Best so far: epoch 0, Record F1 0.1180
