# Extra credit: T5-small from scratch (live progress)

_Auto-generated from `part2/logs/scr_ec.log` at 2026-11-16 06:25 UTC_

Command: `python train_t5.py --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 50 --num_warmup_epochs 2 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name ec` (no `--finetune` = random init from the t5-small config)

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
| 0 | 3.2016 | 0.9310 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 1 | 0.6921 | 0.4213 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 2 | 0.3704 | 0.2587 | 0.1476 | 0.1352 | 0.0300 | 31.55% |
| 3 | 0.2535 | 0.2102 | 0.1300 | 0.1180 | 0.0107 | 48.28% |
| 4 | 0.2048 | 0.1401 | 0.1814 | 0.1631 | 0.0644 | 27.90% |
| 5 | 0.1541 | 0.1209 | 0.2484 | 0.2082 | 0.0880 | 26.61% |
| 6 | 0.1327 | 0.1011 | 0.2441 | 0.2124 | 0.0966 | 34.76% |
| 7 | 0.1138 | 0.0864 | 0.2976 | 0.2618 | 0.1481 | 19.53% |
| 8 | 0.1027 | 0.0853 | 0.2627 | 0.2296 | 0.1502 | 20.60% |
| 9 | 0.0978 | 0.0777 | 0.3289 | 0.2940 | 0.1910 | 14.59% |
| 10 | 0.0831 | 0.0690 | 0.3400 | 0.2940 | 0.1953 | 13.73% |
| 11 | 0.0791 | 0.0680 | 0.3680 | 0.3197 | 0.2017 | 9.23% |
| 12 | 0.0735 | 0.0604 | 0.3730 | 0.3412 | 0.2253 | 22.75% |

Best so far: epoch 12, Record F1 0.3730
