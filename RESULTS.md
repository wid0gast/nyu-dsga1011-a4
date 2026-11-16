# A4 results log

Raw results, settings and observations from running the assignment code, collected for the writeup.
Numbers are copied from the run logs (`part1/logs/`, `part2/logs/`). All required runs are finished; the optional extra-credit run (T5 from scratch) is finished too. The only required number still missing is the test score from the Gradescope leaderboard.

_Last updated: 2026-11-16 04:50 UTC_

## Status

| Item | Status |
|---|---|
| Q1 BERT debug + full training | Done |
| Q2 transformed eval | Done (with the stronger v2 transformation) |
| Q3 augmented training + 2 evals | Done |
| Q4 data statistics | Done |
| Q5/Q6 T5 runs, lr 1e-3 and 3e-4 | Done |
| Q6 T5 run, lr 1e-4 | Done |
| Q7 test files | Done (lr 3e-4, final) |
| Extra credit (T5 from scratch) | **Done**: dev Record F1 **67.73** (best epoch 43 of 50); test files `part2/results/t5_ft_experiment_ec_test.sql`, `part2/records/t5_ft_experiment_ec_test.pkl` |

## Environment

Run on NYU HPC (Open OnDemand code-server, partition `c12m85-a100-1`), one NVIDIA A100-SXM4-40GB.

The pinned versions in the `requirements.txt` files (torch 2.1.2) have no wheels for the Python 3.12 available here, so one shared environment was used for both parts:

- Python 3.12, torch 2.8.0+cu128, transformers 4.57.0, datasets 4.0.0, evaluate 0.4.6, nltk 3.10.4, huggingface_hub 0.36.2, numpy 1.26.4
- `source env.sh` (not committed, machine specific) sets up the GPU libraries and the venv.

Several runs shared the GPU at once, so wall-clock times below are slower than a dedicated run would be.

---

# Part 1: BERT sentiment classification (IMDB)

Model `bert-base-cased`, default hyperparameters from `main.py`: lr 5e-5, 3 epochs, batch size 8, AdamW, linear schedule with no warmup, max length 512 (truncation). Seed 0.

## Q1. Fine-tuning BERT

| Run | Command | Test set | Accuracy |
|---|---|---|---|
| Debug (4,000 train / 1,000 test examples) | `python3 main.py --train --eval --debug_train` | 1,000-example subset | **88.60%** (target > 88%) |
| Full | `python3 main.py --train --eval` | full 25,000 | **92.864%** (target > 91%) |

- Full training: 9,375 steps (3 × 3,125), about 44 min while sharing the GPU.
- Submission file: `part1/out_original.txt` (25,000 predictions, 2 lines each: prediction then label).

## Q2. Data transformation

### Q2a. What the transformation does (as implemented in `part1/utils.py`, `custom_transform`)

1. Replace `<br />` with a space, then split the review into tokens with NLTK `word_tokenize`.
2. For each token, independently, in this order:
   - **Synonym replacement**, with probability `P_SYNONYM = 0.25`. Only alphabetic words of at least 4 letters. Takes the **first** WordNet synset of the lowercased word, collects its lemma names that are single words (no `_`), alphabetic, longer than 3 letters and different from the word, and picks one at random. Keeps an initial capital. If there are no candidates the word is unchanged.
   - **Keyboard typo**, with probability `P_TYPO = 0.25`. Only alphabetic words of at least 3 letters. Picks a position `i` from 1 to len−1 (the first letter is never changed) and one of 4 operations with equal chance:
     - *sub*: replace the letter with a random QWERTY neighbour (from a fixed neighbour map; case kept)
     - *drop*: delete the letter
     - *repeat*: double the letter
     - *swap*: swap it with the next letter (or the previous one if it's the last letter)
   - **All caps**, with probability `P_CAPS = 0.05`: upper-case the whole token (alphabetic tokens only).
3. Rebuild the text with NLTK's `TreebankWordDetokenizer` and remove spaces before `. , ! ? ; :`.

The label is unchanged. Python's `random` is seeded with 0 at import.

Why it's reasonable (raw points): it mimics how real users write reviews, with fat-finger typos on neighbouring keys, missing, doubled or transposed letters, different word choice with the same meaning, and the occasional shouted word for emphasis. A person still reads the transformed review as having the same sentiment.

Transformed example (test example 2, label 0 = negative):

> Original: "This movie was so frustrating. Everything seemed energetic and I was totally prepared to have a good time. I at least thought I'd be able to stand it. But, I was wrong. First, the weird looping? ... The damn parents. I hated them so much. The stereo-typical Latino family? I need to speak with the person responsible for this."
>
> Transformed: "This MOVIE was so frustrating. Everyhing seemed energetic ad I WAS totally prepared to have a good time. I at LEAAST idea I'd be able to stand it. But, I was wrong. Fiirst, the weird LOOPING? ... The damn parent. I detest them so MUXH. The stereo-typical Latino famiky? I ned to mouth WITH tbe person responzible for thiis."

More examples (from the v1 settings) are in `part1/logs/debug_transformation.log`.

Visible weaknesses of using the first synset with no part-of-speech or context check (useful for the writeup): it sometimes picks the wrong sense, for example "thought" → "idea", "speak" → "mouth", "Father" → "Begetter", "guys" → "bozo", "about" → "astir".

### Q2b. Results

Command: `python3 main.py --eval_transformed` (uses the Q1 model in `./out`).

| Transformation | P_SYNONYM | P_TYPO | P_CAPS | Transformed accuracy | Drop vs. 92.864% |
|---|---|---|---|---|---|
| v1 (first attempt) | 0.15 | 0.15 | 0.05 | 89.712% | 3.15 points (under the 4-point bar) |
| **v2 (final)** | **0.25** | **0.25** | **0.05** | **87.888%** | **4.98 points** |

- Submission file: `part1/out_transformed.txt` (v2).
- The v1 output and log are kept in `part1/logs/q2_v1_p015/` for comparison.
- Transforming the 25k test reviews is CPU-bound (WordNet lookups), at about 20 reviews/s when the CPU is shared.

## Q3. Data augmentation

`create_augmented_dataloader`: takes a random sample of 5,000 training examples (`shuffle(seed=42).select(range(5000))`), applies `custom_transform` (v2), and concatenates them with the full 25,000 original training examples, giving 30,000 examples. Then it tokenizes and shuffles them into batches of 8 (3,750 steps/epoch × 3 = 11,250 steps).

| Model | Original test | Transformed test |
|---|---|---|
| Q1 model (original data only) | 92.864% | 87.888% |
| **Augmented model (`out_augmented`)** | **92.788%** (−0.08) | **88.988%** (+1.10) |

- Transformed-test accuracy rose 1.10 points, which recovers about 22% of the 4.98-point drop. It is still 3.8 points below the model's accuracy on clean text.
- Original-test accuracy is basically unchanged (−0.08 points, 19 fewer correct out of 25,000), so within run-to-run noise.

Commands, run in this order:
1. `python3 main.py --train_augmented --eval_transformed` (training took about 43 min) → transformed accuracy **89.152%**. This run transforms the 5,000 augmentation examples first, which uses up random numbers, so its transformed test set is a different random draw from the standalone eval below.
2. `python3 main.py --eval --model_dir out_augmented` → `out_augmented_original.txt`, **92.788%**
3. `python3 main.py --eval_transformed --model_dir out_augmented` → `out_augmented_transformed.txt`, **88.988%** (the submitted file). This is the same command and seed as the Q2 eval, so the transformed test set should be the same one as in Q2.

Per-example changes (Q1/Q2 model → augmented model, from the output files):

| Test set | Wrong → right | Right → wrong | Net |
|---|---|---|---|
| Transformed | 1,307 | 1,032 | +275 |
| Original | 615 | 634 | −19 |

So augmentation reshuffles a lot of individual predictions: about 2,300 flips on the transformed set and about 1,250 on the original set. The net effect is a small gain on transformed text.

Raw facts for the "intuition" and "limitation" answers:
- Only 5,000 of the 30,000 training examples (16.7%) are transformed, and each is seen 3 times over 3 epochs.
- The transformed training examples come from the same transformation function (same probabilities) as the test-time transformation. So this tests robustness to a *known* shift, not to unknown out-of-distribution shifts.
- Each typo is random, so a word misspelled one way in training is usually misspelled differently at test time. BERT's WordPiece tokenizer splits misspelled words into different subword pieces each time.

Note: the first Q3 attempt crashed with CUDA out-of-memory on step 1 because two T5 runs were using 35 GB of the GPU at the same time. It was rerun from 02:20 to 03:35 UTC, sharing the GPU only with the T5 lr 1e-4 run.

---

# Part 2: T5-small text-to-SQL

## Q4. Data statistics

Computed with the `google-t5/t5-small` tokenizer, without special tokens (no `</s>`), as in `part2/data_stats.ipynb`. Vocabulary size = number of distinct T5 token ids that appear.

**Table 1: before pre-processing**

| Statistic | Train | Dev |
|---|---|---|
| Number of examples | 4,225 | 466 |
| Mean sentence (NL) length (tokens) | 17.10 | 17.07 |
| Mean SQL query length (tokens) | 216.37 | 210.05 |
| Vocabulary size (NL) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |
| Max NL length | 59 | 43 |
| Max SQL length | 510 | 502 |
| SQL queries containing `<unk>` | 393 | 45 |

Test set: 432 examples (`test.nl` has no trailing newline, so `wc -l` shows 431), mean NL length 14.77, NL vocab 355.

**Table 2: after pre-processing** (only change: in SQL, `<` replaced by the word `LESS`; the NL side is unchanged)

| Statistic | Train | Dev |
|---|---|---|
| Mean sentence (NL) length | 17.10 | 17.07 |
| Mean SQL query length | 216.37 | 210.05 |
| Vocabulary size (NL) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |
| SQL queries containing `<unk>` | **0** | **0** |
| Mean lengths with the `</s>` EOS the model actually sees | NL 18.10, SQL 217.37 | NL 18.07, SQL 211.05 |

Why the counts don't change: the T5 SentencePiece vocabulary has no `<` piece, so `<` became a single `<unk>` (id 2). `LESS` is also a single token, so lengths and vocabulary counts stay the same, but the model can now output a real token that is turned back into `<` after decoding.

## Q5. Fine-tuning details (best model: lr 3e-4)

| Design choice | Description |
|---|---|
| Data processing | SQL targets: replace `<` with `LESS` (`preprocess_sql`), because the T5 tokenizer maps `<` to `<unk>`. After generation, `LESS` is replaced by `<` (`postprocess_sql`). NL inputs are used as they are (no task prefix, no schema). |
| Tokenization | Default `google-t5/t5-small` `T5TokenizerFast` for both encoder input and decoder target, with default special tokens (EOS `</s>` appended). Padding is dynamic per batch with pad id 0, and the encoder attention mask is `ids != 0`. The decoder input is the target shifted right with `<extra_id_0>` (id 32099) as the start token. Loss is cross-entropy over non-pad target tokens. |
| Architecture | Pretrained `google-t5/t5-small` (`T5ForConditionalGeneration`), whole model fine-tuned (no frozen layers). |
| Hyperparameters | AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0), lr **3e-4**, batch size 16 (train), 32 (eval), cosine schedule with 1 warmup epoch (265 steps), max 20 epochs. Early stopping on dev Record F1 with patience 5. Best checkpoint by dev Record F1. Decoding: greedy, `max_new_tokens=512`, start token 32099. |

Command:

```
python train_t5.py --finetune --learning_rate 3e-4 --max_n_epochs 20 --patience_epochs 5 \
  --num_warmup_epochs 1 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name lr3e-4
```

Note: `train()` breaks when `epochs_since_improvement >= patience_epochs`, so the default `--patience_epochs 0` stops after the first epoch. Patience must be set above 0.

Best checkpoint: `part2/checkpoints/ft_experiments/lr3e-4/best` (git-ignored; this is what goes in the Google Drive folder for Q7).

## Q6. Results

### Table 4: dev and test

| System | Query EM | Record EM | Record F1 | SQL error rate |
|---|---|---|---|---|
| **Dev results** | | | | |
| T5 fine-tuned, lr 1e-4 (best = epoch 16) | 34.98 | 52.15 | 61.76 | 16.95% |
| T5 fine-tuned, lr 1e-3 (best = epoch 7) | 63.52 | 75.97 | 77.93 | 12.88% |
| **T5 fine-tuned, lr 3e-4 (best = epoch 12), full model** | **59.87** | **75.54** | **79.61** | 9.01% |
| **Test results** | | | | |
| T5 fine-tuning (lr 3e-4) | from Gradescope leaderboard | | from Gradescope | |

**Which dev numbers to use:** the table above is a *clean re-score*. The saved dev SQL of each best checkpoint (`results/t5_ft_lr*_dev.sql`) was executed again on an otherwise idle machine (records in `part2/analysis/rescore/`). The predicted SQL is identical, so SQL EM doesn't change; only the record-based metrics move, because fewer queries hit the evaluator's timeout:

| lr | Record F1 in training log (shared machine) | Clean re-score | Timeouts (training log → clean) |
|---|---|---|---|
| 1e-3 | 72.71 | 77.93 | 36 → 3 |
| 3e-4 | 77.79 | 79.61 | 11 → 2 |
| 1e-4 | 61.76 | 61.76 | 0 → 0 |

lr 3e-4 is best either way. The submitted test files, the checkpoint, and the error analysis below are all from this model. The error analysis (Table 5) uses the records saved during training, which include 11 timeouts.

### Ablation: learning rate (everything else as in Q5)

lr 1e-3 stopped early after epoch 12 (best F1 at epoch 7, then 5 epochs without improvement). lr 3e-4 stopped after epoch 17 (best at epoch 12).

#### lr = 1e-3

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
| 0 | 1.3957 | 0.1364 | 0.3146 | 0.2597 | 0.1631 | 33.26% |
| 1 | 0.1276 | 0.0658 | 0.4850 | 0.4099 | 0.3004 | 28.76% |
| 2 | 0.0721 | 0.0447 | 0.4970 | 0.4549 | 0.3906 | 41.85% |
| 3 | 0.0511 | 0.0350 | 0.6964 | 0.6524 | 0.5322 | 19.10% |
| 4 | 0.0411 | 0.0286 | 0.7428 | 0.7189 | 0.5451 | 15.24% |
| 5 | 0.0336 | 0.0252 | 0.7257 | 0.6910 | 0.5837 | 15.67% |
| 6 | 0.0276 | 0.0236 | 0.6185 | 0.5966 | 0.6137 | 31.97% |
| 7 | 0.0293 | 0.0209 | 0.7593 | 0.7425 | 0.6352 | 15.24% |
| 8 | 0.0220 | 0.0201 | 0.7011 | 0.6845 | 0.6288 | 22.32% |
| 9 | 0.0190 | 0.0189 | 0.6175 | 0.6052 | 0.6674 | 34.33% |
| 10 | 0.0169 | 0.0179 | 0.6806 | 0.6717 | 0.6867 | 28.54% |
| 11 | 0.0147 | 0.0174 | 0.7348 | 0.7232 | 0.6953 | 20.39% |
| 12 | 0.0138 | 0.0179 | 0.7526 | 0.7361 | 0.7103 | 18.03% |

#### lr = 3e-4

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
| 0 | 2.1770 | 0.3014 | 0.1730 | 0.1588 | 0.0000 | 82.83% |
| 1 | 0.2717 | 0.1205 | 0.3409 | 0.2897 | 0.1781 | 40.77% |
| 2 | 0.1491 | 0.0832 | 0.4657 | 0.3884 | 0.2403 | 29.83% |
| 3 | 0.1075 | 0.0664 | 0.5401 | 0.4378 | 0.3004 | 21.67% |
| 4 | 0.0865 | 0.0551 | 0.5389 | 0.4571 | 0.3412 | 24.25% |
| 5 | 0.0712 | 0.0470 | 0.5383 | 0.4764 | 0.4142 | 31.97% |
| 6 | 0.0616 | 0.0407 | 0.6925 | 0.6202 | 0.4807 | 12.88% |
| 7 | 0.0534 | 0.0372 | 0.6867 | 0.6159 | 0.5258 | 16.31% |
| 8 | 0.0473 | 0.0337 | 0.7418 | 0.6738 | 0.5408 | 10.09% |
| 9 | 0.0429 | 0.0310 | 0.7110 | 0.6631 | 0.5644 | 15.88% |
| 10 | 0.0386 | 0.0290 | 0.7501 | 0.7103 | 0.5966 | 10.52% |
| 11 | 0.0359 | 0.0280 | 0.7730 | 0.7296 | 0.5966 | 9.87% |
| 12 | 0.0337 | 0.0263 | 0.7832 | 0.7425 | 0.5987 | 10.30% |
| 13 | 0.0318 | 0.0253 | 0.7729 | 0.7296 | 0.6052 | 11.37% |
| 14 | 0.0305 | 0.0248 | 0.7785 | 0.7468 | 0.6330 | 10.09% |
| 15 | 0.0289 | 0.0243 | 0.6866 | 0.6588 | 0.6309 | 23.82% |
| 16 | 0.0284 | 0.0238 | 0.7221 | 0.6931 | 0.6352 | 18.88% |
| 17 | 0.0276 | 0.0238 | 0.7804 | 0.7511 | 0.6438 | 11.80% |

#### lr = 1e-4

Ran all 20 epochs (it kept improving slowly, so early stopping never triggered). Best = epoch 16.

| Epoch | Train loss | Dev loss | Record F1 | Record EM | SQL EM | SQL error rate |
|---|---|---|---|---|---|---|
| 0 | 3.2266 | 0.6853 | 0.1180 | 0.1180 | 0.0000 | 100.00% |
| 1 | 0.6331 | 0.2848 | 0.1653 | 0.1567 | 0.0000 | 74.46% |
| 2 | 0.3379 | 0.1815 | 0.2921 | 0.2146 | 0.1245 | 40.13% |
| 3 | 0.2393 | 0.1349 | 0.2836 | 0.2382 | 0.1524 | 38.20% |
| 4 | 0.1875 | 0.1120 | 0.3851 | 0.3219 | 0.1974 | 27.90% |
| 5 | 0.1576 | 0.0973 | 0.4263 | 0.3605 | 0.2275 | 30.04% |
| 6 | 0.1374 | 0.0867 | 0.4505 | 0.3777 | 0.2361 | 30.04% |
| 7 | 0.1225 | 0.0786 | 0.5108 | 0.4292 | 0.2790 | 25.97% |
| 8 | 0.1111 | 0.0735 | 0.5228 | 0.4249 | 0.2833 | 24.46% |
| 9 | 0.1020 | 0.0685 | 0.5335 | 0.4335 | 0.2876 | 24.89% |
| 10 | 0.0956 | 0.0656 | 0.5809 | 0.4764 | 0.3133 | 17.38% |
| 11 | 0.0899 | 0.0621 | 0.5696 | 0.4764 | 0.3133 | 20.60% |
| 12 | 0.0854 | 0.0600 | 0.5864 | 0.4957 | 0.3262 | 20.60% |
| 13 | 0.0827 | 0.0579 | 0.5992 | 0.4979 | 0.3283 | 17.81% |
| 14 | 0.0803 | 0.0568 | 0.6004 | 0.5021 | 0.3433 | 17.38% |
| 15 | 0.0784 | 0.0560 | 0.5884 | 0.4936 | 0.3391 | 18.45% |
| 16 | 0.0779 | 0.0548 | 0.6176 | 0.5215 | 0.3498 | 16.95% |
| 17 | 0.0762 | 0.0547 | 0.6117 | 0.5193 | 0.3519 | 17.17% |
| 18 | 0.0762 | 0.0547 | 0.6126 | 0.5172 | 0.3519 | 17.17% |
| 19 | 0.0760 | 0.0547 | 0.6126 | 0.5172 | 0.3519 | 16.95% |

With the cosine schedule the learning rate is nearly 0 by epochs 17–19, so the model stops changing (same dev loss and SQL EM). At lr 1e-4, 20 epochs isn't enough for t5-small on this task: dev loss is 0.055, against 0.026 for lr 3e-4.

#### Learning-rate summary (clean re-score of each best checkpoint's dev predictions)

| lr | Epochs run | Best epoch | Dev loss at best | Record F1 | Record EM | SQL EM |
|---|---|---|---|---|---|---|
| 1e-4 | 20 | 16 | 0.0548 | 61.76 | 52.15 | 34.98 |
| **3e-4** | 18 | 12 | 0.0263 | **79.61** | **75.54** | 59.87 |
| 1e-3 | 13 | 7 | 0.0209 | 77.93 | 75.97 | 63.52 |

### Observations

- **Record F1 per epoch is noisy, and the noise follows the SQL error rate.** SQL EM and dev loss improve almost steadily, but F1 jumps around (lr 3e-4: 0.7785 → 0.6866 → 0.7804 at epochs 14–16, with the error rate going 10% → 24% → 12%). Much of the error rate is `Query timed out`: `compute_records` gives the whole set of 466 queries a single 120 s budget (`as_completed(timeout=120)`), and every query not done by then is marked timed out with empty records. Two training runs and a BERT job were sharing CPU and GPU, so how many queries got through depended on load. Example: lr 1e-3 had 36 timeouts in its final dev eval, and lr 3e-4 had 11. When re-run alone, the lr 3e-4 test SQL gave 0 timeouts. The slowest queries are usually malformed ones (for example missing join conditions, which create huge cross products).
- lr 1e-3 reaches higher **SQL EM** (71% at epoch 12) but more of its queries fail or time out, so its Record F1 is lower. lr 3e-4 produces fewer broken queries.
- Because F1 is noisy, early stopping and checkpoint choice partly depend on load. In the clean re-score, lr 1e-3 goes from 72.71 to 77.93 F1 with no change to its SQL, which shows how much of the gap was timeouts and not model quality.
- Learning rate: 1e-4 is too slow for 20 epochs (61.76 F1). 1e-3 learns fastest (best epoch 7, highest SQL EM 63.52) but produces more broken or slow queries. 3e-4 has the best Record F1 and the lowest error rate.
- Practical issue: a query that times out keeps running in its worker thread (`future.cancel()` can't stop a running SQLite call), so the training process can hang at exit. The lr 1e-4 run had to be killed by hand 12 minutes after writing its last file.

### Table 5: error analysis (dev set, lr 3e-4 best checkpoint)

From `part2/analysis/error_analysis.py lr3e-4` (full output with examples in `part2/analysis/errors_lr3e-4.md`; the same for lr 1e-3 in `errors_lr1e-3.md`).

Overall: 344/466 queries return the correct records (set comparison); 279/466 are exact SQL matches.

Each wrong query falls into exactly one of these two groups:
- **Execution error: 51/466.** 40 are `OperationalError` (syntax errors such as unbalanced parentheses, or undefined table aliases), and 11 timed out.
- **Runs but returns wrong records: 78/466.** These are sub-typed below; one query can have several sub-types.

| Error type | Example (dev idx) | Description | Count |
|---|---|---|---|
| Syntax / invalid SQL | idx 2: "list all arrivals from any airport to baltimore on thursday morning arriving before 9am" → the model repeats `AND( flight_1.arrival_time < 900` three times, nests the date clause twice, and ends with unbalanced `)` → `OperationalError: near ")": syntax error` | The decoder loses track of nesting in long queries and repeats clauses. Parentheses don't balance, so SQLite rejects the query. | 40/466 (`OperationalError`) |
| Undefined alias / schema mismatch | idx 47: "show me the ground transportation schedule in philadelphia in the morning on wednesday" → uses `airport_1.airport_code`, `airport_service_1`, `city_2` without adding them to `FROM` → `no such column: airport_1.airport_code` | WHERE clauses copied from similar training queries mention table aliases that were never declared in FROM. | included in the 40 above |
| Wrong literal values (cities, times, dates) | idx 32: "between 430 and 530pm" → model `departure_time >= 1600 AND departure_time <= 530`, gold `>= 1630 AND <= 1730` | Numbers and times in the NL must be turned into the database's 24-hour HHMM integers; the model gets the am/pm conversion and minutes wrong. Also extra city constraints, for example idx 5 "i'm starting from denver" → adds `to … city_name = 'DENVER'` (destination = origin). | 66/466 (records wrong and literal multiset differs) |
| Missing or extra conditions | idx 20: "with the maximum number of stops" → model `flight_1.stops = 1200` instead of the `MAX(stops)` subquery; idx 57: "tomorrow" date constraint attached to `fare_basis` days instead of `flight_days` | Aggregation subqueries and less common constraints are left out or attached to the wrong table. | fewer AND/OR: 26/466; more: 12/466 |
| Wrong SELECT target / wrong table | idx 71: "what does iah mean" → `airline.airline_code = 'IAH'`, gold `airport.airport_code = 'IAH'`; idx 19: "flight times" → selects `departure_time, arrival_time`, gold only `departure_time` | The model mixes up similar entity types (airline vs. airport codes) or how much to return. | wrong SELECT: 17/466; different FROM tables: 30/466 |
| Timeout | (11 queries) | Query not done within the evaluator's shared 120 s budget; usually a query with a cross product. | 11/466 |

Data noise worth mentioning: some gold queries don't match their questions. For example, dev idx 2's gold SQL is `SELECT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD';` for a question about arrivals into Baltimore. No model can get this right.

How the categories were assigned (heuristics in `error_analysis.py`): "wrong literal" = multiset of quoted strings and integers differs from gold; "fewer/more conditions" = count of ` AND `/` OR ` is lower/higher than gold; "wrong tables" = multiset of table names in the top-level FROM differs.

## Q7. Test submission files

| File | Source |
|---|---|
| `part2/results/t5_ft_experiment_test.sql` | copy of `t5_ft_lr3e-4_test.sql` (432 lines, one per line of `data/test.nl`) |
| `part2/records/t5_ft_experiment_test.pkl` | records for the above (432 entries; 36 SQL errors, 0 timeouts) |

The lr 3e-4 run was killed while it was computing the test records, because the first Slurm job ended. The `.pkl` was rebuilt from the saved `.sql` with `utils.compute_records`. The generated SQL itself is from the run.

---

## Extra credit: T5-small from scratch

Same code, run without `--finetune`: `initialize_model` builds `T5ForConditionalGeneration(T5Config.from_pretrained('google-t5/t5-small'))`, so the architecture is identical but every weight is random. Same tokenizer and data processing as the fine-tuned model.

| Attempt | Settings | Outcome |
|---|---|---|
| 1 | lr 1e-3, 80 epochs, patience 10, 2 warmup epochs | Stopped by hand after 2 epochs: with the GPU shared with two other jobs (~4–5 min/epoch), 80 epochs would not finish before the Slurm job ends. Log: `part2/logs/aborted/scr_ec_80ep_aborted.log` |
| 2 | lr 1e-3, 50 epochs, patience 10, 2 warmup epochs | **Diverged.** Train loss went 2.38 → 0.61 (epoch 1), then *rose* to 1.0–1.5 once warmup reached the 1e-3 peak (epochs 2–10). Dev loss 0.61 → 1.05–1.37. 100% of dev outputs were invalid SQL (degenerate repetition such as `date_day_day_day_…`), Record F1 stuck at 11.80 (the floor from gold queries with empty results). Early stopping ended it after epoch 10 because F1 never improved. Log: `part2/logs/aborted/scr_ec_lr1e-3_diverged.log` |
| 3 (final) | lr 3e-4, 50 epochs, patience 50 (effectively off), 2 warmup epochs, cosine, bs 16 | **Converged.** Best dev Record F1 67.73 at epoch 43. Log: `part2/logs/scr_ec.log` |

Lessons so far: (1) without pretrained weights and without gradient clipping, 1e-3 is unstable for t5-small; (2) F1-based early stopping is a bad fit from scratch, because F1 stays at the floor until the model starts producing valid SQL, so patience must be long or off.

### Final run (attempt 3)

Command: `python train_t5.py --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 50 --num_warmup_epochs 2 --scheduler_type cosine --batch_size 16 --test_batch_size 32 --experiment_name ec` (05:37–09:56 UTC, ~4.5 min/epoch on the shared GPU). Same data processing as the fine-tuned model (`<` → `LESS`, BOS 32099, greedy decoding, max 512 new tokens).

Dev curve (every 5 epochs; full table in `part2/analysis/scratch_progress.md`):

| Epoch | Dev loss | Record F1 | Record EM | SQL EM |
|---|---|---|---|---|
| 0 | 0.9310 | 11.80 | 11.80 | 0.00 |
| 5 | 0.1209 | 24.84 | 20.82 | 8.80 |
| 10 | 0.0690 | 34.00 | 29.40 | 19.53 |
| 15 | 0.0569 | 38.54 | 35.19 | 25.11 |
| 20 | 0.0518 | 40.85 | 38.84 | 25.11 |
| 25 | 0.0363 | 54.14 | 49.79 | 37.98 |
| 30 | 0.0313 | 61.36 | 57.30 | 44.64 |
| 35 | 0.0286 | 64.10 | 61.16 | 49.36 |
| 40 | 0.0285 | 66.30 | 63.30 | 50.43 |
| **43 (best)** | **0.0282** | **67.73** | **64.59** | **51.29** |
| 49 (last) | 0.0285 | 67.20 | 63.95 | 51.72 |

Final train loss (epoch 49): 0.0061.

Clean dev re-score of the best checkpoint's outputs (fresh process, `part2/analysis/rescore_ec.py`): F1 **67.73**, Record EM 64.59, SQL EM 51.29, 21 execution errors (4.51%), 0 timeouts, the same as the training log.

Comparison with the fine-tuned model (dev, clean re-scores):

| Model | Record F1 | Record EM | SQL EM | Exec errors |
|---|---|---|---|---|
| T5-small fine-tuned (lr 3e-4, best epoch 12) | 79.61 | 75.54 | 59.87 | 51 (40 SQL errors + 11 timeouts) |
| T5-small from scratch (lr 3e-4, best epoch 43) | 67.73 | 64.59 | 51.29 | 21 |

Error analysis (`part2/analysis/errors_scr_ec.md`, script `part2/analysis/error_analysis_scr.py`; categories overlap, counts out of 466):

| Category | Fine-tuned lr 3e-4 | From scratch |
|---|---|---|
| Record-correct | 344 | 301 |
| SQL exact match | 279 | 239 |
| Execution error | 51 | 21 |
| Wrong records (valid SQL) | 78 | 151 |
| Wrong literal | 66 | 75 |
| Fewer conditions | 26 | 24 |
| More conditions | 12 | 22 |
| Wrong SELECT | 17 | 19 |
| Wrong tables | 30 | 39 |

Observations:
- From scratch needs about 4× as many epochs: F1 stays at the 11.80 floor for 2 epochs, is still only ~41 at epoch 20, and jumps 41 → 54 between epochs 20 and 25 once the model learns the join templates. The fine-tuned model got past 75 within ~10 epochs.
- The scratch model makes **fewer** execution errors (21 vs 51) but many more valid-yet-wrong queries (151 vs 78). It learned the SQL syntax templates well from 4,225 examples, but understands the NL less well: it adds spurious joins and conditions (e.g. "i'm starting from denver" also gets a `to_airport = 'DENVER'` clause; "flights into atlanta's airport" becomes `airport_code = 'ATLANTA'`). Pretraining mainly helps map the question's meaning onto the correct conditions and literals.
- Dev loss flattens from epoch ~35 (0.0286 → 0.0282) while train loss keeps falling to 0.006, so more epochs would add little; the cosine schedule had decayed to ~0 by the end anyway.
- Test: 432 predictions, 20 execution errors (4.6%). Test F1 is shown only on Gradescope.

## Submission checklist

| Deliverable | File | Status |
|---|---|---|
| Q1 | `part1/out_original.txt` | Done |
| Q2 | `part1/out_transformed.txt` | Done |
| Q3 | `part1/out_augmented_original.txt`, `part1/out_augmented_transformed.txt` | Done |
| Q7 | `part2/results/t5_ft_experiment_test.sql`, `part2/records/t5_ft_experiment_test.pkl` | Done (lr 3e-4, final) |
| Extra credit | `part2/results/t5_ft_experiment_ec_test.sql`, `part2/records/t5_ft_experiment_ec_test.pkl` | Done (from scratch, dev F1 67.73) |
| Model checkpoint on Google Drive | `part2/checkpoints/ft_experiments/lr3e-4/best/` (plus `part2/checkpoints/scr_experiments/ec/best/` for extra credit) | Needs uploading by you |
| GitHub link | this repo | |
