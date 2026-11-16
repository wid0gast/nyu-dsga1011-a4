import os, pickle, sys
sys.path.insert(0, '.')
from utils import read_queries, compute_records, compute_metrics
qs = read_queries('results/t5_scr_ec_dev.sql')
recs, errs = compute_records(qs)
pickle.dump((recs, errs), open('analysis/rescore/t5_scr_ec_dev.pkl', 'wb'))
sql_em, rec_em, f1, _ = compute_metrics('data/dev.sql', 'results/t5_scr_ec_dev.sql', 'records/ground_truth_dev.pkl', 'analysis/rescore/t5_scr_ec_dev.pkl')
n_err = sum(1 for e in errs if e); n_to = sum(1 for e in errs if 'imeout' in str(e))
print(f'F1 {f1*100:.2f} RecEM {rec_em*100:.2f} SQLEM {sql_em*100:.2f} errors {n_err} timeouts {n_to}', flush=True)
os._exit(0)
