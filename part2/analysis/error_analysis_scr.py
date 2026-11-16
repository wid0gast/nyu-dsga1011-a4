import pickle, re, sys, collections
from transformers import T5TokenizerFast
exp = sys.argv[1]
nl = open('data/dev.nl').read().splitlines()
gt = [l.strip() for l in open('data/dev.sql')]
pr = [l.strip() for l in open(f'results/t5_scr_{exp}_dev.sql')]
gr, _ = pickle.load(open('records/ground_truth_dev.pkl','rb'))
mr, me = pickle.load(open(f'analysis/rescore/t5_scr_{exp}_dev.pkl','rb'))
N = len(gt)
def lits(q): return collections.Counter(re.findall(r"'[^']*'|\b\d+\b", q))
def tables(q):
    m = re.search(r'FROM (.*?) WHERE', q)
    return collections.Counter(re.findall(r'\b([a-z_]+) [a-z_]+_\d+\b', m.group(1))) if m else collections.Counter()
def sel(q):
    m = re.match(r'SELECT (.*?) FROM', q); return m.group(1) if m else ''
cats = collections.defaultdict(list)
for i in range(N):
    rec_ok = set(gr[i]) == set(mr[i])
    if me[i]:
        kind = 'timeout' if 'timed out' in me[i] else me[i].split(':')[0]
        cats['exec_error'].append(i); cats['exec_error:'+kind].append(i); continue
    if rec_ok: continue
    cats['wrong_records'].append(i)
    if lits(gt[i]) != lits(pr[i]): cats['wrong_literal'].append(i)
    if sel(gt[i]) != sel(pr[i]): cats['wrong_select'].append(i)
    if tables(gt[i]) != tables(pr[i]): cats['wrong_tables'].append(i)
    gw = gt[i].count(' AND ') + gt[i].count(' OR '); pw = pr[i].count(' AND ') + pr[i].count(' OR ')
    if pw < gw: cats['fewer_conditions'].append(i)
    if pw > gw: cats['more_conditions'].append(i)
print(f'## Error analysis for {exp} (dev, N={N})')
print(f'record-correct: {sum(set(gr[i])==set(mr[i]) for i in range(N))}/{N}; SQL exact: {sum(g==p for g,p in zip(gt,pr))}/{N}')
for k in sorted(cats): print(f'{k}: {len(cats[k])}/{N}')
for k in ['exec_error','wrong_literal','wrong_select','fewer_conditions','more_conditions','wrong_tables']:
    print(f'\n### examples: {k}')
    for i in cats[k][:3]:
        print(f'- idx {i}\n  NL: {nl[i]}\n  GT: {gt[i]}\n  PR: {pr[i]}')
        if me[i]: print(f'  ERR: {me[i]}')
        else:
            gl, pl = lits(gt[i]), lits(pr[i])
            print(f'  literal diff: GT-only {dict(gl-pl)} | PR-only {dict(pl-gl)}')
