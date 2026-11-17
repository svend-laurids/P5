import itertools, random
from collections import Counter

SPACE = {
    'gnn.layers_mp':       [2, 4, 6, 8],
    'gnn.dim_inner':       [32, 64, 128],
    'gnn.agg':             ['add', 'mean'],
    'gnn.stage_type':      ['stack', 'skipsum'],
    'model.graph_pooling': ['add', 'mean'],
    'gnn.dropout':         [0.0, 0.1],
    'optim.base_lr':       [0.01, 0.001],
}
MODES = ['shared', 'sum', 'product']
N, EPOCHS, SEED = 100, 30, 0

all_cfgs = list(itertools.product(*SPACE.values()))      # 384 combinations
random.Random(SEED).shuffle(all_cfgs)
sample = all_cfgs[:N]

lines = []
for i, vals in enumerate(sample):
    base = ' '.join(f'{k} {v}' for k, v in zip(SPACE, vals))
    for m in MODES:
        lines.append(f'out_dir results/search/c{i:03d}_{m} gnn.weight_sharing {m} '
                     f'optim.max_epoch {EPOCHS} qm9.subset 0 tensorboard_agg False {base}')

with open('configs/search_list.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(len(all_cfgs), 'possible configs,', len(sample), 'sampled,', len(lines), 'runs')
for k in SPACE:                                           # balance check
    print(k, dict(Counter(v for cfg in sample for kk, v in zip(SPACE, cfg) if kk == k)))