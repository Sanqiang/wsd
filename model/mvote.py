"""Simple majority vote impl"""
from collections import Counter, defaultdict


PATH_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
PATH_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'

assign_collect = defaultdict(Counter)
lid = 0
for line in open(PATH_TRAIN):
    ents = [w for w in line.split() if w.startswith('abbr|')]
    for ent in ents:
        items = ent.split('|')
        abbr = items[1]
        cui = items[2]
        assign_collect[abbr].update([cui])
    lid += 1
    if lid % 10000 == 0:
        print('Processed %s lines' % lid)

assign_map = {}
correct_cnt, total_cnt = 0.0, 0.0
for abbr in assign_collect:
    assign_map[abbr] = assign_collect[abbr].most_common(1)[0][0]

for line in open(PATH_EVAL):
    ents = [w for w in line.split() if w.startswith('abbr|')]
    for ent in ents:
        items = ent.split('|')
        abbr = items[1]
        cui = items[2]
        if abbr in assign_map:
            pred_cui = assign_map[abbr]
            if cui == pred_cui:
                correct_cnt += 1.0
        total_cnt += 1.0

acc = correct_cnt / total_cnt
print('Accuray = %s' % acc)

