"""
Split data into train and eval randomly and build the vocab.
"""
import numpy as np
import random as rd
from os import listdir
from collections import Counter, defaultdict
from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder


PATH_PROCESSED_FODER = '/home/zhaos5/projs/wsd/wsd_data/mimic/find_longform_mimic_processed/'
PATH_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
PATH_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'
PATH_VOCAB = '/home/zhaos5/projs/wsd/wsd_data/mimic/vocab'
PATH_SUBVOCAB = '/home/zhaos5/projs/wsd/wsd_data/mimic/subvocab'
PATH_ABBR = '/home/zhaos5/projs/wsd/wsd_data/mimic/abbr'
PATH_CUI = '/home/zhaos5/projs/wsd/wsd_data/mimic/cui'
PATH_ABBR_MASK = '/home/zhaos5/projs/wsd/wsd_data/mimic/abbr_mask'


rd.seed(1234)
train_lines, eval_lines = [], []
c = Counter()
for file in listdir(PATH_PROCESSED_FODER):
    for line in open(PATH_PROCESSED_FODER + file):
        if 'abbr|' not in line:
            continue
        # Split the train and eval data
        if rd.random() < 0.005:
            eval_lines.append(line)
        else:
            c.update(line.split())
            train_lines.append(line)
open(PATH_TRAIN, 'w').write(''.join(train_lines))
open(PATH_EVAL, 'w').write(''.join(eval_lines))
print('Generate Split files.')

abbrs = set()
cuis = set()
abbr2cui = defaultdict(set)

c = c.most_common()
lines = []
for w, cnt in c:
    if w.startswith('abbr|') and len(w.split('|')) == 3:
        pair = w.split('|')
        abbrs.add(pair[1])
        cuis.add(pair[2])
        abbr2cui[pair[1]].add(pair[2])
        continue
    line = '%s\t%s' % (w, cnt)
    lines.append(line)
open(PATH_VOCAB, 'w').write('\n'.join(lines))
print('Created Vocab %s' % PATH_VOCAB)

sub_word_feeder = {}
for line in open(PATH_VOCAB):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    sub_word_feeder[word] = cnt

c = Counter(sub_word_feeder)
sub_word = SubwordTextEncoder.build_to_target_size(8000, c, 1, 1e5,
                                                               num_iterations=10)
for i, subtoken_string in enumerate(sub_word._all_subtoken_strings):
    if subtoken_string in text_encoder.RESERVED_TOKENS_DICT:
        sub_word._all_subtoken_strings[i] = subtoken_string + "_"
sub_word.store_to_file(PATH_SUBVOCAB)
print('Created SubVocab %s' % PATH_SUBVOCAB)

# Prepare abbr/cui/mask
abbrs = list(abbrs)
cuis = list(cuis)
abbr2id = dict(zip(abbrs, range(len(abbrs))))
cui2id = dict(zip(cuis, range(len(cuis))))
open(PATH_ABBR, 'w').write('\n'.join(abbrs))
open(PATH_CUI, 'w').write('\n'.join(cuis))

mask = np.zeros((len(abbrs), len(cuis)), dtype=np.bool)
for abbr in abbr2cui:
    for cui in abbr2cui[abbr]:
        mask[abbr2id[abbr], cui2id[cui]] = True
np.savetxt(PATH_ABBR_MASK, mask)
