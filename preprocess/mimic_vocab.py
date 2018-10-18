"""
Split data into train and eval randomly and build the vocab.
"""
import numpy as np
import random as rd
from os import listdir
from collections import Counter, defaultdict
from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder


ROOT_PATH = "/home/mengr/Project/wsd/wsd_data/mimic/"
# ROOT_PATH = "/Users/memray/Project/upmc_wsd/wsd_data/mimic/"
PATH_PROCESSED_FODER = ROOT_PATH + 'processed/'
PATH_TRAIN = ROOT_PATH + 'train'
PATH_EVAL = ROOT_PATH + 'eval'
PATH_VOCAB = ROOT_PATH + 'vocab'
PATH_SUBVOCAB = ROOT_PATH + 'subvocab'
PATH_ABBR = ROOT_PATH + 'abbr'
PATH_CUI = ROOT_PATH + 'cui'
PATH_ABBR_MASK = ROOT_PATH + 'abbr_mask'


rd.seed(1234)
train_lines, eval_lines = [], []
c = Counter()

# read regex-processed lines and split into train/test randomly
for file in listdir(PATH_PROCESSED_FODER):
    for line in open(PATH_PROCESSED_FODER + file):
        # ignore lines with no regex-replaced abbr tags
        if 'abbr|' not in line:
            continue
        # Split the train and eval data randomly
        if rd.random() <= 0.025:
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

# Build vocab by iterating each word appearing in the text
for w, cnt in c:
    # special tokens for labels,
    if w.startswith('abbr|') and len(w.split('|')) >= 3:
        pair = w.split('|')
        abbrs.add(pair[1])
        cuis.add(pair[2])
        abbr2cui[pair[1]].add(pair[2])
        continue
    line = '%s\t%s' % (w, cnt)
    lines.append(line)
open(PATH_VOCAB, 'w').write('\n'.join(lines))
print('Created Vocab %s' % PATH_VOCAB)

# Create subword vocab
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
print('Generate abbr/cui/mask.')