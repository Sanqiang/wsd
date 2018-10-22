from data_generator.vocab import Vocab
from util.constant import NONTAR, UNK, BOS, EOS, PAD
import random as rd
import os
import numpy as np
import pickle
from collections import defaultdict
import nltk
from nltk.corpus import stopwords


STOP_WORDS_SET = set(stopwords.words('english'))


class Data:
    def __init__(self, model_config):
        self.model_config = model_config
        # For Abbr
        self.populate_abbr()
        # For Context
        self.voc = Vocab(model_config, model_config.voc_file)

        if 'stype' in model_config.extra_loss or 'def' in model_config.extra_loss:
            self.populate_cui()

    def populate_abbr(self):
        self.id2abbr = [abbr.strip() for abbr in
                        open(self.model_config.abbr_file).readlines()]
        self.abbr2id = dict(zip(self.id2abbr, range(len(self.id2abbr))))
        self.id2sense = [cui.strip() for cui in
                         open(self.model_config.cui_file).readlines()]
        self.sense2id = dict(zip(self.id2sense, range(len(self.id2sense))))
        self.sen_cnt = len(self.id2sense)

    def populate_cui(self):
        self.stype2id, self.id2stype = {}, []
        self.id2stype = [stype.split('\t')[0].lower()
                         for stype in open(self.model_config.stype_voc_file).readlines()]
        self.stype2id = dict(zip(self.id2stype, range(len(self.id2stype))))

        self.cui2stype = {}
        with open(self.model_config.cui_extra_pkl, 'rb') as cui_file:
            cui_extra = pickle.load(cui_file)
            for cui in cui_extra:
                info = cui_extra[cui]
                self.cui2stype[cui] = self.stype2id[info[1].lower()]

        self.cui2def = {}
        with open(self.model_config.cui_extra_pkl, 'rb') as cui_file:
            cui_extra = pickle.load(cui_file)
            for cui in cui_extra:
                info = cui_extra[cui]
                if self.model_config.subword_vocab_size <= 0:
                    definition = [self.voc.encode(w) for w in info[0].lower().strip().split()]
                else:
                    definition = self.voc.encode(info[0].lower().strip())

                if len(definition) > self.model_config.max_def_len:
                    definition = definition[:self.model_config.max_def_len]
                else:
                    num_pad = self.model_config.max_def_len - len(definition)
                    if self.model_config.subword_vocab_size <= 0:
                        definition.extend([self.voc.encode(PAD)] * num_pad)
                    else:
                        definition.extend(self.voc.encode(PAD) * num_pad)
                assert len(definition) == self.model_config.max_def_len
                self.cui2def[cui] = definition

        np_mask = np.loadtxt(self.model_config.abbr_mask_file)
        self.cuiud2abbrid = defaultdict(list)
        for abbrid in range(len(self.id2abbr)):
            cuiids = list(np.where(np_mask[abbrid] == 1)[0])
            for cuiid in cuiids:
                self.cuiud2abbrid[cuiid].append(abbrid)

    def process_line(self, line, line_id):
        contexts = []
        targets = []
        words = line.split()

        # if self.model_config.subword_vocab_size <= 0:
        #     contexts.append(self.voc.encode(BOS))
        # else:
        #     contexts.extend(self.voc.encode(BOS))

        for id, word in enumerate(words):
            if word.startswith('abbr|'):
                pair = word.split('|')
                abbr = pair[1]
                # if abbr in self.abbrs_filterout:
                #     continue
                sense = pair[2]

                if 'add_abbr' in self.model_config.voc_process:
                    wid = self.voc.encode(abbr)
                else:
                    wid = self.voc.encode(NONTAR)
                if abbr not in self.abbr2id:
                    continue
                abbr_id = self.abbr2id[abbr]
                if sense in self.sense2id:
                    sense_id = self.sense2id[sense]
                    targets.append([id, abbr_id, sense_id, line_id])
            else:
                wid = self.voc.encode(word)

            if self.model_config.subword_vocab_size <= 0:
                contexts.append(wid)
            else:
                contexts.extend(wid)

        # if self.model_config.subword_vocab_size <= 0:
        #     contexts.append(self.voc.encode(EOS))
        # else:
        #     contexts.extend(self.voc.encode(EOS))

        objs = []
        window_size = int(self.model_config.max_context_len / 2)
        for target in targets:
            step = target[0]
            extend_size = 0
            if step < window_size:
                left_idx = 0
                extend_size = window_size - step
            else:
                left_idx = step - window_size

            if step + window_size > len(contexts):
                right_idx = len(contexts)
            else:
                right_idx = min(
                    step + window_size + extend_size, len(contexts))

            cur_contexts = contexts[left_idx:right_idx]

            if len(cur_contexts) > self.model_config.max_context_len:
                cur_contexts = cur_contexts[:self.model_config.max_context_len]
            else:
                num_pad = self.model_config.max_context_len - len(cur_contexts)
                if self.model_config.subword_vocab_size <= 0:
                    cur_contexts.extend([self.voc.encode(PAD)] * num_pad)
                else:
                    cur_contexts.extend(self.voc.encode(PAD) * num_pad)
            assert len(cur_contexts) == self.model_config.max_context_len

            obj = {
                'contexts': cur_contexts,
                'target': target,
                'line': line,
            }

            objs.append(obj)
        return objs

    def populate_data(self, path):
        self.datas = []
        line_id = 0
        for line in open(path):
            objs = self.process_line(line, line_id)
            self.datas.extend(objs)
            line_id += 1
            if line_id % 10000 == 0:
                print('Process %s lines.' % line_id)


class TrainData(Data):
    def __init__(self, model_config):
        Data.__init__(self, model_config)
        if not model_config.it_train:
            self.populate_data(self.model_config.train_file)
            print('Finished Populate Data with %s samples.' % str(len(self.datas)))
        else:
            self.data_it = self.get_sample_it()
            if self.model_config.extra_loss:
                self.data_it_cui = self.get_cui_sample_it()
            self.size = self.get_size()
            print('Finished Data Iter with %s samples.' % str(self.size))

    def get_size(self):
        return len(open(self.model_config.train_file, encoding='utf-8').readlines())

    def get_sample(self):
        i = rd.sample(range(len(self.datas)), 1)[0]
        return self.datas[i]

    def get_cui_sample_it(self):
        """Get feed data from cui extra (e.g. def, stype)"""
        while True:
            i = rd.sample(range(len(self.id2sense)), 1)[0]
            cui = self.id2sense[i]
            sdef = self.cui2def[cui]
            stype = self.cui2stype[cui]
            for abbr_id in self.cuiud2abbrid[i]:
                obj = {
                    'abbr_id': abbr_id,
                    'cui_id': i,
                    'def': sdef,
                    'stype': stype}
                yield obj

    def prepare_data_for_masked_lm(self, line, objs):
        if not self.model_config.lm_mask_rate:
            return objs
        # print('Use Masked LM with rate %s' % self.model_config.lm_mask_rate)

        masked_cnt = len(objs)
        words = line.split()
        masked_contexts = []
        masked_words = []
        masked_contexts.extend(self.voc.encode(BOS))
        masked_idxs = rd.sample(range(len(words)),
                                1 + int(len(words) * self.model_config.lm_mask_rate))
        for id, word in enumerate(words):
            if word.startswith('abbr|'):
                pair = word.split('|')
                abbr = pair[1]
                if 'add_abbr' in self.model_config.voc_process:
                    wid = self.voc.encode(abbr)
                else:
                    wid = self.voc.encode(NONTAR)
            else:
                wid = self.voc.encode(word)
            if masked_cnt and word not in STOP_WORDS_SET and not word.startswith('abbr|') and id in masked_idxs:
                masked_contexts.extend(self.voc.encode(PAD))
                twid = wid
                if len(twid) > self.model_config.max_subword_len:
                    twid = twid[:self.model_config.max_subword_len]
                else:
                    num_pad = self.model_config.max_subword_len - len(twid)
                    twid.extend(self.voc.encode(PAD) * num_pad)
                masked_words.append((id, twid))
                masked_cnt -= 1
            else:
                masked_contexts.extend(wid)
        masked_contexts.extend(self.voc.encode(EOS))
        window_size = int(self.model_config.max_context_len / 2)
        if not masked_words:
            return objs
        for i in range(len(objs)):
            masked_word = masked_words[i % len(masked_words)]
            step = masked_word[0]
            extend_size = 0
            if step < window_size:
                left_idx = 0
                extend_size = window_size - step
            else:
                left_idx = step - window_size

            if step + window_size > len(masked_contexts):
                right_idx = len(masked_contexts)
            else:
                right_idx = min(
                    step + window_size + extend_size, len(masked_contexts))

            cur_masked_contexts = masked_contexts[left_idx:right_idx]

            if len(cur_masked_contexts) > self.model_config.max_context_len:
                cur_masked_contexts = cur_masked_contexts[:self.model_config.max_context_len]
            else:
                num_pad = self.model_config.max_context_len - len(cur_masked_contexts)
                cur_masked_contexts.extend(self.voc.encode(PAD) * num_pad)
            assert len(cur_masked_contexts) == self.model_config.max_context_len

            objs[i]['cur_masked_contexts'] = cur_masked_contexts
            objs[i]['masked_words'] = masked_word
        return objs

    def get_sample_it(self):
        """Get ffed data from task"""
        i = 0
        f = open(self.model_config.train_file)
        while True:
            if i >= self.size:
                i = 0
                f = open(self.model_config.train_file)

            line = f.readline()
            if rd.random() < 0.5 or i >= self.size:
                i += 1
                continue

            objs = self.process_line(line, i)
            if len(objs) > 0:
                objs = self.prepare_data_for_masked_lm(line, objs)
                for obj in objs:
                    yield obj
            i += 1


class EvalData(Data):
    def __init__(self, model_config):
        Data.__init__(self, model_config)
        self.populate_data(self.model_config.eval_file)
        self.i = 0
        print('Finished Populate Data with %s samples.' % str(len(self.datas)))

    def get_sample(self):
        if self.i < len(self.datas):
            data = self.datas[self.i]
            self.i += 1
            return data
        else:
            return None

    def reset(self):
        self.i = 0