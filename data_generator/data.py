from data_generator.vocab import Vocab
from util.constant import NONTAR, UNK, BOS, EOS, PAD
import random as rd
import os
import pickle
from collections import defaultdict


class Data:
    def __init__(self, model_config):
        self.model_config = model_config
        # For Abbr
        self.populate_abbr()
        # For Context
        self.voc = Vocab(model_config, model_config.voc_file)

    def populate_abbr(self):
        self.abbr2id, self.id2abbr = {}, []
        self.sense2id, self.id2sense = {}, []
        self.id2abbr = [abbr.strip() for abbr in
                        open(self.model_config.abbr_file).readlines()]
        self.abbr2id = dict(zip(self.id2abbr, range(len(self.id2abbr))))
        self.id2sense = [cui.strip() for cui in
                         open(self.model_config.cui_file).readlines()]
        self.sense2id = dict(zip(self.id2sense, range(len(self.id2sense))))
        self.sen_cnt = len(self.id2sense)

    # Deprecated
    def populate_abbr_deprecated(self):
        def update(item, item2id, id2item):
            if item not in item2id:
                item2id[item] = len(id2item)
                id2item.append(item)

        s_i = 0
        self.abbrs_pos = {}
        self.abbr2id, self.id2abbr = {}, []
        self.sense2id, self.id2sense = {}, []
        for line in open(self.model_config.abbr_common_file):
            items = line.strip().split('|')
            abbr = items[0]
            update(abbr, self.abbr2id, self.id2abbr)
            senses = items[1].split()

            abbr_id = self.abbr2id[abbr]
            if abbr_id not in self.abbrs_pos:
                self.abbrs_pos[abbr_id] = {}
                self.abbrs_pos[abbr_id]['s_i'] = s_i
                self.abbrs_pos[abbr_id]['e_i'] = s_i + len(senses)
                s_i = s_i + len(senses)
            for sense in senses:
                update(abbr + '|' + sense, self.sense2id, self.id2sense)
        self.sen_cnt = s_i

        self.abbrs_filterout = set()
        for line in open(self.model_config.abbr_rare_file):
            self.abbrs_filterout.add(line.strip())

    def process_line(self, line):
        contexts = []
        targets = []
        words = line.split()
        contexts.extend(self.voc.encode(BOS))
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
                    targets.append([id, abbr_id, sense_id])
            else:
                wid = self.voc.encode(word)
            contexts.extend(wid)
        contexts.extend(self.voc.encode(EOS))

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
                cur_contexts.extend(self.voc.encode(PAD) * num_pad)
            assert len(cur_contexts) == self.model_config.max_context_len

            obj = {
                'contexts': cur_contexts,
                'target': target,
                'line': line
            }
            objs.append(obj)
        return objs

    def populate_data(self, path):
        # if os.path.exists(self.model_config.train_pickle):
        #     with open(self.model_config.train_pickle, 'rb') as inv_file:
        #         self.datas = pickle.load(inv_file)
        self.datas = []
        line_id = 0
        for line in open(path):
            objs = self.process_line(line)
            self.datas.extend(objs)
            line_id += 1
            if line_id % 10000 == 0:
                print('Process %s lines.' % line_id)
            # break
        # with open(self.model_config.train_pickle, 'wb') as output_file:
        #     pickle.dump(self.datas, output_file)


class TrainData(Data):
    def __init__(self, model_config):
        Data.__init__(self, model_config)
        if not model_config.it_train:
            self.populate_data(self.model_config.train_file)
            print('Finished Populate Data with %s samples.' % str(len(self.datas)))
        else:
            self.data_it = self.get_sample_it()
            self.size = self.get_size()
            print('Finished Data Iter with %s samples.' % str(self.size))

    def get_size(self):
        return len(open(self.model_config.train_file, encoding='utf-8').readlines())

    def get_sample(self):
        i = rd.sample(range(len(self.datas)), 1)[0]
        return self.datas[i]

    def get_sample_it(self):
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

            objs = self.process_line(line)
            if len(objs) > 0:
                for obj in objs:
                    yield obj
            else:
                print('error obj:%s' % objs)
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