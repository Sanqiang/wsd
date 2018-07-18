from data_generator.vocab import Vocab
from util.constant import NONTAR, UNK, BOS, EOS, PAD
import random as rd
from collections import defaultdict


class Data:
    def __init__(self, model_config):
        self.model_config = model_config
        # For Abbr
        self.populate_abbr()
        # For Context
        self.voc = Vocab(model_config, model_config.voc_file)

    def populate_abbr(self):
        def update(item, item2id, id2item):
            if item not in item2id:
                item2id[item] = len(id2item)
                id2item.append(item)

        s_i = 0
        self.abbrs_pos_s, self.abbrs_pos_e = [], []
        self.abbr2id, self.id2abbr = {}, []
        self.sense2id, self.id2sense = {}, []
        for line in open(self.model_config.abbr_common_file):
            items = line.strip().split('|')
            abbr = items[0]
            update(abbr, self.abbr2id, self.id2abbr)
            senses = items[1].split()

            self.abbrs_pos_s.append(s_i)
            self.abbrs_pos_e.append(s_i + len(senses))
            s_i = s_i + len(senses)

            # if abbr_id not in self.abbrs_pos:
            #     self.abbrs_pos[abbr_id] = {}
            #     self.abbrs_pos[abbr_id]['s_i'] = s_i
            #     self.abbrs_pos[abbr_id]['e_i'] = s_i + len(senses)
            #     s_i = s_i + len(senses)
            for sense in senses:
                update(abbr + '|' + sense, self.sense2id, self.id2sense)
        self.sen_cnt = s_i

        self.abbrs_filterout = set()
        for line in open(self.model_config.abbr_rare_file):
            self.abbrs_filterout.add(line.strip())

        # def update(item, item2id, id2item):
        #     if item not in item2id:
        #         item2id[item] = len(id2item)
        #         id2item.append(item)
        #
        # self.abbr2id, self.id2abbr, self.senses = {}, [], []
        # for line in open(self.model_config.abbr_common_file):
        #     items = line.strip().split('|')
        #     abbr = items[0]
        #     update(abbr, self.abbr2id, self.id2abbr)
        #     senses = items[1].split()
        #     sense2id, id2sense = {}, []
        #     for sense in senses:
        #         update(sense, sense2id, id2sense)
        #     self.senses.append(
        #         {'sense2id': sense2id,
        #         'id2sense': id2sense})
        #
        # self.abbrs_filterout = set()
        # for line in open(self.model_config.abbr_rare_file):
        #     self.abbrs_filterout.add(line.strip())

    def process_line(self, line):
        window_size = int(self.model_config.max_context_len / 2)

        def populate_contexts(contexts, wid, cnt=1):
            if isinstance(wid, int):
                wid = [wid]
            contexts.extend(wid * cnt)

        checker = set()
        contexts = []
        targets = []
        words = line.split()
        populate_contexts(contexts, self.voc.encode(BOS))
        for step, word in enumerate(words):
            if word.startswith('abbr|'):
                pair = word.split('|')
                abbr = pair[1]
                if abbr in self.abbrs_filterout:
                    continue
                sense = pair[2]

                if abbr not in self.abbr2id:
                    continue
                abbr_id = self.abbr2id[abbr]
                wid = abbr_id + self.voc.vocab_size()
                if abbr_id not in checker:
                    if abbr + '|' + sense in self.sense2id:
                        sense_id = self.sense2id[abbr + '|' + sense]
                        targets.append([step, wid, sense_id])
                        checker.add(abbr_id)
            else:
                wid = self.voc.encode(word)
            populate_contexts(contexts, wid)
        populate_contexts(contexts, self.voc.encode(EOS))

        objs = []
        for target in targets:
            step = target[0]
            if step < window_size:
                left_idx = 0
            else:
                left_idx = step - window_size

            if step + window_size > len(contexts):
                right_idx = len(contexts)
            else:
                right_idx = step + window_size

            cur_context = contexts[left_idx:right_idx]
            obj = {
                'contexts': cur_context,
                'targets': targets,
                'line': line
            }
            objs.append(obj)
        return objs

    def populate_data(self, path):
        self.datas = []
        for line in open(path):
            objs = self.process_line(line)
            self.datas.extend(objs)
            # break


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

            obj = self.process_line(line)
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