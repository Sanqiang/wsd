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

    def populate_data(self, path):
        self.datas = []
        for line in open(path):
            checker = set()
            contexts = []
            targets = []
            words = line.split()
            contexts.extend(self.voc.encode(BOS))
            for id, word in enumerate(words):
                if word.startswith('abbr|'):
                    pair = word.split('|')
                    abbr = pair[1]
                    if abbr in self.abbrs_filterout:
                        continue
                    sense = pair[2]
                    wid = self.voc.encode(NONTAR)
                    if abbr not in self.abbrs_pos:
                        if abbr not in self.abbr2id:
                            continue
                        abbr_id = self.abbr2id[abbr]
                        if abbr_id not in checker and len(targets) < self.model_config.max_abbrs:
                            if abbr + '|' + sense in self.sense2id:
                                sense_id = self.sense2id[abbr + '|' + sense]
                                targets.append([id, abbr_id, sense_id])
                                checker.add(abbr_id)
                else:
                    wid = self.voc.encode(word)
                contexts.extend(wid)
            contexts.extend(self.voc.encode(EOS))

            if len(contexts) > self.model_config.max_context_len:
                contexts = contexts[:self.model_config.max_context_len]
            else:
                num_pad = self.model_config.max_context_len - len(contexts)
                contexts.extend(self.voc.encode(PAD) * num_pad)
            assert len(contexts) == self.model_config.max_context_len

            if len(targets) > self.model_config.max_abbrs:
                targets = targets[:self.model_config.max_abbrs]
            else:
                num_pad = self.model_config.max_abbrs - len(targets)
                targets.extend([[0,0,0]] * num_pad)
            assert len(targets) == self.model_config.max_abbrs

            self.datas.append({
                'contexts': contexts,
                'targets': targets
            })
            #break
            
class TrainData(Data):
    def __init__(self, model_config):
        Data.__init__(self, model_config)
        self.populate_data(self.model_config.train_file)
        print('Finished Populate Data with %s samples.' % str(len(self.datas)))

    def get_sample(self):
        i = rd.sample(range(len(self.datas)), 1)[0]
        return self.datas[i]


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
