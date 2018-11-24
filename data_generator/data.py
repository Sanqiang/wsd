from baseline import dataset_helper
from data_generator.vocab import Vocab
from util.constant import NONTAR, UNK, BOS, EOS, PAD
import random as rd
import os
import numpy as np
import pickle
from collections import defaultdict
from nltk.corpus import stopwords

import tensorflow as tf


STOP_WORDS_SET = set(stopwords.words('english'))



def get_feed(data_feeds, data_loader, model_config, is_train):
    """
    Create a new batch of input_feed by loading from data iteratively
    The key of each element in feed_dict is given by the model when generating the graph
    TODO, because it is device-dependent
    :param data_feeds: providing the keys for generating feed_dict, number of data_feeds depends on number of devices
    :param data_loader: a TrainData/EvalData object
    :param model_config:
    :param is_train:
    :return:
    """
    input_feed_dict = {}
    excluded_example_count = 0

    for data_feed in data_feeds:
        tmp_contexts, tmp_targets, tmp_lines = [], [], []
        tmp_masked_contexts, tmp_masked_words = [], []
        example_in_batch_count = 0
        while example_in_batch_count < model_config.batch_size:
            if model_config.it_train:
                example = next(data_loader.data_it, None)
            else:
                # TODO: not tested
                example = data_loader.get_sample()

            # mainly used in evaluation when testset file reaches EOF, create a dummy input to feed model
            if example is None:
                example = {}
                example['contexts'] = [0] * model_config.max_context_len
                example['target'] = {'pos_id': 0,
                                     'abbr_id': 0,
                                     'abbr': None,
                                     'sense_id': 0,
                                     'sense': None,
                                     'line_id': data_loader.size,
                                     'inst_id': 0
                                     }
                example['line'] = ''
                # sample['def'] = [0] * model_config.max_def_len
                # sample['stype'] = 0
                excluded_example_count += 1  # Assume eval use single GPU

            # print(example_in_batch_count)
            # print(excluded_example_count)
            # print(example)

            tmp_contexts.append(example['contexts'])
            tmp_targets.append(example['target'])
            tmp_lines.append(example['line'])
            # print('input:\t%s\t%s.' % (sample['line'], sample['target']))
            if model_config.lm_mask_rate and 'cur_masked_contexts' in example:
                tmp_masked_contexts.append(example['cur_masked_contexts'])
                tmp_masked_words.append(example['masked_words'])

            # print('done one example, current len(batch)=%d' % len(tmp_contexts))
            example_in_batch_count += 1

        for step in range(model_config.max_context_len):
            input_feed_dict[data_feed['contexts'][step].name] = [
                tmp_contexts[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

        if model_config.hub_module_embedding:
            input_feed_dict[data_feed['text_input'].name] = [
                tmp_lines[batch_idx]
                for batch_idx in range(model_config.batch_size)]

        input_feed_dict[data_feed['abbr_inp'].name] = [
            tmp_targets[batch_idx]['abbr_id']
            for batch_idx in range(model_config.batch_size)
        ]

        input_feed_dict[data_feed['sense_inp'].name] = [
            tmp_targets[batch_idx]['sense_id']
            for batch_idx in range(model_config.batch_size)
        ]

        if model_config.lm_mask_rate and tmp_masked_contexts:
            i = 0
            while len(tmp_masked_contexts) < model_config.batch_size:
                tmp_masked_contexts.append(tmp_masked_contexts[i % len(tmp_masked_contexts)])
                tmp_masked_words.append(tmp_masked_words[i % len(tmp_masked_contexts)])
                i += 1

            for step in range(model_config.max_context_len):
                input_feed_dict[data_feed['masked_contexts'][step].name] = [
                    tmp_masked_contexts[batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

            for step in range(model_config.max_subword_len):
                input_feed_dict[data_feed['masked_words'][step].name] = [
                    tmp_masked_words[batch_idx][1][step]
                    for batch_idx in range(model_config.batch_size)]

    return input_feed_dict, excluded_example_count, tmp_targets


def get_feed_cui(obj, data, model_config):
    """Feed the CUI model."""
    input_feed = {}
    tmp_extra_cui_def, tmp_extra_cui_stype, tmp_cuiid, tmp_abbrid = [], [], [], []
    cnt = 0
    while cnt < model_config.batch_size:
        sample = next(data.data_it_cui)
        tmp_cuiid.append(sample['cui_id'])
        tmp_abbrid.append(sample['abbr_id'])
        if 'def' in model_config.extra_mode:
            tmp_extra_cui_def.append(sample['def'])
        if 'stype' in model_config.extra_mode:
            tmp_extra_cui_stype.append(sample['stype'])
        cnt += 1

    input_feed[obj['abbr_inp'].name] = [
        tmp_abbrid[batch_idx]
        for batch_idx in range(model_config.batch_size)
    ]
    input_feed[obj['sense_inp'].name] = [
        tmp_cuiid[batch_idx]
        for batch_idx in range(model_config.batch_size)
    ]

    if 'def' in model_config.extra_mode:
        for step in range(model_config.max_def_len):
            input_feed[obj['def'][step].name] = [
                tmp_extra_cui_def[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

    if 'stype' in model_config.extra_mode:
        input_feed[obj['stype'].name] = [
            tmp_extra_cui_stype[batch_idx]
            for batch_idx in range(model_config.batch_size)
        ]

    return input_feed


def get_session_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.log_device_placement = True
    config.gpu_options.allocator_type = "BFC"
    config.gpu_options.allow_growth = True
    return config


class Data:
    def __init__(self, model_config):
        self.model_config = model_config
        # For Abbr
        self.populate_abbr()
        # For Context
        self.voc = Vocab(model_config, model_config.voc_file)

        if 'stype' in model_config.extra_mode or 'def' in model_config.extra_mode:
            self.populate_cui()

    def populate_abbr(self):
        self.id2abbr = [abbr.strip() for abbr in
                        open(self.model_config.abbr_file).readlines()]
        self.abbr2id = dict(zip(self.id2abbr, range(len(self.id2abbr))))
        self.id2sense = [cui.strip() for cui in
                         open(self.model_config.cui_file).readlines()]
        self.sense2id = dict(zip(self.id2sense, range(len(self.id2sense))))
        self.sen_cnt = len(self.id2sense)

        if self.model_config.extra_mode:
            self.id2abbr.append(con)


    def populate_cui(self):
        self.stype2id, self.id2stype = {}, []
        self.id2stype = [stype.split('\t')[0].lower()
                         for stype in open(self.model_config.stype_voc_file).readlines()]
        self.id2stype.append('unk')
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

    def process_line(self, line, line_id, inst_id):
        '''
        Process each line and return tokens. Each line may contain multiple labels.
        :param line:
        :param line_id:
        :return:
        '''
        contexts = []
        targets = []
        words = line.split()

        # if self.model_config.subword_vocab_size <= 0:
        #     contexts.append(self.voc.encode(BOS))
        # else:
        #     contexts.extend(self.voc.encode(BOS))

        for pos_id, word in enumerate(words):
            if word.startswith('abbr|'):
                abbr, sense, long_form = dataset_helper.process_abbr_token(word)

                if 'add_abbr' in self.model_config.voc_process:
                    wid = self.voc.encode(abbr)
                else:
                    wid = self.voc.encode(NONTAR)

                # Set abbr_id or sense_id to None if either one is not in our vocab/inventory
                if abbr not in self.abbr2id:
                    # print('abbr %s not found in abbr vocab (size=%d), ignore this data example'
                    #       % (abbr, len(self.abbr2id)))
                    abbr_id = 0
                else:
                    abbr_id = self.abbr2id[abbr]

                if sense in self.sense2id:
                    sense_id = self.sense2id[sense]
                else:
                    sense_id = 0
                    # print('sense %s is not in sense inventory (size=%d), ignore this data example'
                    #       % (sense, len(self.sense2id)))

                # return each target as a dict instead of a list
                targets.append(
                    {'pos_id': pos_id,
                     'abbr_id': abbr_id,
                     'abbr': abbr,
                     'sense_id': sense_id,
                     'sense': sense,
                     'long_form': long_form,
                     'line_id': line_id,
                     'inst_id': inst_id
                     }
                )
                # targets.append([pos_id, abbr_id, sense_id, line_id, inst_id])
                # targets.append([id, abbr_id, sense_id, line_id, inst_id, longform_tokens])
                inst_id += 1  # global instance id increment
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

        examples = []
        window_size = int(self.model_config.max_context_len / 2)
        for target in targets:
            pos_id = target['pos_id']
            extend_size = 0
            if pos_id < window_size:
                left_idx = 0
                extend_size = window_size - pos_id
            else:
                left_idx = pos_id - window_size

            if pos_id + window_size > len(contexts):
                right_idx = len(contexts)
            else:
                right_idx = min(
                    pos_id + window_size + extend_size, len(contexts))

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

            example = {
                'contexts': cur_contexts,
                'target': target,
                'line': line,
            }

            examples.append(example)

        return examples, inst_id

    def populate_data(self, path):
        self.datas = []
        line_id = 0
        inst_id = 0
        for line in open(path):
            objs, inst_id = self.process_line(line, line_id, inst_id)
            self.datas.extend(objs)
            line_id += 1
            if line_id % 10000 == 0:
                print('Process %s lines.' % line_id)
        print('Finished processing with inst:%s' % inst_id)


class TrainData(Data):
    def __init__(self, model_config):
        Data.__init__(self, model_config)
        if not model_config.it_train:
            self.populate_data(self.model_config.train_file)
            print('Finished Populate Data with %s samples.' % str(len(self.datas)))
        else:
            self.data_it = self.get_sample_it(self.model_config.train_file)
            if self.model_config.extra_mode:
                self.data_it_cui = self.get_cui_sample_it()
            self.size = self.get_size(self.model_config.train_file)
            print('Finished Data Iter with %s samples.' % str(self.size))

    def get_size(self, data_file):
        return len([l for l in open(data_file, encoding='utf-8').readlines() if len(l.strip()) > 0])

    def get_sample(self):
        i = rd.sample(range(len(self.datas)), 1)[0]
        return self.datas[i]

    def get_cui_sample_it(self):
        """Get feed data from cui extra (e.g. def, stype)"""
        while True:
            i = rd.sample(range(len(self.id2sense)), 1)[0]
            cui = self.id2sense[i]
            if cui in self.cui2def:
                sdef = self.cui2def[cui]
            else:
                sdef = [self.voc.encode(PAD)] * self.model_config.max_def_len

            if cui in self.cui2stype:
                stype = self.cui2stype[cui]
            else:
                stype = self.stype2id['unk']

            for abbr_id in self.cuiud2abbrid[i]:
                obj = {
                    'abbr_id': abbr_id,
                    'cui_id': i,
                    'def': sdef,
                    'stype': stype}
                yield obj

    def prepare_data_for_masked_lm(self, line, examples):
        if not self.model_config.lm_mask_rate:
            return examples
        # print('Use Masked LM with rate %s' % self.model_config.lm_mask_rate)

        masked_cnt = len(examples)
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
            return examples
        for i in range(len(examples)):
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

            examples[i]['cur_masked_contexts'] = cur_masked_contexts
            examples[i]['masked_words'] = masked_word
        return examples

    def get_sample_it(self, data_file, random_dropout=True, reopen_at_EOF=True):
        """
        Load data from file

        reopen_at_EOF: a boolean indicating whether we reopen file after reaching the end of file
         usually True for training and False for testing
        random_dropout: whether to drop out data points by rate 50%, enabled for training
        """
        # line number, each line may contain multiple instances
        data_line_id = 0
        # data instance number
        instance_id = 0
        f = open(data_file, 'r')
        while True:
            if data_line_id >= self.size:
                if reopen_at_EOF:
                    data_line_id = 0
                    f = open(data_file, 'r')
                else:
                    break

            line = f.readline()

            # skip some lines randonly for training to shuffle
            if (random_dropout and rd.random() < 0.5):
                data_line_id += 1
                continue

            examples, instance_id = self.process_line(line, data_line_id, inst_id=instance_id)
            if len(examples) > 0:
                examples = self.prepare_data_for_masked_lm(line, examples)
                for example in examples:
                    yield example

            data_line_id += 1
            # print('line_id=%d' % line_id)


class EvalData(TrainData):
    def __init__(self, train_data, data_config, dataset_name):
        self.model_config = data_config
        self.dataset_name = dataset_name

        # Abbr
        abbr_attributes = ['id2abbr', 'abbr2id', 'id2sense', 'sense2id', 'sen_cnt']
        # Context
        context_attributes = ['voc']
        # CUI
        cui_attributes = ['stype2id', 'id2stype', 'cui2stype', 'cui2def']
        for attribute in  abbr_attributes + context_attributes + cui_attributes:
            setattr(self, attribute, getattr(train_data, attribute, None))

        self.data_it = self.get_sample_it(self.model_config.eval_file, random_dropout=False, reopen_at_EOF=False)
        self.size = self.get_size(self.model_config.eval_file)

        print('Test file path: %s' % self.model_config.eval_file)
        print('Finished Data Iter with %s samples.' % str(self.size))
