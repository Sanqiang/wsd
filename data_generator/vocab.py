from util import constant
from util.data.text_encoder import SubwordTextEncoder


class Vocab:
    def __init__(self, model_config, vocab_path=None):
        self.model_config = model_config
        self.vocab_path = vocab_path
        if self.model_config.subword_vocab_size <= 0:
            self.init_vocab()
            if vocab_path is not None:
                self.populate_vocab()
        else:
            if vocab_path is not None:
                self.populate_subword_vocab()

    def populate_subword_vocab(self):
        self.subword = SubwordTextEncoder(self.vocab_path)
        print('Subword Vocab Populated with size %d for path %s.'
              % (len(self.subword._all_subtoken_strings), self.vocab_path))

    def init_vocab(self):
        self.w2i = {}
        self.i2w = []
        # self.w2i[constant.PAD] = 0
        # self.i2w.append(constant.PAD)
        # self.w2i[constant.UNK] = 1
        # self.i2w.append(constant.UNK)
        # self.w2i[constant.BOS] = 2
        # self.i2w.append(constant.BOS)
        # self.w2i[constant.BOS] = 3
        # self.i2w.append(constant.EOS)
        # self.w2i[constant.EOS] = 4
        # self.i2w.append(constant.NONTAR)

    def populate_vocab(self, mincount=-1):
        # mincount = max(mincount, self.model_config.min_count)

        for line in open(self.vocab_path, encoding='utf-8'):
            items = line.strip().split('\t')
            w = items[0]
            if len(items) > 1:
                cnt = int(items[1])
            else:
                # Accept all words
                cnt = 99999
            if cnt >= mincount:
                self.w2i[w] = len(self.i2w)
                self.i2w.append(w)
        self.i2w.append(constant.PAD)
        self.w2i[constant.PAD] = len(self.w2i)
        print('Vocab Populated with size %d including %d reserved vocab for path %s.'
              % (len(self.i2w), constant.REVERED_VOCAB_SIZE, self.vocab_path))

    def encode(self, w):
        if self.model_config.subword_vocab_size <= 0:
            if w in self.w2i:
                return self.w2i[w]
            else:
                if constant.UNK in self.w2i:
                    return self.w2i[constant.UNK]
                else:
                    return self.w2i[constant.PAD]
        else:
            return self.subword.encode(w)

    def contain(self, w):
        return w in self.w2i

    def describe(self, i):
        if self.model_config.subword_vocab_size <= 0:
            if i < len(self.i2w):
                return self.i2w[i]
        else:
            # Note in subword case, i should be list of id, i.e. ids.
            return self.subword.decode(i)

    def vocab_size(self):
        if self.model_config.subword_vocab_size <= 0:
            return len(self.i2w)
        else:
            return len(self.subword._all_subtoken_strings)

# if __name__ == '__main__':
#     subword = SubwordTextEncoder('/Users/zhaosanqiang916/git/wsd_data/medline/subvoc.txt')
#     res = subword.encode('cd106')
#     for re in res:
#         print(subword.decode([re]))