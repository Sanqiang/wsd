from model.model_config import BaseConfig
from data_generator.vocab import Vocab
import pickle
import re
from collections import Counter

c = Counter()
max_len = -1
voc = Vocab(BaseConfig(),
            '/Users/sanqiangzhao/git/wsd_data/mimic/subvocab')
with open('/Users/sanqiangzhao/git/wsd_data/mimic/cui_extra.pkl', 'rb') as cui_file:
    cui_extra = pickle.load(cui_file)
    for cui in cui_extra:
        info = cui_extra[cui]
        text = info[0]
        text = re.compile(r'<[^>]+>').sub('', text)
        l = len(voc.encode(text))
        max_len = max(max_len, l)
        c.update([l])

print(max_len)
print(c.most_common())