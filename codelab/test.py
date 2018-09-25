import json
from collections import Counter


c = Counter()
path = '/Users/sanqiangzhao/git/wsd_data/mimic/final_cleaned_sense_inventory.json'
for line in open(path):
    obj = json.loads(line)
    longforms = obj['LONGFORM']
    longforms = [longform for longform in longforms if len(longform.split()) >= 2]
    c.update(longforms)

c = c.most_common()
for pair in c:
    print(pair)