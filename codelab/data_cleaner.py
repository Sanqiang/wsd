"""Cleaning the data from Zhimeng Luo."""
from util import constant

def is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    return True


nlines = []
for line in open('/Users/zhaosanqiang916/git/wsd_data/zhimeng/medline_procs_new.txt'):
    line = line.strip()
    if len(line) == 0:
        continue
    nline = []
    words = line.split()
    for word in words:
        if 'SPLIT' in word:
            word = word.replace('SPLIT', '|')
            word = 'abbr|' + word
        elif is_number(word):
            word = constant.NUM
        else:
            word = word.lower()
        nline.append(word)
    nlines.append(' '.join(nline))

f = open('/Users/zhaosanqiang916/git/wsd_data/zhimeng/medline_procs_new.processed.txt', 'w')
f.write('\n'.join(nlines))
f.close()



