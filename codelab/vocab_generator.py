from collections import Counter


c = Counter()
for line in open('/Users/zhaosanqiang916/git/wsd_data/medline/train.txt'):
    words = line.split()
    c.update(words)

f_voc = open('/Users/zhaosanqiang916/git/wsd_data/medline/voc.txt', 'w')
f_abbr = open('/Users/zhaosanqiang916/git/wsd_data/medline/abbr.txt', 'w')
c = c.most_common()
for w, cnt in c:
    if w.startswith('abbr|'):
        f_abbr.write(w)
        f_abbr.write('\t')
        f_abbr.write(str(cnt))
        f_abbr.write('\n')
    else:
        f_voc.write(w)
        f_voc.write('\t')
        f_voc.write(str(cnt))
        f_voc.write('\n')
f_voc.close()
f_abbr.close()