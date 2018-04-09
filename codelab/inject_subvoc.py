from model.model_config import get_path

abbrs = set()

for line in open(get_path('../wsd_data/medline/abbr_common.txt')):
    items = line.strip().split('\t')
    w = items[0]
    ws = w.split('|')
    abbr = ws[0]
    abbrs.add(abbr)

for line in open(get_path('../wsd_data/medline/abbr_rare.txt')):
    items = line.strip().split('\t')
    w = items[0]
    ws = w.split('|')
    abbr = ws[0]
    abbrs.add(abbr)

f = open(get_path('../wsd_data/medline/abbr_all.txt'), 'w')
for abbr in abbrs:
    f.write('\'' + abbr + '_\'')
    f.write('\n')
f.close()