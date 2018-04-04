f_abbr = open('/Users/zhaosanqiang916/git/wsd_data/medline/abbr.txt')

mapper = {}

for line in f_abbr:
    items = line.strip().split('|')
    abbr = items[1]
    sense = items[2]

    if abbr not in mapper:
        mapper[abbr] = set()
    if sense not in mapper[abbr]:
        mapper[abbr].add(sense.split()[0])

f_abbr_rare = open('/Users/zhaosanqiang916/git/wsd_data/medline/abbr_rare.txt', 'w')
f_abbr_common = open('/Users/zhaosanqiang916/git/wsd_data/medline/abbr_common.txt', 'w')

lines_rare, lines_common = [], []
for abbr in mapper:
    senses = mapper[abbr]
    if len(senses) <= 1:
        lines_rare.append(abbr)
    else:
        lines_common.append(abbr + '|' + ' '.join(mapper[abbr]))

f_abbr_rare.write('\n'.join(lines_rare))
f_abbr_rare.close()

f_abbr_common.write('\n'.join(lines_common))
f_abbr_common.close()