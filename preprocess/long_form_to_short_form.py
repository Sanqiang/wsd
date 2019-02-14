import tqdm
import json
from collections import defaultdict

umls_path = '/exp_data/wsd_data/umls/2018AB'
umls_file = umls_path + '/META/MRCONSO.RRF'
lexicon_abbr_file = umls_path + '/LEX/LRABR'

# lexicon_abbr_file = r'C:\data\upmc\umls\LEX\LRABR'
# umls_file = r"C:\data\upmc\umls\META\MRCONSO.RRF"

# load lexicon abbr records and mapping short form to long form and vice versa
lexicon_abbr_record = []
with open(lexicon_abbr_file, 'r', encoding='utf-8') as file:
    for line in file:
        lexicon_abbr_record.append(line.rstrip('\n').split('|'))

lexicon_sf2lf = defaultdict(set)
lexicon_lf2sf = defaultdict(set)
for record in lexicon_abbr_record:
    lexicon_lf2sf[record[4]].add(record[1])
    lexicon_sf2lf[record[1]].add(record[4])


# load umls records
umls_record = []
with open(umls_file, 'r', encoding='utf-8') as file:
    for line in tqdm.tqdm(file):
        umls_record.append(line.rstrip('\n'))

umls_lf2cui = defaultdict(set)
for record in tqdm.tqdm(umls_record):
    record = record.split('|')
    if record[1] == 'ENG':
        umls_lf2cui[record[14]].add(record[0])

cui_pairs = defaultdict(list)
for long_form, short_form_set in lexicon_lf2sf.items():
    if long_form in umls_lf2cui:
        for cui in umls_lf2cui[long_form]:
            for short_form in short_form_set:
                cui_pairs[cui].append([short_form, long_form])

# save to file
with open('cui_pairs.json', 'w') as file:
    for cui, pairs in cui_pairs.items():
        temp_dict = {"CUI": cui}
        temp_dict["LONGFORM_ABBR_MAPPINGS"] = pairs
        file.write(json.dumps(temp_dict)+'\n')

print()
