"""Get Extra information for each CUI. (e.g. definition, stype)
   Generate pickle which is a map from cui to (definition, semantic type)
"""
import json
import pickle
from collections import defaultdict, Counter


UMLS_DEF = '/Users/sanqiangzhao/git/wsd_data/2018AA/META/MRDEF.RRF'
UMLS_STYPE = '/Users/sanqiangzhao/git/wsd_data/2018AA/META/MRSTY.RRF'

PATH_INVENTORY_JSON = '/Users/sanqiangzhao/git/wsd_data/mimic/final_cleaned_sense_inventory.json'
PATH_EXTRA_CUI = '/Users/sanqiangzhao/git/wsd_data/mimic/cui_extra.pkl'
PATH_EXTRA_CUI_STYPE_VOC = '/Users/sanqiangzhao/git/wsd_data/mimic/cui_extra_stype.voc'

# Get definition
definition_map = defaultdict(str)
for line in open(UMLS_DEF):
    # Doc: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.definitions_file_mrdef_rrf/
    items = line.split('|')
    cui = items[0]
    definition = items[5]
    definition_map[cui] = definition
print('Finish loading definition_map')

# Get semantic type
stype_map = defaultdict(str)
for line in open(UMLS_STYPE):
    # Doc: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/
    items = line.split('|')
    cui = items[0]
    stype = items[3]
    stype_map[cui] = stype
print('Finish loading stype_map')

# Populate extra information for each CUI
output = {}
stype_counter = Counter()
for line in open(PATH_INVENTORY_JSON):
    obj = json.loads(line)
    cui = obj['CUI']
    output[cui] = (definition_map[cui], stype_map[cui])
    stype_counter.update([stype_map[cui]])

f_stype_voc = open(PATH_EXTRA_CUI_STYPE_VOC, 'w')
for w, cnt in stype_counter.most_common():
    f_stype_voc.write('%s\t%s\n' % (w, str(cnt)))

with open(PATH_EXTRA_CUI, 'wb') as output_file:
    pickle.dump(output, output_file)