"""Get Extra information for each CUI. (e.g. definition, stype)
   Generate pickle which is a map from cui to (definition, semantic type)
"""
import json
import pickle
from collections import defaultdict, Counter


UMLS_DEF = '/Users/sanqiangzhao/git/wsd/wsd_data/2018AA/META/MRDEF.RRF'
UMLS_STYPE = '/Users/sanqiangzhao/git/wsd/wsd_data/2018AA/META/MRSTY.RRF'
UMLS_ATOM = '/Users/sanqiangzhao/git/wsd/wsd_data/2018AA/META/MRCONSO.RRF'

PATH_EXTRA_DEF = '/exp_data/wsd_data/umls/def.txt'
PATH_EXTRA_TYPE = '/exp_data/wsd_data/umls/type.txt'
PATH_EXTRA_ATOM = '/exp_data/wsd_data/umls/atom.txt'
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

# Get Atom
atom_map = defaultdict(str)
for line in open(UMLS_ATOM):
    # Doc: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
    items = line.split('|')
    cui = items[0]
    atom = items[14]
    atom_map[cui] += atom + ';'
print('Finish loading atom')


