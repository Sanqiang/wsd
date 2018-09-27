"""
Example for pre-processing functions.

"""

import os
import tqdm
import json
from preprocess.dataset_helpers.mimic import sub_deid_patterns
from preprocess.text_helper import TextPreProcessor, CoreNLPTokenizer, white_space_remover, sub_patterns


######################################
# Read texts from dataset
######################################

PATH_FOLDER = '/home/zhaos5/projs/wsd/wsd_data/mimic/find_longform_mimic/'
mimic_txt = []
for line in open(PATH_FOLDER + "processed_text_chunk_1.json"):
    obj = json.loads(line)
    text = obj['TEXT']
    mimic_txt.append(text)

print(len(mimic_txt))


######################################
# Pre-processing
######################################

processor = TextPreProcessor([
    white_space_remover,
    sub_deid_patterns])

# for list of texts
mimic_txt = processor.process_texts(mimic_txt, 30)


######################################
# Tokenizing
######################################

toknizer = CoreNLPTokenizer()

# for single text
temp = toknizer.process_single_text(mimic_txt[0])
print(temp)

# for list of texts
mimic_txt_tokenized = toknizer.process_texts(mimic_txt, 4)
print(len(mimic_txt_tokenized))
print(mimic_txt_tokenized[352])
