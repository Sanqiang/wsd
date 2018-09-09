# -*- coding: utf-8 -*-
"""
Find longform appearances with a simple reverse index, proved very slow as there are too many longforms
"""

import os
import pickle
import pandas as pd
import re
import json

from mimic.abbr_sense import load_medline_abbr_dict

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def get_possible_row_idx(long_form, reverse_index):
    tokens = re.split(r'\W+|_+', long_form.lower())
    tokens = list(filter(lambda x: len(x) > 1 and not re.match('\d+', x), tokens))

    row_idx = []
    for token in tokens:
        if token in reverse_index:
            row_idx.extend(reverse_index[token])
        else:
            return []

    return list(set(row_idx))


def get_matching_row_idx(long_form, text_dict, row_idx):
    filtered_row_idx = []
    long_form = long_form.lower()

    for row_id in row_idx:
        if row_id not in text_dict:
            continue
        text = text_dict[row_id].lower()
        if text.find(long_form) > 0:
            filtered_row_idx.append(int(row_id))

    return filtered_row_idx


if __name__ == '__main__':
    # load text, put them into a dict as well
    mimic3_csv_path = '/Users/memray/Data/upmc/mimic3/NOTEEVENTS.csv'
    mimic_df = pd.read_csv(mimic3_csv_path, chunksize=10000)
    text_dict = {}

    example_count = 0

    for chunk in mimic_df:
        for example_id, example in chunk.iterrows():
            if example_count % 10000 == 0:
                print('Loaded %d data examples' % example_id)
            row_id = example['ROW_ID']
            text = example['TEXT']
            example_count += 1
            if isinstance(text, str):
                text_dict[row_id] = text

    print('Load %d notes' % len(text_dict))

    '''
    # load the index, a dict whose key is a word (string) and value is row_id indicating the text contains the key
    index_file_path = '../../wsd_data/mimic/mimic_index.pkl'
    print('Loading from %s' % os.path.abspath(index_file_path))
    with open(index_file_path, 'rb') as index_file:
        reverse_index = pickle.load(index_file)
        print('Load index complete')

    abbr_long_dict, long_abbr_dict, long_list = load_medline_abbr_dict()
    example_count = 0

    # Process each long form and get its relevant text
    data_json_path = '../../wsd_data/mimic/matching_text.json'
    with open(data_json_path, 'w') as data_json_file:
        for long_form_count, (long_form, abbr_list) in enumerate(long_abbr_dict.items()):
            if long_form_count % 1000 == 0:
                print('Progress: processed long-forms: %d/%d' % (long_form_count, len(long_abbr_dict)))

            row_idx = get_possible_row_idx(long_form, reverse_index)
            row_idx = get_matching_row_idx(long_form, text_dict, row_idx)

            ex = {'row_indices': row_id, 'short_forms': abbr_list, 'long_form': long_form}
            # print(row_id)
            # print(abbr_list)
            # print(long_form)
            # print(text_dict[row_id])
            # print(json.dumps(ex) + '\n')
            data_json_file.write(json.dumps(ex) + '\n')
            example_count += 1

    print('Export json to file done, %d examples' % example_count)

    '''