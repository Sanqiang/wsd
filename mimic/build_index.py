# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import pandas as pd
import re
import pickle

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def update_index(reverse_index, row_id, text):
    try:
        tokens = re.split(r'\W+|_+', text.lower())
        tokens = list(filter(lambda x: len(x) > 1 and not re.match('\d+', x), tokens))
        unique_tokens = set(tokens)

        for unique_token in unique_tokens:
            posting_list = reverse_index.get(unique_token, [])
            posting_list.append(row_id)
            reverse_index[unique_token] = posting_list
    except Exception:
        pass

def main():
    mimic3_csv_path = '/Users/memray/Data/upmc/mimic3/NOTEEVENTS.csv'
    mimic_df = pd.read_csv(mimic3_csv_path, chunksize=1)
    reverse_index = {}

    for example_id, example in enumerate(mimic_df):
        if example_id % 1000 == 0:
            print('Processed %d data examples' % example_id)
            print('\t#(vocab)=%d' % len(reverse_index))
        row_id = example.iloc[0]['ROW_ID']
        text = example.iloc[0]['TEXT']
        update_index(reverse_index, row_id, text)

    print('In total processed %d data examples' % example_id)
    print('\t#(final vocab)=%d' % len(reverse_index))

    index_dump_path = '../../wsd_data/mimic/mimic_index.pkl'
    print('Dumping to %s' % os.path.abspath(index_dump_path))
    with open(index_dump_path, 'wb') as index_dump:
        pickle.dump(reverse_index, index_dump)


if __name__ == '__main__':
    main()