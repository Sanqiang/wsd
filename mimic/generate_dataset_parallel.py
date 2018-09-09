# -*- coding: utf-8 -*-
"""
Find longform appearances with a simple reverse index, proved very slow as there are too many longforms
"""

import os
import pickle
import pandas as pd
import re
import json
from joblib import Parallel, delayed

from mimic.abbr_sense import load_medline_abbr_dict, load_final_sense_inventory

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def tokenize(text):
    tokens = re.split(r'\W+|_+', text.lower())
    return tokens

def find_match_longforms(long_sense_dict, chunk_df):

    longform_token_sets = [set(tokenize(lf)) for lf in long_sense_dict.keys()]

    for row_id, row_series in chunk_df.iterrows():
        row_dict = row_series.to_dict()

        text = row_dict['TEXT']
        text_tokens_set = set(tokenize(text))

        for longform, sense, longform_token_set in zip(long_sense_dict.items(), longform_token_sets):
            if len(longform_token_set & text_tokens_set) > 0:
                pass

if __name__ == '__main__':
    # load text, put them into a dict as well
    mimic3_csv_path = '/Users/memray/Data/upmc/mimic3/NOTEEVENTS.csv'
    chunk_size = 10000
    n_jobs = 1

    sense_inventory_dict, long_sense_dict = load_final_sense_inventory()
    mimic_csv_df = pd.read_csv(mimic3_csv_path, chunksize=chunk_size)
    text_dict = {}

    def producer(mimic_csv_df):
        for chunk_id, chunk_df in enumerate(mimic_csv_df):
            print('Produced %d/2,083,180 notes' % chunk_id * chunk_size)
            yield chunk_df

    example_count = 0

    with Parallel(n_jobs=n_jobs, verbose=100, pre_dispatch='2*n_jobs') as parallel:
        accumulator = 0.
    n_iter = 0

    results = parallel(delayed(find_match_longforms)(long_sense_dict, chunk_df) for chunk_df in producer(mimic_csv_df))
    accumulator += sum(results)  # synchronization barrier
    n_iter += 1

