# -*- coding: utf-8 -*-
"""
Find longform appearances with a simple reverse index, proved very slow as there are too many longforms
"""

import os
import pickle
import time

import pandas as pd
import re
import json
from joblib import Parallel, delayed

from mimic.abbr_sense import load_final_sense_inventory, Sense
from mimic.string_utils.suffix_tree import SuffixTree

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def tokenize(text):
    tokens = re.split(r'\W+|_+', text.lower())
    return tokens


def find_match_longforms(long_sense_dict, chunk_id, chunk_dataframe, output_json_path):
    """
    Search through each note in the dataframe to find the present senses,
     store them in disk in JSON file and return the present information of each sense
    :param long_sense_dict:
    :param chunk_dataframe:
    :return:
    dataset_sense_present_dict: A dict to store the presences of each sense in clinical notes,
                                key is the CUI of a sense, value is a list,
                                each element is a tuple (present_longform, note_id)
    """
    print("*" * 50)
    print("Processing chunk %d..." % chunk_id)
    print("*" * 50)

    dataset_sense_present_dict = {}
    # use a set to quick check if a long form appears in a note
    longform_token_sets = [set(tokenize(lf)) for lf in long_sense_dict.keys()]

    total_doc_num = len(chunk_dataframe)

    with open(output_json_path, 'w') as output_json:
        for doc_id, doc_series in chunk_dataframe.iterrows():
            if doc_id % 1000 == 0:
                print("Processing chunk %d, doc %d/%d..." % (chunk_id, doc_id, total_doc_num))

            note_dict = doc_series.to_dict()

            note_present_longform_dict = {}
            note_id = note_dict['ROW_ID']
            text = note_dict['TEXT']
            text_tokens_set = set(tokenize(text))

            text_st = SuffixTree(text, case_insensitive=False)

            for sense_id, ((longform, sense), longform_token_set) \
                    in enumerate(zip(long_sense_dict.items(), longform_token_sets)):
                if len(longform_token_set & text_tokens_set) > 0:
                    if text_st.has_substring(longform):
                        note_present_longform_dict[longform] = sense.cui
                        sense_present_records = dataset_sense_present_dict.get(sense.cui, [])
                        sense_present_records.append((longform, note_id))
                        dataset_sense_present_dict[sense.cui] = sense_present_records
                    # else:
                    #     print('Match by SuffixTree failed, chunk %d, doc %d, sense %d' % (chunk_id, doc_id, sense_id))

                # else:
                #     print('Match by Set failed, chunk %d, doc %d, sense %d' % (chunk_id, doc_id, sense_id))

            note_dict['present_senses'] = note_present_longform_dict

            output_json.write(json.dumps(note_dict) + '\n')

    print("*" * 50)
    print("Process chunk %d done!" % chunk_id)
    print("*" * 50)

    return dataset_sense_present_dict


if __name__ == '__main__':
    # load text, put them into a dict as well
    mimic3_csv_path = '/home/luoz3/data/mimic/mimic3/NOTEEVENTS.csv'
    # mimic3_csv_path = '/Users/memray/Data/upmc/mimic3/NOTEEVENTS.csv'
    data_root_path = '/home/mengr/Project/wsd/wsd_data/mimic/'
    sense_inventory_path = os.path.join(data_root_path, 'final_cleaned_sense_inventory.pkl')
    output_folder_path = os.path.join(data_root_path, 'find_longform_mimic/')
    # output_folder_path = '/Users/memray/Project/upmc_wsd/wsd_data/mimic/find_longform_mimic/'

    sense_inventory_dict, long_sense_dict = load_final_sense_inventory(sense_inventory_path)

    chunk_size = 50000
    n_jobs = 16
    mimic_csv_df = pd.read_csv(mimic3_csv_path, chunksize=chunk_size)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    text_dict = {}

    def producer(mimic_csv_df):
        for chunk_id, chunk_df in enumerate(mimic_csv_df):
            print('Produced %d/2,083,180 notes\n' % ((chunk_id + 1) * chunk_size))
            yield chunk_id, chunk_df

    example_count = 0

    start_time = time.time()
    with Parallel(n_jobs=n_jobs, verbose=100, pre_dispatch='2*n_jobs') as parallel:
        accumulator = 0.
    n_iter = 0

    processed_text_by_chunk_json_path = os.path.join(output_folder_path, 'processed_text_chunk_%d.json')
    dataset_sense_present_dicts = parallel(delayed(find_match_longforms)(long_sense_dict, chunk_id, chunk_df, processed_text_by_chunk_json_path % chunk_id) for chunk_id, chunk_df in producer(mimic_csv_df))

    print(len(dataset_sense_present_dicts))

    end_time = time.time()
    print(end_time - start_time)

    # merge sense-present dicts
    merged_dict = {}

    for dataset_sense_present_dict in dataset_sense_present_dicts:
        for cui, sense_present_records in dataset_sense_present_dict.items():
            current_cui_records = merged_dict.get(cui, [])
            current_cui_records.extend(sense_present_records)
            merged_dict[cui] = current_cui_records

    total_sense_presence_count = sum([len(present_record) for present_record in merged_dict.values()])
    print('#(sense) = %d' % len(merged_dict))
    print('#(total presence) = %d' % total_sense_presence_count)
    print('#(avg presence) = %.4f' % (float(total_sense_presence_count)/len(merged_dict)))

    sense_present_distribution_path = os.path.join(output_folder_path, 'sense_present_record.json')
    with open(sense_present_distribution_path, 'w') as output_json:
        output_json.write(json.dumps(merged_dict))
