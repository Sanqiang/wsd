# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

from mimic import build_sense_inventory
from mimic.build_sense_inventory import Sense

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    sense_inventory, long_sense_dict = build_sense_inventory.load_final_sense_inventory()
    data_root_path = '/Users/memray/Project/upmc_wsd/wsd_data/mimic/'
    output_folder_path = os.path.join(data_root_path, 'find_longform_mimic/')
    sense_present_distribution_path = os.path.join(output_folder_path, 'sense_present_record.json')

    dataset_sense_present_dict = None
    with open(sense_present_distribution_path, 'r') as output_json:
        dataset_sense_present_dict = json.load(output_json)

    total_sense_presence_count = sum([len(present_record) for present_record in dataset_sense_present_dict.values()])
    print('#(sense) = %d' % len(dataset_sense_present_dict))
    print('#(total presence) = %d' % total_sense_presence_count)
    print('#(avg presence) = %.4f' % (float(total_sense_presence_count)/len(dataset_sense_present_dict)))

    longform_count_dict = {}
    longform_length_at_K_count_dict = {}
    longform_length_count_dict = {}
    for cui, present_record in dataset_sense_present_dict.items():
        for longform, doc_id in present_record:
            longform_count = longform_count_dict.get(longform, 0)
            longform_count_dict[longform] = longform_count + 1

            longform_length = len(longform.split())
            longform_length_count = longform_length_count_dict.get(longform_length, 0)
            longform_length_count_dict[longform_length] = longform_length_count + 1

            longform_length_at_K_count = longform_length_at_K_count_dict.get(longform_length, {})
            longform_length_at_K_count[longform] = longform_count + 1
            longform_length_at_K_count_dict[longform_length] = longform_length_at_K_count

    longform_count_sorted = sorted(longform_count_dict.items(), key=lambda x:x[1])
    longform_length_count_sorted = sorted(longform_length_count_dict.items(), key=lambda x:x[1])
    longform_length_at_K_counts_sorted = dict([(k, sorted(d.items(), key=lambda x:x[1])) for k, d in longform_length_at_K_count_dict.items()])

    output_folder_path = os.path.join(data_root_path, 'find_longform_mimic/analysis/')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    print('=' * 50)
    print('Length distribution')
    with open(output_folder_path + 'length_distribution.txt', 'w') as file:
        for longform_length, count in longform_length_count_sorted:
            print('len=%d, number=%d' % (longform_length, count))
            file.write('%d,%d\n' % (longform_length, count))

    print('=' * 50)
    print('Most frequent longforms:')
    with open(output_folder_path + 'longforms_sorted_all.txt', 'w') as file:
        for longform, count in longform_count_sorted[::-1]:
            print('%s : number=%d' % (longform, count))
            file.write('%s,%d\n' % (longform, count))

    print('=' * 50)

    for length, longform_length_at_K_count_sorted in longform_length_at_K_counts_sorted.items():
        with open(output_folder_path + 'longforms_sorted_len_%d.txt' % length, 'w') as file:
            print('Top frequent longforms at length=%d:' % (length))
            for longform, count in longform_length_at_K_count_sorted[::-1]:
                print('%s : number=%d' % (longform, count))
                file.write('%s,%d\n' % (longform, count))