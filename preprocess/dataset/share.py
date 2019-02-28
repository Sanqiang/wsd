"""
Helper functions for ShARe/CLEF dataset.

"""

import os
import re
import tqdm
from collections import defaultdict
from preprocess.text_helper import white_space_remover, repeat_non_word_remover, recover_upper_cui, is_valid_abbr
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from preprocess.file_helper import txt_writer, json_writer, json_reader
from preprocess.dataset.mimic_preprocess import sub_deid_patterns_mimic


toknizer = CoreNLPTokenizer()


def add_annotation_share(folder_path):
    """
    Add annotation and build abbr sense inventory.
    To replace original abbr "AB" to "abbr|AB|C0123456 "

    """
    print("Processing annotations...")
    # read original data
    abbr_dict = defaultdict(list)
    abbr_invalid_dict = defaultdict(list)
    file_list = sorted(os.listdir(folder_path))

    docs_processed = []
    # process files
    for annot_file in tqdm.tqdm(file_list):
        with open(os.path.join(folder_path, annot_file), 'r') as file:
            instance_list = []
            for row in file:
                instance_list.append(row.rstrip("\n").split("||"))
        # check empty file
        if instance_list[0] == [""]:
            continue

        # read doc file
        with open(os.path.join(share_reports_path, instance_list[0][0]), "r") as file:
            doc = "".join(file.readlines())

        # collect abbr info and add annotations to doc
        temp_doc = ""
        last_abbr_end = 0
        for instance in instance_list:
            label, start, end = instance[2:5]
            start = int(start)
            end = int(end)
            abbr = doc[start:end]

            # skip instances without CUI
            if label == "CUI-less":
                continue
            # skip multi-word abbrs
            elif " " in abbr:
                if label not in abbr_invalid_dict[abbr]:
                    abbr_invalid_dict[abbr].append(label)
                continue
            # skip invalid abbrs
            elif not is_valid_abbr(abbr):
                if label not in abbr_invalid_dict[abbr]:
                    abbr_invalid_dict[abbr].append(label)
                continue
            # skip abbrs which will be splitted by tokenizer
            elif " " in toknizer.process_single_text(abbr):
                if label not in abbr_invalid_dict[abbr]:
                    abbr_invalid_dict[abbr].append(label)
                continue
            # correct start position
            elif abbr.startswith('\n'):
                start += 1
                abbr = doc[start:end]

            # add unseen CUIs to abbr
            if label not in abbr_dict[abbr]:
                abbr_dict[abbr].append(label)

            temp_doc = "".join([
                temp_doc,
                doc[last_abbr_end:start],
                " abbr|%s|%s " % (abbr, label),
            ])
            last_abbr_end = end
        temp_doc += doc[last_abbr_end:]
        docs_processed.append(temp_doc)
    return docs_processed, abbr_dict, abbr_invalid_dict


def merge_inventories(inventory1, inventory2):
    inventory = inventory1.copy()
    for abbr, cuis in inventory2.items():
        for cui in cuis:
            if cui not in inventory[abbr]:
                inventory[abbr].append(cui)
    return inventory


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    data_path = "/home/luoz3/wsd_data"
    share_path = data_path + "/share/original"
    share_processed_path = data_path + "/share/processed"
    os.makedirs(share_processed_path, exist_ok=True)

    share_train_annot_path = share_path + "/Task2TrainSetSILVER2pipe"
    share_test_annot_path = share_path + "/Task2ReferenceStd_CLEFShARe2013Test_StrictAndLenientpipe"
    share_reports_path = share_path + "/ALLREPORTS"

    #############################
    # Process ShARe/CLEF documents (only one word abbrs)
    #############################

    share_txt_train_annotated, train_sense_inventory, train_sense_inventory_invalid = add_annotation_share(share_train_annot_path)
    share_txt_test_annotated, test_sense_inventory, test_sense_inventory_invalid = add_annotation_share(share_test_annot_path)

    # combine corpus
    share_txt_all_annotated = share_txt_train_annotated.copy()
    share_txt_all_annotated.extend(share_txt_test_annotated)
    all_sense_inventory = merge_inventories(train_sense_inventory, test_sense_inventory)
    all_sense_inventory_invalid = merge_inventories(train_sense_inventory_invalid, test_sense_inventory_invalid)

    # save sense inventory to json
    json_writer(train_sense_inventory, share_processed_path + "/train_sense_inventory.json")
    json_writer(test_sense_inventory, share_processed_path + "/test_sense_inventory.json")
    json_writer(all_sense_inventory, share_processed_path + "/all_sense_inventory.json")
    json_writer(all_sense_inventory_invalid, share_processed_path + "/all_sense_inventory_invalid.json")

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover,
        sub_deid_patterns_mimic])

    toknizer = CoreNLPTokenizer()

    token_filter = TextTokenFilter()
    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    # pre-processing
    share_txt = processor.process_texts(share_txt_all_annotated, n_jobs=30)
    # tokenizing
    share_txt_tokenized = toknizer.process_texts(share_txt, n_jobs=30)
    # Filter trivial tokens and Remove repeat non-words
    share_txt_filtered = filter_processor.process_texts(share_txt_tokenized, n_jobs=30)
    # Write to file
    txt_writer(share_txt_filtered, share_processed_path+"/share_all_processed.txt")
