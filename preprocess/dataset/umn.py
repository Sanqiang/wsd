"""
Helper functions for UMN dataset.

"""

import os
import re
import tqdm
from collections import defaultdict
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer
from preprocess.file_helper import txt_reader, txt_writer, json_writer


def is_umn_senses(sense):
    """
    Whether input sense is created by UMN.
    """
    # Senses created by UMN
    sense_black_list = [
        "NAME",
        "GENERAL ENGLISH",
        "UNSURED SENSE",
        "MISTAKE:",  # error abbr ("MISTAKE:correct expression")
        ":"  # error abbr ("long form:correct acronym or abbreviation")
    ]
    for item in sense_black_list:
        if item in sense:
            return True
    return False


def sub_deid_patterns_umn(txt):
    # DATE
    txt = sub_patterns(txt, [
        # 02/07/2000 or 02/07
        r"_%#MMDD(\d{4})?#%_",
        # 07/02/2000 or 07/02
        r"_%#DDMM(\d{4})?#%_",
        # Feb, 2000 or Feb
        r"_%#MM(\d{4})?#%_",
        # Feb, 07
        r"_%#MM#%_ _%#DD#%_",
        # day
        r"_%#DD(\d{4})?#%_",
        # others
        r"_%#MM200_#%_",
        r"_%#MM2001S#%_",
        r"_%#MDMD\d{4}#%_",
        r"_%#MMDD\d{3}#%_",
        r"_%#MMDD\?\?\?\?#%_",
        r"_%#MMDD\d{4}-\d{4}#%_",
        r"_%#\d{4}#%_",
    ], "DATE-DEID")

    # NAME
    txt = sub_patterns(txt, [
        r"_%#NAME#%_",
    ], "NAME-DEID")

    # zip codes
    txt = sub_patterns(txt, [
        r"_%#\d{5}#%_",
    ], "ZIPCODE-DEID")

    # tel/fax
    txt = sub_patterns(txt, [
        r"_%#FAX#%_",
        r"_%#TEL#%_",
    ], "PHONE-DEID")

    # PLACE
    txt = sub_patterns(txt, [
        # city
        r"_%#CITY#%_",
        # county
        r"_%#COUNTY#%_",
        # town
        r"_%#TOWN#%_",
        # street
        r"_%#STREET#%_",
    ], "PLACE-DEID")

    # STREET-ADDRESS
    txt = sub_patterns(txt, [
        r"_%#ADDRESS#%_",
    ], "STREET-ADDRESS-DEID")

    # Medical record numbers
    txt = sub_patterns(txt, [
        r"_%#MRN#%_",
    ], "MRN-DEID")

    # E-mail
    txt = sub_patterns(txt, [
        r"_%#EMAIL#%_"
    ], "EMAIL-DEID")

    # Device
    txt = sub_patterns(txt, [
        r"_%#DEVICE#%_"
    ], "DEVICE-DEID")

    # other
    txt = sub_patterns(txt, [
        r"_%#(.*)?#%_",
    ], "OTHER-DEID")
    return txt


# Load UMN DataSet
def load_umn(umn_file_path, remove_umn_senses=True):
    dataset_file = []
    umn_txt = []
    umn_file_original = txt_reader(umn_file_path, encoding="latin-1")

    for line in umn_file_original:
        items = line.split("|")
        abbr, sense, start = items[0], items[1], items[3]
        if remove_umn_senses and is_umn_senses(sense):
            continue
        else:
            dataset_file.append((abbr, sense, start))
            umn_txt.append(items[6])
    return dataset_file, umn_txt


def load_cleaned_sense_inventory(inventory_file_path):
    longform2cui = {}
    with open(inventory_file_path, "r") as file:
        for line in file:
            obj = json.loads(line)
            cui = obj["CUI"]
            long_forms = obj["LONGFORM"]
            for long_form in long_forms:
                longform2cui[long_form] = cui
    return longform2cui


# Build sense inventory
def sense_inventory_umn(dataset_file):
    sense_inventory = defaultdict(list)
    for items in dataset_file:
        abbr = items[0]
        sense = items[1]
        if sense not in sense_inventory[abbr]:
            sense_inventory[abbr].append(sense)
    return sense_inventory


def map_longform2cui(lf2cui_dict, long_form_set):
    count = 0
    count_have_cui = 0
    lf2cui = {}
    lf2cui_valid = {}
    for long_form in long_form_set:
        if long_form in lf2cui_dict:
            lf2cui[long_form] = lf2cui_dict[long_form]
            count += 1
            if lf2cui_dict[long_form] is not "":
                lf2cui_valid[long_form] = lf2cui_dict[long_form]
                count_have_cui += 1
    return lf2cui, lf2cui_valid


def add_annotation_umn(sense_inventory, txt_list):
    print("Processing annotations...")

    docs_procs = []
    for items, doc in zip(dataset_file, txt_list):
        abbr = items[0]
        sense = items[1]
        # <todo> zhimeng: use CUI to replace long form
        start = int(items[2])

        doc_processed = "".join([
            doc[:start],
            "abbr|%s|%s" % (abbr, sense),
            doc[start+len(abbr):]
        ])
        docs_procs.append(doc_processed)
    return docs_procs


if __name__ == '__main__':

    # File paths
    data_path = "/home/luoz3/data"
    umn_path = data_path + "/umn"
    umn_processed_path = data_path + "/umn/umn_processed"

    #############################
    # Build sense inventory
    #############################

    dataset_file, umn_txt = load_umn(umn_path+"/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt")
    long_form2cui_our = load_cleaned_sense_inventory(data_path+"/final_cleaned_sense_inventory.json")

    UMN_sense_inventory = sense_inventory_umn(dataset_file)

    # save sense inventory to json
    json_writer(UMN_sense_inventory, umn_processed_path+"/UMN_sense_inventory.json")

    long_form2cui = {}
    with open(umn_path+"/ClinicalSenseInventoryI_MasterFile.txt", "r", encoding='latin-1') as file:
        file.readline()
        for line in file:
            items = line.rstrip('\n').split('|')
            long_form = items[1]
            cui = items[5]
            if cui is "":
                cui = items[2]
            # if ";" not in cui:
            long_form2cui[long_form] = cui

    long_form_set = set()
    for abbr, long_forms in UMN_sense_inventory.items():
        for item in long_forms:
            long_form_set.add(item)

    # #############################
    # # Process MSH documents (only one word abbrs)
    # #############################
    # msh_txt_annotated = add_annotation_msh(MSH_sense_inventory_one_word, msh_path)
    #
    # #######################################
    # # Tokenize
    # #######################################
    #
    # tokenizer = CoreNLPTokenizer()
    # msh_txt_tokenized = tokenizer.process_texts(msh_txt_annotated, n_jobs=10)
    #
    # # Write to file
    # txt_writer(msh_txt_tokenized, msh_processed_path+"/msh_processed.txt")
