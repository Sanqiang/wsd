"""
Helper functions for MSH dataset.

"""

import os
import csv
import re
import tqdm
# please "pip install liac-arff" to read arff files
import arff
from preprocess.text_helper import white_space_remover, repeat_non_word_remover, recover_upper_cui
from preprocess.text_helper import CoreNLPTokenizer, TextProcessor, TextTokenFilter
from preprocess.file_helper import txt_reader, txt_writer, json_writer


# Build sense inventory
def sense_inventory_msh(benchmark_mesh_file_path, abbr_list):
    inventory_file = txt_reader(benchmark_mesh_file_path)

    sense_inventory = {}
    sense_inventory_one_word = {}
    for line in inventory_file:
        items = line.split("\t")
        abbr = items[0]
        cuis = items[1:]
        if abbr in abbr_list:
            sense_inventory[abbr] = cuis
            if " " not in abbr:
                sense_inventory_one_word[abbr] = cuis
    return sense_inventory_one_word, sense_inventory


def add_annotation_msh(sense_inventory, arff_folder_path):
    print("Processing annotations...")
    docs_procs = []
    for abbr, cuis in tqdm.tqdm(sense_inventory.items()):
        documents = arff.load(open(arff_folder_path + "/%s_pmids_tagged.arff" % abbr, "r", encoding='latin-1'))
        for doc in documents["data"]:
            text = doc[1]
            sense_id = int(doc[2].lstrip("M")) - 1
            txt_processed = re.sub(r"<e>.+?</e>", " abbr|%s|%s " % (abbr, cuis[sense_id]), text)
            docs_procs.append(txt_processed)
    return docs_procs


def find_abbrs(csv_file):
    abbr_list = []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Type'] == 'A ':
                abbr_list.append(row['Term'])
    return abbr_list


if __name__ == '__main__':

    # File paths
    data_path = "/home/luoz3/data"
    msh_path = data_path + "/msh/MSHCorpus"
    msh_processed_path = data_path + "/msh/msh_processed"

    abbr_list = find_abbrs(msh_path + '/12859_2010_4593_MOESM1_ESM.CSV')

    # Read original sense inventory (only one word abbrs)
    MSH_sense_inventory_one_word, MSH_sense_inventory = sense_inventory_msh(msh_path+"/benchmark_mesh.txt", abbr_list)

    # save sense inventory to json
    json_writer(MSH_sense_inventory_one_word, msh_processed_path + "/MSH_sense_inventory_one_word.json")
    json_writer(MSH_sense_inventory, msh_processed_path + "/MSH_sense_inventory.json")

    #############################
    # Process MSH documents (only one word abbrs)
    #############################
    msh_txt_annotated = add_annotation_msh(MSH_sense_inventory_one_word, msh_path)

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover])
    toknizer = CoreNLPTokenizer()
    token_filter = TextTokenFilter()
    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    # pre-processing
    msh_txt = processor.process_texts(msh_txt_annotated, n_jobs=10)
    # tokenizing
    msh_txt_tokenized = toknizer.process_texts(msh_txt, n_jobs=10)
    # Filter trivial tokens and Remove repeat non-words
    msh_txt_filtered = filter_processor.process_texts(msh_txt_tokenized, n_jobs=10)
    # Write to file
    txt_writer(msh_txt_filtered, msh_processed_path+"/msh_processed.txt")
