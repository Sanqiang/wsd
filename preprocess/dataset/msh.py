"""
Helper functions for MSH dataset.

"""

import os
import re
import tqdm
# please "pip install liac-arff" to read arff files
import arff
from preprocess.text_helper import TextPreProcessor, CoreNLPTokenizer
from preprocess.file_helper import txt_reader, txt_writer, json_writer


# Build sense inventory
def sense_inventory_msh(benchmark_mesh_file_path):
    inventory_file = txt_reader(benchmark_mesh_file_path)

    sense_inventory = {}
    sense_inventory_one_word = {}
    for line in tqdm.tqdm(inventory_file):
        items = line.split("\t")
        abbr = items[0]
        cuis = items[1:]
        sense_inventory[abbr] = cuis
        if " " not in abbr:
            sense_inventory_one_word[abbr] = cuis
    return sense_inventory_one_word, sense_inventory


def add_annotation_msh(sense_inventory, arff_folder_path, splitter='\u2223'):
    print("Processing annotations...")
    docs_procs = []
    for abbr, cuis in tqdm.tqdm(sense_inventory.items()):
        documents = arff.load(open(arff_folder_path + "/%s_pmids_tagged.arff" % abbr, "r", encoding='latin-1'))
        for doc in documents["data"]:
            text = doc[1]
            sense_id = int(doc[2].lstrip("M")) - 1
            txt_processed = re.sub(r"<e>.+?</e>", "%s%s%s" % (abbr, splitter, cuis[sense_id]), text)
            # txt_processed = text.replace("<e>%s</e>" % abbr, "%s%s%s" % (abbr, splitter, cuis[sense_id]))
            docs_procs.append(txt_processed)
    return docs_procs


if __name__ == '__main__':

    # File paths
    data_path = "/home/luoz3/data"
    msh_path = data_path + "/msh/MSHCorpus"
    msh_processed_path = data_path + "/msh/msh_processed"

    # Read original sense inventory (only one word abbrs)
    MSH_sense_inventory_one_word, MSH_sense_inventory = sense_inventory_msh(msh_path+"/benchmark_mesh.txt")

    # save sense inventory to json
    json_writer(MSH_sense_inventory_one_word, msh_processed_path + "/MSH_sense_inventory_one_word.json")
    json_writer(MSH_sense_inventory, msh_processed_path + "/MSH_sense_inventory.json")

    #############################
    # Process MSH documents (only one word abbrs)
    #############################
    msh_txt_annotated = add_annotation_msh(MSH_sense_inventory_one_word, msh_path)

    #######################################
    # Tokenize
    #######################################

    tokenizer = CoreNLPTokenizer()
    msh_txt_tokenized = tokenizer.process_texts(msh_txt_annotated, n_jobs=10)

    # Write to file
    txt_writer(msh_txt_tokenized, msh_processed_path+"/msh_processed.txt")
