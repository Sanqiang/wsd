"""
Example processing pipeline for general DataSet:
-- First customize DataSet functions below
-- Change the order of annotator or other processors in main function if necessary
-- Make sure CoreNLP Java Server is open, by "screen -r nlp" in terminal
-- Process your DataSet
"""
import os
import re
import tqdm
import operator
import multiprocessing as mp
from preprocess.text_helper import sub_patterns, white_space_remover, repeat_non_word_remover, recover_upper_cui
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from preprocess.file_helper import txt_reader, txt_writer, json_writer


def sub_deid_patterns_dataset(txt):
    """
    Replace DeID strings to uniform format ("*-DEID").

    :param txt:
    :return:
    """
    # DATE
    txt = sub_patterns(txt, [
        r"",
        r"",
    ], "DATE-DEID")
    # ...
    return txt


# Build sense inventory
def sense_inventory_dataset(file_path):
    """
    Write a function to build abbr sense inventory

    :param file_path:
    :return: sense_inventory
    """
    print("Building sense inventory...")
    sense_inventory = {}
    return sense_inventory


def add_annotation_dataset(sense_inventory, file_path):
    """
    Annotation adder function
    To replace original abbr "AB" to "abbr|AB|C0123456 "

    :param sense_inventory:
    :param file_path:
    :return:
    """
    print("Processing annotations...")
    docs_procs = []
    return docs_procs


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    data_path = "/home/luoz3/wsd_data"
    dataset_path = data_path + "/path/to/dataset"
    dataset_processed_path = data_path + "/path/to/dataset/processed"
    os.makedirs(dataset_processed_path, exist_ok=True)

    # Read or build sense inventory (only one word abbrs)
    sense_inventory = sense_inventory_dataset(data_path+"/inventory.file")

    # save sense inventory to json
    json_writer(sense_inventory, dataset_processed_path + "/dataset_sense_inventory.json")

    #############################
    # Process DataSet documents (only one word abbrs)
    #############################

    dataset_txt_annotated = add_annotation_dataset(sense_inventory, dataset_path)

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover,
        sub_deid_patterns_dataset])

    toknizer = CoreNLPTokenizer()

    token_filter = TextTokenFilter()
    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    # pre-processing
    dataset_txt = processor.process_texts(dataset_txt_annotated, n_jobs=30)
    # tokenizing
    dataset_txt_tokenized = toknizer.process_texts(dataset_txt, n_jobs=30)
    # Filter trivial tokens and Remove repeat non-words
    dataset_txt_filtered = filter_processor.process_texts(dataset_txt_tokenized, n_jobs=30)
    # Write to file
    txt_writer(dataset_txt_filtered, dataset_processed_path+"/dataset_processed.txt")
