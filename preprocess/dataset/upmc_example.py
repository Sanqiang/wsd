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


def replace(token: str):
    """
    Fix annotation error
    :param token:
    :return:
    """
    if token.startswith("abbr|"):
        segments = token.split("|")
        middle = int((len(segments) - 1) / 2)
        return segments[0] + "|" + segments[middle] + "|" + segments[len(segments) - 1]
    else:
        return token


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    data_path = "/home/luoz3/wsd_data"
    dataset_path = data_path + "/upmc/example"
    dataset_processed_path = data_path + "/upmc/example/processed"
    os.makedirs(dataset_processed_path, exist_ok=True)

    # # fix annotation error
    # with open(dataset_path + "/training_data.txt") as input, open(dataset_path + "/training_data_fixed.txt", "w") as output:
    #     for line in input:
    #         new_line = " ".join([replace(token) for token in line.rstrip("\n").split(" ")])
    #         output.write(new_line + "\n")

    #############################
    # Process DataSet documents (only one word abbrs)
    #############################

    dataset_txt_annotated = txt_reader(dataset_path + "/training_data_fixed.txt")

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
    dataset_txt = processor.process_texts(dataset_txt_annotated, n_jobs=30)
    # tokenizing
    dataset_txt_tokenized = toknizer.process_texts(dataset_txt, n_jobs=30)
    # Filter trivial tokens and Remove repeat non-words
    dataset_txt_filtered = filter_processor.process_texts(dataset_txt_tokenized, n_jobs=30)
    # Write to file
    txt_writer(dataset_txt_filtered, dataset_processed_path+"/upmc_example_processed.txt")

    print()
