"""
Processing UPMC data.
"""
import os
import re
import tqdm
import random
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


def process_annotated_data(txt_preprocessed_path, upmc_processed_path, train_ratio=0.8, n_jobs=30):
    os.makedirs(upmc_processed_path, exist_ok=True)
    upmc_txt_annotated = txt_reader(txt_preprocessed_path)
    # pre-processing
    upmc_txt = all_processor.process_texts(upmc_txt_annotated, n_jobs=n_jobs)
    # train/test split (80% train)
    random.shuffle(upmc_txt)
    num_instances = len(upmc_txt)
    train_idx = set(random.sample(range(num_instances), int(train_ratio*num_instances)))
    upmc_train_txt = []
    upmc_test_txt = []
    for idx, txt in enumerate(tqdm.tqdm(upmc_txt)):
        if idx in train_idx:
            upmc_train_txt.append(txt)
        else:
            upmc_test_txt.append(txt)
    # Write to file
    txt_writer(upmc_train_txt, upmc_processed_path+"/upmc_train.txt")
    txt_writer(upmc_test_txt, upmc_processed_path+"/upmc_test.txt")


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    data_path = "/home/luoz3/wsd_data"
    # dataset_path = data_path + "/upmc/example"
    # dataset_processed_path = data_path + "/upmc/example/processed"
    # os.makedirs(dataset_processed_path, exist_ok=True)

    # # fix annotation error
    # with open(dataset_path + "/training_data.txt") as input, open(dataset_path + "/training_data_fixed.txt", "w") as output:
    #     for line in input:
    #         new_line = " ".join([replace(token) for token in line.rstrip("\n").split(" ")])
    #         output.write(new_line + "\n")

    #############################
    # Process DataSet documents (only one word abbrs)
    #############################

    # dataset_txt_annotated = txt_reader(dataset_path + "/training_data_fixed.txt")

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover])

    toknizer = CoreNLPTokenizer()

    token_filter = TextTokenFilter()
    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    all_processor = TextProcessor([
        white_space_remover,
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    # # pre-processing
    # dataset_txt = processor.process_texts(dataset_txt_annotated, n_jobs=30)
    # # tokenizing
    # dataset_txt_tokenized = toknizer.process_texts(dataset_txt, n_jobs=30)
    # # Filter trivial tokens and Remove repeat non-words
    # dataset_txt_filtered = filter_processor.process_texts(dataset_txt_tokenized, n_jobs=30)
    # # Write to file
    # txt_writer(dataset_txt_filtered, dataset_processed_path+"/upmc_example_processed.txt")


    # ######################################
    # # Processing UPMC AB
    # ######################################
    #
    # upmc_ab_path = data_path + "/upmc/AB"
    # upmc_ab_processed_path = upmc_ab_path + "/processed"
    # os.makedirs(upmc_ab_processed_path, exist_ok=True)
    #
    # upmc_ab_txt_annotated = txt_reader(upmc_ab_path + "/training_data_AB.txt")
    # # pre-processing
    # upmc_ab_txt = processor.process_texts(upmc_ab_txt_annotated, n_jobs=30)
    # # tokenizing
    # upmc_ab_txt_tokenized = toknizer.process_texts(upmc_ab_txt, n_jobs=30)
    # # Filter trivial tokens and Remove repeat non-words
    # upmc_ab_txt_filtered = filter_processor.process_texts(upmc_ab_txt_tokenized, n_jobs=30)
    #
    # # train/test split (80% train)
    # random.shuffle(upmc_ab_txt_filtered)
    # num_instances = len(upmc_ab_txt_filtered)
    # train_idx = random.sample(range(num_instances), int(0.8*num_instances))
    # upmc_ab_train_txt = []
    # upmc_ab_test_txt = []
    # for idx, txt in enumerate(upmc_ab_txt_filtered):
    #     if idx in train_idx:
    #         upmc_ab_train_txt.append(txt)
    #     else:
    #         upmc_ab_test_txt.append(txt)
    # # Write to file
    # txt_writer(upmc_ab_train_txt, upmc_ab_processed_path+"/upmc_ab_train.txt")
    # txt_writer(upmc_ab_test_txt, upmc_ab_processed_path + "/upmc_ab_test.txt")


    ######################################
    # Processing UPMC AD
    ######################################

    process_annotated_data("/home/wangz12/scripts/generate_trainning_data/training_data_AD.txt", data_path + "/upmc/AD/processed")

    print()
