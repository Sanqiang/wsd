"""
Processing UPMC batch 1-4.
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


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    data_path = "/home/luoz3/wsd_data"
    upmc_all_path = data_path + "/upmc/batch1_4"
    upmc_all_processed_path = upmc_all_path + "/processed"
    os.makedirs(upmc_all_processed_path, exist_ok=True)

    #############################
    # Process DataSet documents (only one word abbrs)
    #############################

    # Initialize processor and tokenizer
    token_filter = TextTokenFilter()
    processor = TextProcessor([
        white_space_remover,
        token_filter,
        repeat_non_word_remover,
    ])

    upmc_all_txt = txt_reader(data_path + "/upmc_batch1_4/upmc_no_mark_new.txt")
    # pre-processing
    upmc_all_txt = processor.process_texts(upmc_all_txt, n_jobs=30)
    # Write to file
    txt_writer(upmc_all_txt, upmc_all_processed_path+"/train_no_mark.txt")

    print()
