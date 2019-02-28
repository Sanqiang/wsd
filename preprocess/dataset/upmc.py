"""
Helper functions for MIMIC-III dataset.

"""
import re
import tqdm
import operator
import multiprocessing as mp

from baseline.dataset_helper import DataSetPaths
from preprocess.file_helper import txt_writer, pickle_reader

from mimic.string_utils.suffix_tree import SuffixTree

def split_non_valid_cui(txt):
    txt = re.sub(r"(abbr\|\w+\|C?\d+)([\\/])(?=[^ ]+?)", r"\1 \2 ", txt)
    return txt


def longform_replacer_job(txt_list, sense_list, rmapper):
    '''
    Find a longform in the text, replace it to the target format abbr|
    :param idxs:
    :param txt_list:
    :param sense_list:
    :param txt_queue:
    :param rmapper:
    :return:
    '''


    for idx, txt in enumerate(txt_list):

        text_st = SuffixTree(txt, case_insensitive=False)

        for sense_id, ((longform, sense), longform_token_set) \
                in enumerate(zip(long_sense_dict.items(), longform_token_sets)):
            if len(longform_token_set & text_tokens_set) > 0:
                if text_st.has_substring(longform):
                    note_present_longform_dict[longform] = sense.cui
                    sense_present_records = dataset_sense_present_dict.get(sense.cui, [])
                    sense_present_records.append((longform, note_id))
                    dataset_sense_present_dict[sense.cui] = sense_present_records



        for sense in senses:
            longform = sense.lower()
            if longform in rmapper:
                # rmapper[longform][0] is abbr, rmapper[longform][1] is CUI
                txt = re.sub(
                    r'\b' + longform + r'\b',
                    ' abbr|%s|%s|%s ' % (
                        '_'.join(rmapper[longform][0].split('\W+')),
                        rmapper[longform][1],
                        '_'.join(longform.split('\W+'))
                    ),
                    txt)
        txt_queue.put((idx, txt))


def longform_replacer(txt_list, present_senses_list, rmapper, n_jobs=8):
    print("Replacing long forms...")

    txt_list_processed = []
    q = mp.Queue()
    # how many docs per worker
    step = len(txt_list) // n_jobs
    # assign workers
    workers = [mp.Process(target=longform_replacer_job,
                          args=(
                              range(i * step, (i + 1) * step),
                              txt_list[i * step:(i + 1) * step],
                              present_senses_list[i * step:(i + 1) * step],
                              q,
                              rmapper,
                          )) for i in range(n_jobs - 1)]
    workers.append(mp.Process(target=longform_replacer_job,
                              args=(
                                  range((n_jobs - 1) * step, len(txt_list)),
                                  txt_list[(n_jobs - 1) * step:],
                                  present_senses_list[(n_jobs - 1) * step:],
                                  q,
                                  rmapper,
                              )))
    # start working
    with tqdm.tqdm(total=len(txt_list)) as pbar:
        for i in range(n_jobs):
            workers[i].start()
        for i in range(len(txt_list)):
            txt_list_processed.append(q.get())
            pbar.update()
        for i in range(n_jobs):
            workers[i].join()

    txt_list_processed_sorted = sorted(txt_list_processed, key=operator.itemgetter(0))
    return [txt for _, txt in txt_list_processed_sorted]


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################
    # BASE_FOLDER = '/home/mengr/Project/wsd/wsd_data/'
    dataset_paths = DataSetPaths(environment='luoz3_x1')
    DATASET_PATH = dataset_paths.upmc_all_no_mark_txt
    OUTPUT_PATH = dataset_paths.upmc_all_no_mark_folder

    PATH_PROCESSED_INVENTORY_PKL = dataset_paths.sense_inventory_pkl

    # Get pickle generated from mimic_inventory.py
    inventory = pickle_reader(PATH_PROCESSED_INVENTORY_PKL)
    inventory_rmapper = inventory['longform-abbr_cui']

    ######################################
    # Processing
    ######################################
    txt_list = list(open(DATASET_PATH, 'r').readlines())
    print("Loaded %d docs from %s" % (len(txt_list), DATASET_PATH))
    # Replace Long forms to abbrs
    mimic_txt_processed = longform_replacer(txt_list, inventory_rmapper, n_jobs=50)
    # Save to file
    txt_writer(mimic_txt_processed, OUTPUT_PATH+'train_no_mark_longform_replaced.txt')
