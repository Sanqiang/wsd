"""
Helper functions for MIMIC-III dataset.

"""
import os
import re
import json
import tqdm
import operator
import multiprocessing as mp
from preprocess.text_helper import sub_patterns, white_space_remover, repeat_non_word_remover
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from preprocess.file_helper import txt_reader, txt_writer, json_writer, pickle_reader


# DeID replacement for MIMIC (ShARe/CLEF)
def sub_deid_patterns_mimic(txt):
    # DATE
    txt = sub_patterns(txt, [
        # normal date
        r"\[\*\*(\d{4}-)?\d{1,2}-\d{1,2}\*\*\]",
        # date range
        r"\[\*\*Date [rR]ange.+?\*\*\]",
        # month/year
        r"\[\*\*-?\d{1,2}-?/\d{4}\*\*\]",
        # year
        r"\[\*\*(Year \([24] digits\).+?)\*\*\]",
        # holiday
        r"\[\*\*Holiday.+?\*\*\]",
        # XXX-XX-XX
        r"\[\*\*\d{3}-\d{1,2}-\d{1,2}\*\*\]",
        # date with format
        r"\[\*\*(Month(/Day)?(/Year)?|Year(/Month)?(/Day)?|Day Month).+?\*\*\]",
        # uppercase month year
        r"\[\*\*(January|February|March|April|May|June|July|August|September|October|November|December).+?\*\*\]",
    ], "DATE-DEID")

    # NAME
    txt = sub_patterns(txt, [
        # name
        r"\[\*\*(First |Last )?Name.+?\*\*\]",
        # name initials
        r"\[\*\*Initial.+?\*\*\]",
        # name with sex
        r"\[\*\*(Female|Male).+?\*\*\]",
        # doctor name
        r"\[\*\*Doctor.+?\*\*\]",
        # known name
        r"\[\*\*Known.+?\*\*\]",
        # wardname
        r"\[\*\*Wardname.+?\*\*\]",
    ], "NAME-DEID")

    # INSTITUTION
    txt = sub_patterns(txt, [
        # hospital
        r"\[\*\*Hospital.+?\*\*\]",
        # university
        r"\[\*\*University.+?\*\*\]",
        # company
        r"\[\*\*Company.+?\*\*\]",
    ], "INSTITUTION-DEID")

    # clip number
    txt = sub_patterns(txt, [
        r"\[\*\*Clip Number.+?\*\*\]",
    ], "CLIP-NUMBER-DEID")

    # digits
    txt = sub_patterns(txt, [
        r"\[\*\* ?\d{1,5}\*\*\]",
    ], "DIGITS-DEID")

    # tel/fax
    txt = sub_patterns(txt, [
        r"\[\*\*Telephone/Fax.+?\*\*\]",
        r"\[\*\*\*\*\]",
    ], "PHONE-DEID")

    # EMPTY
    txt = sub_patterns(txt, [
        r"\[\*\* ?\*\*\]",
    ], "EMPTY-DEID")

    # numeric identifier
    txt = sub_patterns(txt, [
        r"\[\*\*Numeric Identifier.+?\*\*\]",
    ], "NUMERIC-DEID")

    # AGE
    txt = sub_patterns(txt, [
        r"\[\*\*Age.+?\*\*\]",
    ], "AGE-DEID")

    # PLACE
    txt = sub_patterns(txt, [
        # country
        r"\[\*\*Country.+?\*\*\]",
        # state
        r"\[\*\*State.+?\*\*\]",
    ], "PLACE-DEID")

    # STREET-ADDRESS
    txt = sub_patterns(txt, [
        r"\[\*\*Location.+?\*\*\]",
        r"\[\*\*.+? Address.+?\*\*\]",
    ], "STREET-ADDRESS-DEID")

    # MD number
    txt = sub_patterns(txt, [
        r"\[\*\*MD Number.+?\*\*\]",
    ], "MD-NUMBER-DEID")

    # other numbers
    txt = sub_patterns(txt, [
        # job
        r"\[\*\*Job Number.+?\*\*\]",
        # medical record number
        r"\[\*\*Medical Record Number.+?\*\*\]",
        # SSN
        r"\[\*\*Social Security Number.+?\*\*\]",
        # unit number
        r"\[\*\*Unit Number.+?\*\*\]",
        # pager number
        r"\[\*\*Pager number.+?\*\*\]",
        # serial number
        r"\[\*\*Serial Number.+?\*\*\]",
        # provider number
        r"\[\*\*Provider Number.+?\*\*\]",
    ], "OTHER-NUMBER-DEID")

    # info
    txt = sub_patterns(txt, [
        r"\[\*\*.+?Info.+?\*\*\]",
    ], "INFO-DEID")

    # E-mail
    txt = sub_patterns(txt, [
        r"\[\*\*E-mail address.+?\*\*\]",
        r"\[\*\*URL.+?\*\*\]"
    ], "EMAIL-DEID")

    # other
    txt = sub_patterns(txt, [
        r"\[\*\*(.*)?\*\*\]",
    ], "OTHER-DEID")
    return txt


def split_non_valid_cui(txt):
    txt = re.sub(r"(abbr\|\w+\|C?\d+)([\\/])(?=[^ ]+?)", r"\1 \2 ", txt)
    return txt


def longform_replacer_job(idxs, txt_list, sense_list, txt_queue, rmapper):
    '''
    Find a longform in the text, replace it to the target format abbr|
    :param idxs:
    :param txt_list:
    :param sense_list:
    :param txt_queue:
    :param rmapper:
    :return:
    '''
    for idx, txt, senses in zip(idxs, txt_list, sense_list):
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
    BASE_FOLDER = '/Users/memray/Project/upmc_wsd/wsd_data/'

    PATH_FOLDER = BASE_FOLDER + 'mimic/find_longform_mimic/'
    PATH_FOLDER_PROCESSED = BASE_FOLDER + 'mimic/processed/'

    if not os.path.exists(PATH_FOLDER_PROCESSED):
        os.makedirs(PATH_FOLDER_PROCESSED)

    PATH_PROCESSED_INVENTORY_PKL = BASE_FOLDER + 'sense_inventory/final_cleaned_sense_inventory.cased.processed.pkl'

    # Get pickle generated from mimic_inventory.py
    inventory = pickle_reader(PATH_PROCESSED_INVENTORY_PKL)
    inventory_rmapper = inventory['longform-abbr_cui']

    ######################################
    # Processing
    ######################################

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover,
        sub_deid_patterns_mimic])

    toknizer = CoreNLPTokenizer()

    token_filter = TextTokenFilter()
    filter_processor = TextProcessor([
        token_filter])

    remove_repeat_processor = TextProcessor([repeat_non_word_remover])

    for i in range(42):

        # read file
        filename = 'processed_text_chunk_%s.json' % i
        print("-"*50)
        print("Start File for %s" % filename)
        mimic_txt = []
        mimic_present_senses = []

        if not os.path.exists(PATH_FOLDER+filename):
            continue

        for line in open(PATH_FOLDER+filename, "r"):
            obj = json.loads(line)
            text = obj['TEXT']
            present_senses = obj['present_senses']
            mimic_txt.append(text)
            mimic_present_senses.append(present_senses)

        # pre-processing
        mimic_txt = processor.process_texts(mimic_txt, n_jobs=30)
        # Replace Long forms to abbrs
        mimic_txt_processed = longform_replacer(mimic_txt_filtered, mimic_present_senses, inventory_rmapper, n_jobs=16)
        # tokenizing
        mimic_txt_tokenized = toknizer.process_texts(mimic_txt, n_jobs=40)
        # Filter trivial tokens
        mimic_txt_filtered = filter_processor.process_texts(mimic_txt_tokenized, n_jobs=40)
        # Remove repeat non-words
        mimic_txt_processed = remove_repeat_processor.process_texts(mimic_txt_processed, n_jobs=40)
        # Save to file
        txt_writer(mimic_txt_processed, PATH_FOLDER_PROCESSED+'%s.txt' % filename[:-5])
