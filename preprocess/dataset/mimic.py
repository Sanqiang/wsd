"""
Helper functions for MIMIC-III dataset.

"""
import os
import re
import json
import tqdm
from joblib import Parallel, delayed
from preprocess.text_helper import sub_patterns, white_space_remover
from preprocess.text_helper import TextPreProcessor, CoreNLPTokenizer, TextBaseHelper
from preprocess.file_helper import txt_reader, txt_writer, json_writer, pickle_reader


# DeID replacement for MIMIC (ShARe/CLEF)
def sub_deid_patterns_mimic(txt):
    # DATE
    txt = sub_patterns(txt, [
        # normal date
        r"\[\*\*(\d{4}-)?\d{1,2}-\d{1,2}\*\*\]",
        # date range
        r"\[\*\*Date [rR]ange.+\*\*\]",
        # month/year
        r"\[\*\*-?\d{1,2}-?/\d{4}\*\*\]",
        # year
        r"\[\*\*(Year \([24] digits\).+)\*\*\]",
        # holiday
        r"\[\*\*Holiday.+\*\*\]",
        # XXX-XX-XX
        r"\[\*\*\d{3}-\d{1,2}-\d{1,2}\*\*\]",
        # date with format
        r"\[\*\*(Month(/Day)?(/Year)?|Year(/Month)?(/Day)?|Day Month).+\*\*\]",
        # uppercase month year
        r"\[\*\*(January|February|March|April|May|June|July|August|September|October|November|December).+\*\*\]",
    ], "DATE-DEID")

    # NAME
    txt = sub_patterns(txt, [
        # name
        r"\[\*\*(First |Last )?Name.+\*\*\]",
        # name initials
        r"\[\*\*Initial.+\*\*\]",
        # name with sex
        r"\[\*\*(Female|Male).+\*\*\]",
        # doctor name
        r"\[\*\*Doctor.+\*\*\]",
        # known name
        r"\[\*\*Known.+\*\*\]",
        # wardname
        r"\[\*\*Wardname.+\*\*\]",
    ], "NAME-DEID")

    # INSTITUTION
    txt = sub_patterns(txt, [
        # hospital
        r"\[\*\*Hospital.+\*\*\]",
        # university
        r"\[\*\*University.+\*\*\]",
        # company
        r"\[\*\*Company.+\*\*\]",
    ], "INSTITUTION-DEID")

    # clip number
    txt = sub_patterns(txt, [
        r"\[\*\*Clip Number.+\*\*\]",
    ], "CLIP-NUMBER-DEID")

    # digits
    txt = sub_patterns(txt, [
        r"\[\*\* ?\d{1,5}\*\*\]",
    ], "DIGITS-DEID")

    # tel/fax
    txt = sub_patterns(txt, [
        r"\[\*\*Telephone/Fax.+\*\*\]",
        r"\[\*\*\*\*\]",
    ], "PHONE-DEID")

    # EMPTY
    txt = sub_patterns(txt, [
        r"\[\*\* ?\*\*\]",
    ], "EMPTY-DEID")

    # numeric identifier
    txt = sub_patterns(txt, [
        r"\[\*\*Numeric Identifier.+\*\*\]",
    ], "NUMERIC-DEID")

    # AGE
    txt = sub_patterns(txt, [
        r"\[\*\*Age.+\*\*\]",
    ], "AGE-DEID")

    # PLACE
    txt = sub_patterns(txt, [
        # country
        r"\[\*\*Country.+\*\*\]",
        # state
        r"\[\*\*State.+\*\*\]",
    ], "PLACE-DEID")

    # STREET-ADDRESS
    txt = sub_patterns(txt, [
        r"\[\*\*Location.+\*\*\]",
        r"\[\*\*.+ Address.+\*\*\]",
    ], "STREET-ADDRESS-DEID")

    # MD number
    txt = sub_patterns(txt, [
        r"\[\*\*MD Number.+\*\*\]",
    ], "MD-NUMBER-DEID")

    # other numbers
    txt = sub_patterns(txt, [
        # job
        r"\[\*\*Job Number.+\*\*\]",
        # medical record number
        r"\[\*\*Medical Record Number.+\*\*\]",
        # SSN
        r"\[\*\*Social Security Number.+\*\*\]",
        # unit number
        r"\[\*\*Unit Number.+\*\*\]",
        # pager number
        r"\[\*\*Pager number.+\*\*\]",
        # serial number
        r"\[\*\*Serial Number.+\*\*\]",
        # provider number
        r"\[\*\*Provider Number.+\*\*\]",
    ], "OTHER-NUMBER-DEID")

    # info
    txt = sub_patterns(txt, [
        r"\[\*\*.+Info.+\*\*\]",
    ], "INFO-DEID")

    # E-mail
    txt = sub_patterns(txt, [
        r"\[\*\*E-mail address.+\*\*\]",
        r"\[\*\*URL.+\*\*\]"
    ], "EMAIL-DEID")

    # other
    txt = sub_patterns(txt, [
        r"\[\*\*(.*)?\*\*\]",
    ], "OTHER-DEID")
    return txt


def longform_replacer_job(txt, senses, rmapper, splitter):
    for sense in senses:
        longform = sense.lower()
        if longform in rmapper:
            txt = re.sub(
                r'\b' + longform + r'\b',
                '%s%s%s' % (rmapper[longform][0], splitter, rmapper[longform][1]),
                txt)
    return txt


def longform_replacer(txt_list, present_senses_list, rmapper, n_jobs=8, splitter="\u2223"):
    txt_list_processed = []
    for txt, present_senses in zip(txt_list, tqdm.tqdm(present_senses_list)):
        txt_list_processed.append(longform_replacer_job(txt, present_senses, rmapper, splitter))
    # txt_list_processed = Parallel(n_jobs=n_jobs, verbose=10)(delayed(longform_replacer_job)(txt, present_senses, rmapper, splitter) for txt, present_senses in zip(txt_list, present_senses_list))
    return txt_list_processed


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    PATH_FOLDER = '/home/zhaos5/projs/wsd/wsd_data/mimic/find_longform_mimic/'
    PATH_FOLDER_PROCESSED = '/home/luoz3/data/mimic/processed/'
    PATH_PROCESSED_INVENTORY_PKL = '/home/zhaos5/projs/wsd/wsd_data/mimic/final_cleaned_sense_inventory.cased.processed.pkl'

    # Get pickle generated from mimic_inventory.py
    inventory = pickle_reader(PATH_PROCESSED_INVENTORY_PKL)
    inventory_rmapper = inventory['longform-abbr_cui']

    ######################################
    # Processing
    ######################################

    # Initialize processor and tokenizer
    processor = TextPreProcessor([
        white_space_remover,
        sub_deid_patterns_mimic])

    toknizer = CoreNLPTokenizer()

    # for i in range(42):

    # read file
    filename = 'processed_text_chunk_%s.json' % 1
    print("-"*50)
    print("Start File for %s" % filename)
    mimic_txt = []
    mimic_present_senses = []
    for line in open(PATH_FOLDER+filename, "r"):
        obj = json.loads(line)
        text = obj['TEXT']
        present_senses = obj['present_senses']
        mimic_txt.append(text)
        mimic_present_senses.append(present_senses)

    # print(mimic_txt[0])
    txt = mimic_txt[0]
    print(txt)
    print('#'*50)
    print('#' * 50)
    txt = white_space_remover(txt)
    # txt = processor.process_single_text(txt)
    print(txt)
    # txt = toknizer.process_single_text(txt)

        # # pre-processing
        # mimic_txt = processor.process_texts(mimic_txt, n_jobs=30)
        # # tokenizing
        # mimic_txt_tokenized = toknizer.process_texts(mimic_txt, n_jobs=40)
        # # Replace Long forms to abbrs
        # mimic_txt_processed = longform_replacer(mimic_txt_tokenized, mimic_present_senses, inventory_rmapper, n_jobs=1)
        # # Save to file
        # txt_writer(mimic_txt_processed, PATH_FOLDER_PROCESSED+'%s.txt' % filename[:-5])
