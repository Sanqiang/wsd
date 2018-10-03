import re
import json
import pickle
from multiprocessing import Pool

from preprocess.text_helper import TextHelper

BASE_FOLDER = '/home/mengr/project/wsd/wsd_data/mimic/'
PATH_FOLDER = BASE_FOLDER + 'find_longform_mimic/'
PATH_PROCESSED_FOLDER = BASE_FOLDER + 'find_longform_mimic_processed/'

# Get pickle generated from mimic_inventory.py
PATH_PROCESSED_INVENTORY_PKL = BASE_FOLDER + 'final_cleaned_sense_inventory.cased.processed.pkl'
with open(PATH_PROCESSED_INVENTORY_PKL, 'rb') as inv_file:
    inventory = pickle.load(inv_file)
    inventory_rmapper = inventory['longform-abbr_cui']

def substitue_longform(filename):
    """
    Process the raw data to training data (subsite longform to CUI)
    "Consider prior anterior myocardial infarction, although it is non-diagnostic." will be
    "Consider prior abbr|AMI|C0340293|anterior_myocardial_infarction, although it is non-diagnostic."
    :param filename:
    :return:
    """
    print('Start File for %s' % (PATH_FOLDER + filename))
    text_helper = TextHelper()
    nlines = []

    for line in open(PATH_FOLDER + filename):
        obj = json.loads(line)
        text = obj['TEXT']
        text = text_helper.process_context(text)
        text = ' '.join(text)
        present_senses = obj['present_senses']
        for present_sense in present_senses:
            longform = present_sense.lower()
            if longform in inventory_rmapper:
                text = re.sub(
                    r'\b' + longform + r'\b',
                    'abbr|%s|%s|%s' % (
                        inventory_rmapper[longform][0],
                        inventory_rmapper[longform][1],
                        '_'.join(longform.split())
                    ),
                    text)
        # obj['PROCESSED_TEXT'] = text
        # nline = json.dumps(obj)
        nlines.append(text)

    nfilename = filename[:-5] + '.txt'
    open(PATH_PROCESSED_FOLDER + nfilename, 'w').write('\n'.join(nlines))
    print('Finished File for %s' % nfilename)

def thread_process(id):
    substitue_longform('processed_text_chunk_%s.json' % id)

p = Pool(50)
p.map(thread_process, list(range(0, 42)))