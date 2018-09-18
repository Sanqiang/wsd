"""
Process final_cleaned_sense_inventory.json into pickle, the pickle will contains a dict that
'abbr-cui-longforms': mapper for abbr -> cui -> longforms
'longform-abbr_cui': mapper for longform -> (abbr, cui)
"""
import json
import re
import pickle
from nltk.corpus import stopwords
from collections import defaultdict

stopword_set = set(stopwords.words('english'))

PATH_INVENTORY_JSON = '/Users/sanqiangzhao/git/wsd_data/mimic/final_cleaned_sense_inventory.json'
PATH_PROCESSED_INVENTORY_PKL = '/Users/sanqiangzhao/git/wsd_data/mimic/final_cleaned_sense_inventory.cased.processed.pkl'


def unify_abbr(ignore_unigram_longform=True):
    """
    Process inventory to pickle. Check the header of this python file.
    :return:
    """
    def guess_abbrs(longform):
        """Guess a list of abbr for a longform."""
        abbrs = set()
        initals = [w[0] for w in re.split(' |-', longform) if w]
        abbrs.add(''.join(initals))
        abbrs.add(''.join([c for c in initals if c.isalpha()]))
        initals2 = [w[0] for w in re.split(
            ' |-', longform.lower()) if w and w not in stopword_set]
        abbrs.add(''.join(initals2))
        abbrs.add(''.join([c for c in initals2 if c.isalpha()]))
        return abbrs

    def match_abbr(abbrs, estimate_abbrs):
        """Matched abbrs from a list of estimate_abbrs(got from longform)."""
        for abbr in abbrs:
            if abbr in estimate_abbrs:
                return abbr

        estimate_abbrs = set([''.join(sorted(w)) for w in estimate_abbrs])
        for abbr in abbrs:
            abbr = ''.join(sorted(abbr))
            if abbr in estimate_abbrs:
                return abbr

        return None

    output = {}
    mapper = defaultdict(lambda: defaultdict(list)) # mapper for abbr -> cui -> longforms

    cnt_match = 0
    cnt_nomatch = 0
    cnt_processed = 0
    for line in open(PATH_INVENTORY_JSON):
        cnt_processed += 1
        if cnt_processed % 1000 == 0:
            print('Processed %s.' % cnt_processed)

        obj = json.loads(line)
        cui = obj['CUI']

        abbrs = obj['ABBR']
        abbrs = [abbr.lower() for abbr in abbrs]

        longforms = obj['LONGFORM']
        longforms = list(set([longform.lower() for longform in longforms]))

        if len(abbrs) == 1:
            cnt_match += len(longforms)

            if ignore_unigram_longform:
                longforms = [longform for longform in longforms if len(longform.split()) > 1]

            mapper[abbrs[0]][cui].extend(longforms)
            continue

        for longform in longforms:
            if ignore_unigram_longform and len(longform.split()) <= 1:
                continue

            estimate_abbrs = set(guess_abbrs(longform))

            cur_abbr = match_abbr(abbrs, estimate_abbrs)

            if cur_abbr:
                cnt_match += 1
                mapper[cur_abbr][cui].append(longform)
                # print('%s is %s from %s' % (longform, cur_abbr, abbrs))
            else:
                cnt_nomatch += 1
                # print('%s not matched for estimate_abbrs: %s and abbrs:%s' % (longform, estimate_abbrs, abbrs))
    print('match:%s and nonmatch%s' % (cnt_match, cnt_nomatch))

    output['abbr-cui-longforms'] = output
    # Reverse Process
    rmapper = {}
    for abbr in mapper:
        for cui in mapper[abbr]:
            for longform in mapper[abbr][cui]:
                if longform in rmapper:
                    estimate_abbrs = set(guess_abbrs(longform))
                    # Due to some error in inventory (e.g. therapeutic abortion occurs in both ea and ta), we do a postprocess.
                    if abbr in estimate_abbrs:
                        rmapper[longform] = (abbr, cui)
                    if abbr != rmapper[longform][0] and abbr in estimate_abbrs and rmapper[longform][0] in estimate_abbrs:
                        assert 'fatal error! a longform belongs to two correct abbrs.'
                else:
                    rmapper[longform] = (abbr, cui)
    output['longform-abbr_cui'] = rmapper
    with open(PATH_PROCESSED_INVENTORY_PKL, 'wb') as output_file:
        pickle.dump(output, output_file)


unify_abbr()