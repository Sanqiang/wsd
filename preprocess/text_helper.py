"""
Helper functions for preprocessing DataSet.
-- DeID replacement
-- Tokenizer
"""

import re
import tqdm
from joblib import Parallel, delayed
from util import constant


class TokenHelper(object):

    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)


class TextHelper(object):

    def rui_tokenize(self, text, lowercase=True):
        """Simple fast tokenizer based on Regex."""
        text = text.strip()
        if lowercase:
            text = text.lower()

        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
        # tokenize by non-letters
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%@]', text))

        return tokens

    def process_context(self, text, lowercase=True, remove_digit=True):
        tokens = self.rui_tokenize(
            text, lowercase=lowercase)

        # Remove digit to constant.SYMBOL_NUM
        if remove_digit:
            tokens = [w if not re.match('^\d+$', w) else constant.NUM for w in tokens]

        # Remove non-asc2 word
        tokens = [w for w in tokens if TokenHelper.is_ascii(w)]

        return tokens


class TextPreProcessor:
    """
    General pipeline for text pre-processing.
    -- Initialize pre-processing functions
    -- Use for single text
    -- Use for list of texts
    """

    def __init__(self, preprocess_function=None,
                 add_annotation_function=None,
                 remove_function=None,
                 deid_replace_function=None,
                 tokenize_function=None,
                 postprocess_function=None):
        """
        Initialize pre-processing functions.

        :param list of functions, input & output of each functions must be a single text string.
        """
        process_function_list = [
            preprocess_function,
            add_annotation_function,
            remove_function,
            deid_replace_function,
            tokenize_function,
            postprocess_function]
        self.process_function_list = [func for func in process_function_list if callable(func)]

    def process_single_text(self, txt):
        for func in self.process_function_list:
            txt = func(txt)
        return txt

    def process_texts(self, txt_list, n_jobs=1):
        print("Pre-processing texts (n_jobs = %d)..." % n_jobs)
        txt_list_processed = Parallel(n_jobs=n_jobs, verbose=3)(delayed(self.process_single_text)(txt) for txt in txt_list)
        return txt_list_processed


def sub_patterns(txt, pattern_list, sub_string):
    for pattern in pattern_list:
        txt = re.sub(pattern, sub_string, txt)
    return txt


############################
# DeID replacement for MIMIC (ShARe/CLEF)
############################

def sub_deid_patterns(txt):
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

