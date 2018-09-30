"""
Helper functions for pre-processing DataSet.
-- Simple Tokenizer
-- Stanford CoreNLP Tokenizer
-- TextProcessor for simple multiprocessing
-- Other general helper functions
"""

import re
import tqdm
import operator
import multiprocessing as mp
from joblib import Parallel, delayed
from pycorenlp import StanfordCoreNLP
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

        # Remove digit to constant.NUM
        if remove_digit:
            tokens = [w if not re.match('^\d+$', w) else constant.NUM for w in tokens]

        # Remove non-asc2 word
        tokens = [w for w in tokens if TokenHelper.is_ascii(w)]

        # Remove repeatly non-words, e.g. num num into num
        ntokens = []
        for token_id, token in enumerate(tokens):
            if token.isalpha() or token_id == 0 or tokens[token_id-1] != token:
                ntokens.append(token)

        return ntokens


class TextTokenFilter(object):
    """Must be used after tokenizer (tokens in txt can be split by space).

    Implement same functions (except tokenizer) in TextHelper from sanqiang.
    -- lowercase
    -- remove non-ASCII
    -- replace digits to constant.NUM
    -- remove repeat non-words (only keep one)
    """

    def __init__(self, lowercase=True, remove_non_ascii=True, remove_digit=True, remove_repeat=True):
        self.lowercase = lowercase
        self.remove_non_ascii = remove_non_ascii
        self.remove_digit = remove_digit
        self.remove_repeat = remove_repeat

    def __call__(self, txt):
        # To lowercase
        if self.lowercase:
            txt = txt.lower()

        # Split to tokens
        tokens = txt.split(" ")

        # Remove digit to constant.NUM
        if self.remove_digit:
            tokens = [w if not re.fullmatch(r'\d+(\D\d+)?', w) else constant.NUM for w in tokens]
        
        # Remove non-asc2 word
        if self.remove_non_ascii:
            tokens = [w for w in tokens if TokenHelper.is_ascii(w)]

        # Remove repeatly non-words, e.g. num num into num
        if self.remove_repeat:
            ntokens = []
            for token_id, token in enumerate(tokens):
                if token.isalpha() or token_id == 0 or tokens[token_id-1] != token:
                    ntokens.append(token)
            return " ".join(ntokens)
        else:
            return " ".join(tokens)


class TextBaseHelper(object):
    """
    Base Class for TextHelper.
    -- Use for single text
    -- Use for list of texts
    """

    def process_single_text(self, txt):
        raise NotImplementedError

    def process_texts(self, txt_list, n_jobs=1):
        raise NotImplementedError


class TextProcessor(TextBaseHelper):
    """
    General pipeline for text processing (except tokenizer).
    -- Initialize processing functions
    -- Use for single text
    -- Use for list of texts
    """

    def __init__(self, process_function_list=None):
        """
        Initialize processing functions.
        For example:
            -- annotation adder function ("AMI" to "abbr|AMI|C0340293")
            -- pattern remover functions
            -- DeID replacer function

        :param process_function_list: list of functions, input & output of each functions must be a single text string.
        """
        assert isinstance(process_function_list, list)
        self.process_function_list = [func for func in process_function_list if callable(func)]

    def process_single_text(self, txt):
        for func in self.process_function_list:
            txt = func(txt)
        return txt

    def process_texts(self, txt_list, n_jobs=8):
        print("Processing texts (n_jobs = %d)..." % n_jobs)
        txt_list_processed = Parallel(n_jobs=n_jobs, verbose=1)(delayed(self.process_single_text)(txt) for txt in txt_list)
        return txt_list_processed


class CoreNLPTokenizer(TextBaseHelper):
    """
    Stanford CoreNLP Tokenizer (multiprocessing optimized version).
    -- Use for single text: process_single_text
    -- Use for list of texts: process_texts
    """

    def __init__(self, server_port=9000, combine_splitter="\u21F6"):
        """
        Initialize Stanford CoreNLP server.

        Must open CoreNLP server in terminal (in CoreNLP folder) first by:
        "java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -quiet"

        :param server_port: The port of CoreNLP Java Server
        :param combine_splitter: A marker string to split multiple texts
        """
        self.stanford_nlp = StanfordCoreNLP('http://localhost:%d' % server_port)
        self.combine_splitter = combine_splitter

    def process_single_text(self, txt):
        tokens = self.stanford_nlp.annotate(txt, properties={
            'annotators': 'tokenize',
            'outputFormat': 'json'
        })['tokens']
        content = []
        for token in tokens:
            word = token['word']
            content.append(word)
        txt_tokenized = " ".join(content)
        txt_tokenized = self.filter_splitters(txt_tokenized)
        return txt_tokenized

    def process_texts(self, txt_list, n_jobs=32):
        """
        Tokenize list of texts.

        :param txt_list: List of input texts
        :param n_jobs: Number of workers
        :return: List of tokenized texts
        """
        # combine texts
        txt_list_combined = self._combine_texts(txt_list)

        print("Tokenizing (n_jobs = %d)..." % n_jobs)
        write_list = []
        q = mp.Queue()
        # how many docs per worker
        step = len(txt_list_combined) // n_jobs
        workers = [mp.Process(target=self._job, args=(range(i * step, (i + 1) * step), txt_list_combined[i * step:(i + 1) * step], q))
                   for i in range(n_jobs - 1)]
        workers.append(mp.Process(target=self._job, args=(
            range((n_jobs - 1) * step, len(txt_list_combined)), txt_list_combined[(n_jobs - 1) * step:], q)))

        with tqdm.tqdm(total=len(txt_list_combined)) as pbar:
            for i in range(n_jobs):
                workers[i].start()
            for i in range(len(txt_list_combined)):
                write_list.append(q.get())
                pbar.update()
            for i in range(n_jobs):
                workers[i].join()

        write_list_sorted = sorted(write_list, key=operator.itemgetter(0))
        # split multiple texts
        txt_list_processed = self._split_texts(write_list_sorted)
        return txt_list_processed

    def _combine_texts(self, txt_list, max_length=80000):
        """
        Combine multiple texts to one text, in order to speed up.

        :param txt_list: List of original text strings
        :param max_length: Maximum length of a combined text
        :return: List of combined texts
        """
        multi_texts = []
        len_count = 0
        texts_combined = []
        for txt in txt_list:
            temp_len_count = len_count + len(txt)

            if temp_len_count >= max_length:
                multi_texts.append(self.combine_splitter.join(texts_combined))
                len_count = 0
                texts_combined = []

            len_count += len(txt)
            texts_combined.append(txt)
        # combine last block of texts
        multi_texts.append(self.combine_splitter.join(texts_combined))
        return multi_texts

    def _split_texts(self, multi_texts_sorted):
        """
        Split multi-texts to several texts.

        :return: List of splitted texts
        """
        # decode to one doc per line
        texts_split_list = []
        for multi_doc in multi_texts_sorted:
            texts_split_list.extend(multi_doc[1].split(self.combine_splitter))
        return texts_split_list

    def _job(self, idxs, docs, content_queue, debug=False):
        for idx, doc in zip(idxs, docs):
            if debug:
                try:
                    tokens = self.stanford_nlp.annotate(doc, properties={
                        'annotators': 'tokenize',
                        'outputFormat': 'json'
                    })['tokens']
                    content = []
                except TypeError:
                    with open("%d-error.txt" % idx, "w") as file:
                        file.write(doc)
            else:
                tokens = self.stanford_nlp.annotate(doc, properties={
                    'annotators': 'tokenize',
                    'outputFormat': 'json'
                })['tokens']
                content = []

            for token in tokens:
                word = token['word']
                content.append(word)
            txt_tokenized = " ".join(content)
            txt_tokenized = self.filter_splitters(txt_tokenized)
            content_queue.put((idx, txt_tokenized))

    def filter_splitters(self, txt):
        """
        Remove the space around splitters.

        :param txt:
        :return:
        """
        txt = txt.replace(" %s " % self.combine_splitter, self.combine_splitter)

        # abbr annotation pattern (senses must be represented by CUI or digits)
        annotate_ptn = re.compile(r"abbr \| (\w+?) \| (C?\d+)")

        txt = re.sub(annotate_ptn, r"abbr|\1|\2", txt)
        return txt


class AbbrInventoryBuilder(TextBaseHelper):
    """
    Build Abbreviation Sense Inventory from processed texts.
    """
    # <todo> Zhimeng: Build abbr sense inventory from processed texts for all datasets.
    def __init__(self):
        raise NotImplementedError

    def process_single_text(self, txt):
        raise NotImplementedError

    def process_texts(self, txt_list, n_jobs=1):
        raise NotImplementedError


class AbbrInventory:
    # <todo> Zhimeng: General Abbr sense inventory for all datasets.

    def __init__(self):
        self.data = {}
        raise NotImplementedError

    def overlap(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


############################
# General Helper Functions
############################

def sub_patterns(txt, pattern_list, sub_string):
    """
    Replace list of patterns to a string.

    :param txt: Single Document String
    :param pattern_list: A list of Regex patterns (re.compile)
    :param sub_string: Substitution String
    :return:Processed Document String
    """
    for pattern in pattern_list:
        txt = re.sub(pattern, sub_string, txt)
    return txt


def white_space_remover(txt):
    """
    Remove '\n' and redundant spaces.

    :param txt: Single Document String
    :return: Processed Document String
    """
    # remove all "\n"
    txt = re.sub(r"\n", " ", txt)
    # remove all redundant spaces
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt
