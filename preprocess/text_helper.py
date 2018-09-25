import re
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

        # Remove repeatly non-words, e.g. num num into num
        ntokens = []
        for token_id, token in enumerate(tokens):
            if token.isalpha() or token_id == 0 or tokens[token_id-1] != token:
                ntokens.append(token)

        return ntokens



