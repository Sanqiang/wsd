"""
Process data, build abbr index, train tfidf and word2vec.

"""
import gensim
import os
import pickle
import tqdm
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from collections import defaultdict, OrderedDict
from preprocess.file_helper import txt_writer, txt_reader, pickle_writer


def index_label_reader(index_list):
    """Decode index list to dict

    :param index_list:
    :return:
    """
    index_dict = OrderedDict()
    for doc_id, doc_pos in index_list:
        doc_pos = doc_pos.split(" ")
        temp_dict = []
        for i in doc_pos:
            pair = i.split(":")
            temp_dict.append((int(pair[0]), pair[1]))
        index_dict[doc_id] = temp_dict
    return index_dict


class Corpus:
    """
    Document generator
    """
    def __init__(self, txt_list):
        self.txt_list = txt_list

    def __iter__(self):
        for doc in tqdm.tqdm(self.txt_list):
            doc = doc.split(" ")
            yield doc


class AbbrIndex:
    """
    Abbr inverted index
    """
    def __init__(self, file=None):
        if file is not None:
            self.load(file)
        else:
            self.data = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return index_label_reader(self.data[key][1:])
        else:
            raise KeyError

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def save(self, file):
        pickle.dump(self.data, open(file, 'wb'))

    def load(self, file):
        self.data = pickle.load(open(file, 'rb'))

    def add_posting(self, key, doc_id, pos):
        self.data.setdefault(key, [0]).append((doc_id, " ".join(pos)))
        self.data[key][0] += len(pos)

    def num_instances(self, key):
        if key in self.data:
            return self.data[key][0]
        else:
            raise KeyError

    def abbr_index_min_cut(self, min_count):
        assert isinstance(min_count, int)
        new_abbr_index = AbbrIndex()
        for key, value in self.data.items():
            if self.num_instances(key) >= min_count:
                new_abbr_index.data[key] = value
        return new_abbr_index

    def abbr_index_min_max_cut(self, min_count, max_count):
        assert isinstance(min_count, int)
        assert isinstance(max_count, int)
        new_abbr_index = AbbrIndex()
        for key, value in self.data.items():
            num = self.num_instances(key)
            if num >= min_count and num <= max_count:
                new_abbr_index.data[key] = value
        return new_abbr_index


def load_dataset_corpus(file_path):
    """
    Load processed dataset, generate abbr index, and remove abbr marks.

    :param file_path:
    :return:
    """
    abbr_index = AbbrIndex()
    txt_post_processed = []
    for doc_idx, doc in enumerate(tqdm.tqdm(open(file_path, "r"))):
        doc_abbr = defaultdict(list)
        doc_processed = []
        tokens = doc.rstrip('\n').split(" ")
        for idx, token in enumerate(tokens):
            if token.startswith('abbr|'):
                _, abbr, cui = token.split('|')
                # add abbr info to inverted index
                doc_abbr[abbr].append(":".join([str(idx), cui]))
                doc_processed.append(abbr)
            else:
                doc_processed.append(token)
        txt_post_processed.append(" ".join(doc_processed))

        # convert doc_abbr dict to string
        for abbr, pos in doc_abbr.items():
            abbr_index.add_posting(abbr, doc_idx, pos)

    return abbr_index, txt_post_processed


def generate_train_files(txt_path, train_processed_path):
    # os.makedirs(train_processed_path, exist_ok=True)
    # # Find abbrs, build abbr index
    # print("Loading TRAIN data...")
    # abbr_index, train_no_mark = load_dataset_corpus(txt_path)
    # # save files
    # txt_writer(train_no_mark, train_processed_path + '/train_no_mark.txt')
    # abbr_index.save(train_processed_path + '/abbr_index_data.pkl')
    #
    # print("Training Word2Vec...")
    # model = gensim.models.Word2Vec(Corpus(train_no_mark), workers=30, min_count=1)
    # model.save(train_processed_path + '/train.model')

    train_no_mark = txt_reader(train_processed_path + '/train_no_mark.txt')
    print("Generating One-hot...")
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(train_no_mark)
    print("One-hot feature number: ", len(vectorizer.get_feature_names()))
    pickle_writer(vectorizer, train_processed_path + '/count_vectorizer.pkl')
    print("Computing Tf-idf...")
    tfidf_transformer = TfidfTransformer().fit(train_counts)
    pickle_writer(tfidf_transformer, train_processed_path + '/tfidf_transformer.pkl')


def generate_test_files(txt_path, test_processed_path):
    os.makedirs(test_processed_path, exist_ok=True)
    # Find abbrs, build abbr index
    print("Loading Test data...")
    abbr_index, test_no_mark = load_dataset_corpus(txt_path)
    # save files
    txt_writer(test_no_mark, test_processed_path + '/test_no_mark.txt')
    abbr_index.save(test_processed_path + '/abbr_index_data.pkl')


if __name__ == '__main__':

    PATH_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
    train_processed_path = '/home/luoz3/data/mimic/processed/train/'

    PATH_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'
    mimic_test_path = '/home/luoz3/data/mimic/processed/test/'
    data_path = '/home/luoz3/data/'
    msh_txt_path = data_path + 'msh/msh_processed/msh_processed.txt'
    msh_test_path = data_path + 'msh/msh_processed/test/'
    share_txt_path = data_path + 'share/processed/share_all_processed.txt'
    share_test_path = data_path + 'share/processed/test/'

    generate_train_files(PATH_TRAIN, train_processed_path)

    # generate_test_files(PATH_EVAL, mimic_test_path)
    # generate_test_files(msh_txt_path, msh_test_path)
    # generate_test_files(share_txt_path, share_test_path)
