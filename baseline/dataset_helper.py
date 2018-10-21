"""
Helper functions for processed DataSet files.

"""

import tqdm
from collections import Counter, defaultdict, namedtuple
from preprocess.file_helper import pickle_writer, txt_reader, pickle_reader


class DataSetPaths:
    """
    Paths of DataSets
    """
    data_path = '/home/luoz3/data/'

    # DataSet Corpus files
    mimic_train_txt = '/exp_data/wsd_data/mimic/train'
    mimic_eval_txt = '/exp_data/wsd_data/mimic/eval'
    # # mimic v1 (deprecated)
    # mimic_train_txt = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
    # mimic_eval_txt = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'

    share_txt = data_path + 'share/processed/share_all_processed.txt'
    msh_txt = data_path + 'msh/msh_processed/msh_processed.txt'

    # paths for processed files
    mimic_train_folder = data_path + 'mimic/processed/train/'
    mimic_test_folder = data_path + 'mimic/processed/test/'
    share_test_folder = data_path + 'share/processed/test/'
    msh_test_folder = data_path + 'msh/msh_processed/test/'


def process_token(token):
    """
    Unpack token (e.g., abbr|ab|C1234567|long_form).
    If instance format of DataSet change, please only change here.

    :param token:
    :return:
    """
    if token.startswith('abbr|'):
        items = token.split("|")
        if len(items) == 4:
            _, abbr, sense, long_form = items
            return abbr, sense, long_form
        elif len(items) == 3:
            _, abbr, sense = items
            return abbr, sense, None


class AbbrInstanceCollector:
    """
    Collect abbr instance information from processed DataSets.
    """
    def __init__(self, dataset_file_path):
        self.corpus = txt_reader(dataset_file_path)

    def generate_instance_collection(self, save_collection_path=None):
        """
        Collect list of instances (index, abbr, sense, long_form).

        :param save_collection_path:
        :return:
        """
        instance_collection = []
        global_instance_idx = 0
        for line in tqdm.tqdm(self.corpus):
            for token in line.split(" "):
                items = process_token(token)
                if items is not None:
                    abbr, sense, long_form = items
                    instance_collection.append(Instance(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense=sense,
                        long_form=long_form))
                    global_instance_idx += 1

        # save instance collection
        if save_collection_path is not None:
            pickle_writer(instance_collection, save_collection_path)
        return instance_collection

    def generate_counter(self, save_collection_path=None):
        """
        Generate Counters for every abbr-CUI mappings.

        :param save_collection_path:
        :return:
        """
        dataset_counter = defaultdict(Counter)
        for line in tqdm.tqdm(self.corpus):
            for token in line.split(" "):
                items = process_token(token)
                if items is not None:
                    abbr, sense, _ = items
                    dataset_counter[abbr].update([sense])

        # save DataSet Counter
        if save_collection_path is not None:
            pickle_writer(dataset_counter, save_collection_path)
        return dataset_counter

    def generate_inverted_index(self):
        """
        Generate abbr inverted index and remove abbr marks.

        :return:
        """
        from baseline.word_embedding import AbbrIndex
        abbr_index = AbbrIndex()
        txt_post_processed = []
        global_instance_idx = 0
        for doc_idx, doc in enumerate(tqdm.tqdm(self.corpus)):
            doc_abbr = defaultdict(list)
            doc_processed = []
            tokens = doc.rstrip('\n').split(" ")
            for idx, token in enumerate(tokens):
                items = process_token(token)
                if items is not None:
                    abbr, sense, _ = items
                    # add abbr info to inverted index
                    doc_abbr[abbr].append(":".join([str(global_instance_idx), str(idx), sense]))
                    doc_processed.append(abbr)
                    global_instance_idx += 1
                else:
                    doc_processed.append(token)
            txt_post_processed.append(" ".join(doc_processed))

            # convert doc_abbr dict to string
            for abbr, pos in doc_abbr.items():
                abbr_index.add_posting(abbr, doc_idx, pos)
        return abbr_index, txt_post_processed


###################################
# Comparision between Corpus
###################################

def get_cui_set(counter):
    cui_set = []
    for _, cuis in counter.items():
        cui_set.extend(list(cuis))
    return set(cui_set)


def dataset_summary(counter):
    abbrs = counter.keys()
    print("No.abbrs: ", len(abbrs))

    cui_set = get_cui_set(counter)
    print("No.CUIs: ", len(cui_set))

    # count number of instances in DataSet
    count_all_instances = 0
    for abbr, items in counter.items():
        for cui, count in items.items():
            count_all_instances += count
    print("No.instances", count_all_instances)
    print()
    return abbrs, cui_set, count_all_instances


def compare_dataset_summary(counter_train, counter_test):
    # collect CUIs
    cui_set_train = get_cui_set(counter_train)
    cui_set_test = get_cui_set(counter_test)
    print("No.CUIs on test: ", len(cui_set_test))
    print("CUI overlap ratio: ", len(cui_set_train & cui_set_test)/len(cui_set_test))

    # collect abbr info
    abbr_train = set(counter_train.keys())
    abbr_test = set(counter_test.keys())
    print("No. abbrs on test: ", len(abbr_test))
    print("Abbr overlap ratio: ", len(abbr_train & abbr_test)/len(abbr_test))
    print()


def compare_dataset_instances(counter_train, counter_test):
    count_all_instances = 0
    count_no_abbr_instances = 0
    count_overlap_instances = 0
    for abbr, items in counter_test.items():
        for cui, count in items.items():
            count_all_instances += count
            if abbr not in counter_train:
                count_no_abbr_instances += count
            elif cui in counter_train[abbr]:
                count_overlap_instances += count
    return count_overlap_instances, count_all_instances, count_all_instances-count_no_abbr_instances


Instance = namedtuple('Instance', ['index', 'abbr', 'sense', 'long_form'])
InstancePred = namedtuple('InstancePred', ['index', 'abbr', 'sense_pred'])


def evaluation(instance_collection_true, instance_collection_pred):
    """
    Evaluate accuracy based on instance collections.

    :param instance_collection_true:
    :param instance_collection_pred:
    :return:
    """
    assert len(instance_collection_true) == len(instance_collection_pred)
    count_correct, count_total = 0.0, 0.0
    for instance_true, instance_pred in zip(instance_collection_true, instance_collection_pred):
        assert instance_true.index == instance_pred.index
        if instance_true.sense == instance_pred.sense_pred:
            count_correct += 1.0
        count_total += 1.0

    acc = count_correct / count_total
    print('Accuray = %s' % acc)
    print()
    return acc


if __name__ == '__main__':
    dataset_paths = DataSetPaths()
    train_counter_path = dataset_paths.mimic_train_folder+'train_abbr_counter.pkl'

    # # build train collectors
    # mimic_train_collector = AbbrInstanceCollector(dataset_paths.mimic_train_txt)
    # mimic_train_counter = mimic_train_collector.generate_counter(train_counter_path)

    # read train counter from file
    mimic_train_counter = pickle_reader(train_counter_path)

    # summary of training set
    print("Summary of MIMIC train:")
    dataset_summary(mimic_train_counter)

    # build test collectors
    mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_eval_txt)
    share_collector = AbbrInstanceCollector(dataset_paths.share_txt)
    msh_collector = AbbrInstanceCollector(dataset_paths.msh_txt)

    # generate test counters
    mimic_test_counter = mimic_test_collector.generate_counter()
    share_counter = share_collector.generate_counter()
    msh_counter = msh_collector.generate_counter()

    # compare dataset intersections
    print("Intersection on MIMIC test: ")
    compare_dataset_summary(mimic_train_counter, mimic_test_counter)
    print("Intersection on share: ")
    compare_dataset_summary(mimic_train_counter, share_counter)
    print("Intersection on msh: ")
    compare_dataset_summary(mimic_train_counter, msh_counter)

    # compare mapping instances
    print("Compare instances...")
    mimic_test_overlap, mimic_test_all, mimic_test_has_abbr = compare_dataset_instances(mimic_train_counter, mimic_test_counter)
    print("mimic test (all: %d, has abbr no cui: %d, overlap: %d): %f" % (mimic_test_all, mimic_test_has_abbr, mimic_test_overlap, mimic_test_overlap/mimic_test_all))

    share_overlap, share_all, share_has_abbr = compare_dataset_instances(mimic_train_counter, share_counter)
    print("share (all: %d, has abbr no cui: %d, overlap: %d): %f" % (share_all, share_has_abbr, share_overlap, share_overlap / share_all))

    msh_overlap, msh_all, msh_has_abbr = compare_dataset_instances(mimic_train_counter, msh_counter)
    print("msh (all: %d, has abbr no cui: %d, overlap: %d): %f" % (msh_all, msh_has_abbr, msh_overlap, msh_overlap / msh_all))

    print()
