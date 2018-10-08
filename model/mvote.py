"""Simple majority vote impl"""
from collections import Counter, defaultdict
from preprocess.file_helper import pickle_writer, pickle_reader


PATH_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
PATH_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'

PATH_TRAIN_COLLECTION = '/home/luoz3/data/mimic/train_abbr_collection.pkl'
data_path = '/home/luoz3/data/'
msh_txt_path = data_path + 'msh/msh_processed/msh_processed.txt'
share_txt_path = data_path + 'share/processed/share_all_processed.txt'


# Collect abbr info on training set
def collect_train_abbr(train_path, save_collection_path=None):
    assign_collect = defaultdict(Counter)
    lid = 0
    for line in open(train_path):
        ents = [w for w in line.split() if w.startswith('abbr|')]
        for ent in ents:
            items = ent.split('|')
            abbr = items[1]
            cui = items[2]
            assign_collect[abbr].update([cui])
        lid += 1
        if lid % 10000 == 0:
            print('Processed %s lines' % lid)

    if save_collection_path is not None:
        pickle_writer(assign_collect, save_collection_path)
    return assign_collect


def test_majority_vote(train_collection, test_path):
    assign_map = {}
    correct_cnt, total_cnt = 0.0, 0.0
    for abbr in train_collection:
        assign_map[abbr] = train_collection[abbr].most_common(1)[0][0]

    for line in open(test_path):
        ents = [w for w in line.split() if w.startswith('abbr|')]
        for ent in ents:
            items = ent.split('|')
            abbr = items[1]
            cui = items[2]
            if abbr in assign_map:
                pred_cui = assign_map[abbr]
                if cui == pred_cui:
                    correct_cnt += 1.0
            total_cnt += 1.0

    acc = correct_cnt / total_cnt
    print('Accuray = %s' % acc)
    return acc


def build_inventory(train_collection):
    inventory = {}
    for abbr, items in train_collection.items():
        inventory[abbr] = list(items)
    return inventory


def compare_abbr_inventory(inventory1, inventory2):
    # collect CUIs
    cuis_1 = []
    for _, cuis in inventory1.items():
        cuis_1.extend(cuis)
    cuis_1 = set(cuis_1)
    print("No.CUIs on train: ", len(cuis_1))

    cuis_2 = []
    for _, cuis in inventory2.items():
        cuis_2.extend(cuis)
    cuis_2 = set(cuis_2)
    print("No.CUIs on test: ", len(cuis_2))

    # intersection
    print("CUI overlap ratio: ", len(cuis_1 & cuis_2)/len(cuis_2))

    # collect abbr info
    abbr_1 = set(inventory1.keys())
    print("No. abbrs on train: ", len(abbr_1))
    abbr_2 = set(inventory2.keys())
    print("No. abbrs on test: ", len(abbr_2))
    print("Abbr overlap ratio: ", len(abbr_1 & abbr_2)/len(abbr_2))


if __name__ == '__main__':

    # # process train abbr info
    # collect_train_abbr(PATH_TRAIN, PATH_TRAIN_COLLECTION)

    # test on test set
    assign_collect = pickle_reader(PATH_TRAIN_COLLECTION)
    print("Mvote on MIMIC test: ")
    test_majority_vote(assign_collect, PATH_EVAL)
    print("Mvote on share: ")
    test_majority_vote(assign_collect, share_txt_path)
    print("Mvote on msh: ")
    test_majority_vote(assign_collect, msh_txt_path)

    # intersections
    train_inventory = build_inventory(assign_collect)
    mimic_test_inventory = build_inventory(collect_train_abbr(PATH_EVAL))
    share_inventory = build_inventory(collect_train_abbr(share_txt_path))
    msh_inventory = build_inventory(collect_train_abbr(msh_txt_path))

    print("Intersection on MIMIC test: ")
    compare_abbr_inventory(train_inventory, mimic_test_inventory)
    print("Intersection on share: ")
    compare_abbr_inventory(train_inventory, share_inventory)
    print("Intersection on msh: ")
    compare_abbr_inventory(train_inventory, msh_inventory)
