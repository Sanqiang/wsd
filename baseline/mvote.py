"""Simple majority vote impl"""
import tqdm
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


def compare_abbr_inventory(train_inventory, test_inventory):
    # collect CUIs
    cuis_1 = []
    for _, cuis in train_inventory.items():
        cuis_1.extend(cuis)
    cuis_1 = set(cuis_1)
    print("No.CUIs on train: ", len(cuis_1))

    cuis_2 = []
    for _, cuis in test_inventory.items():
        cuis_2.extend(cuis)
    cuis_2 = set(cuis_2)
    print("No.CUIs on test: ", len(cuis_2))

    # intersection
    print("CUI overlap ratio: ", len(cuis_1 & cuis_2)/len(cuis_2))

    # collect abbr info
    abbr_1 = set(train_inventory.keys())
    print("No. abbrs on train: ", len(abbr_1))
    abbr_2 = set(test_inventory.keys())
    print("No. abbrs on test: ", len(abbr_2))
    print("Abbr overlap ratio: ", len(abbr_1 & abbr_2)/len(abbr_2))


def compare_mapping_instances(train_inventory_counter, test_inventory_counter):
    train_inventory = build_inventory(train_inventory_counter)
    count_all_instances = 0
    count_no_abbr_instances = 0
    count_overlap_instances = 0
    for abbr, items in tqdm.tqdm(test_inventory_counter.items()):
        for cui, count in items.items():
            count_all_instances += count
            if abbr not in train_inventory:
                count_no_abbr_instances += count
            elif cui in train_inventory[abbr]:
                count_overlap_instances += count
    return count_overlap_instances, count_all_instances, count_all_instances-count_no_abbr_instances


if __name__ == '__main__':

    # # process train abbr info
    # collect_train_abbr(PATH_TRAIN, PATH_TRAIN_COLLECTION)

    # test on test set
    assign_collect = pickle_reader(PATH_TRAIN_COLLECTION)

    # count number of instances for training dataset
    count_all_instances = 0
    for abbr, items in tqdm.tqdm(assign_collect.items()):
        for cui, count in items.items():
            count_all_instances += count
    print(count_all_instances)

    # print("Mvote on MIMIC test: ")
    # test_majority_vote(assign_collect, PATH_EVAL)
    # print("Mvote on share: ")
    # test_majority_vote(assign_collect, share_txt_path)
    # print("Mvote on msh: ")
    # test_majority_vote(assign_collect, msh_txt_path)

    # # intersections
    # train_inventory = build_inventory(assign_collect)
    # mimic_test_inventory = build_inventory(collect_train_abbr(PATH_EVAL))
    # share_inventory = build_inventory(collect_train_abbr(share_txt_path))
    # msh_inventory = build_inventory(collect_train_abbr(msh_txt_path))
    #
    # print("Intersection on MIMIC test: ")
    # compare_abbr_inventory(train_inventory, mimic_test_inventory)
    # print("Intersection on share: ")
    # compare_abbr_inventory(train_inventory, share_inventory)
    # print("Intersection on msh: ")
    # compare_abbr_inventory(train_inventory, msh_inventory)


    # compare mapping instances
    mimic_test_inventory = collect_train_abbr(PATH_EVAL)
    share_inventory = collect_train_abbr(share_txt_path)
    msh_inventory = collect_train_abbr(msh_txt_path)

    mimic_test_overlap, mimic_test_all, mimic_test_has_abbr = compare_mapping_instances(assign_collect, mimic_test_inventory)
    print("mimic test (all: %d, has abbr: %d, overlap: %d): %f" % (mimic_test_all, mimic_test_has_abbr, mimic_test_overlap, mimic_test_overlap/mimic_test_all))

    share_overlap, share_all, share_has_abbr = compare_mapping_instances(assign_collect, share_inventory)
    print("share (all: %d, has abbr: %d, overlap: %d): %f" % (share_all, share_has_abbr, share_overlap, share_overlap / share_all))

    msh_overlap, msh_all, msh_has_abbr = compare_mapping_instances(assign_collect, msh_inventory)
    print("msh (all: %d, has abbr: %d, overlap: %d): %f" % (msh_all, msh_has_abbr, msh_overlap, msh_overlap / msh_all))
