"""
Train and test SVM baseline.
"""

import os
import numpy as np
import tqdm
import random
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from preprocess.file_helper import pickle_reader, pickle_writer
from baseline.word_embedding import AbbrIndex


def train_sample(x, y, sample_size, random_seed=None):
    """
    Sample training data. Maximum sample_size per class.

    :param x:
    :param y:
    :param sample_size:
    :param random_seed:
    :return:
    """
    random.seed(random_seed)
    # collect label pos
    pos_map = defaultdict(list)
    for i, label in enumerate(y):
        pos_map[label].append(i)
    # sample for each class
    sample_pos = []
    for _, positions in pos_map.items():
        if len(positions) <= sample_size:
            sample_pos.extend(positions)
        else:
            sample_pos.extend(random.sample(positions, sample_size))
    # generate samples
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, (data, target) in enumerate(zip(x, y)):
        if i in sample_pos:
            x_train.append(data)
            y_train.append(target)

    return np.vstack(x_train), y_train


def svm_cross_validation(train_processed_path, abbr='pe'):
    """
    Tuning the parameters on largest abbr in train set.

    :param train_processed_path:
    :return:
    """
    content_vector = pickle_reader(train_processed_path + '/content_vectors/%s_vector.pkl' % abbr)
    label2idx = {}
    label_idx = 0
    x = []
    y = []
    for instance_id, doc_id, pos, content_pos, content_vec, content, label in content_vector:
        if label not in label2idx:
            label2idx[label] = label_idx
            label_idx += 1
        x.append(content_vec)
        y.append(label2idx[label])

    x_train, y_train = train_sample(x, y, 500)
    parameters = {'gamma': [1e-4, 1e-3, 1e-2],
                  'C': [1e-1, 1, 10, 100, 1000]}
    model = SVC(kernel='rbf')
    model_cv = GridSearchCV(model, parameters, cv=5).fit(x_train, y_train)
    print(model_cv.best_params_)
    print(model_cv.best_score_)
    return model_cv


def train_svm(train_processed_path):
    # Load abbr index
    abbr_index = AbbrIndex(train_processed_path+'/abbr_index_data.pkl')
    abbr2idx_inventory = {}
    os.makedirs(train_processed_path+'/svm_models/', exist_ok=True)
    # generate training data & train model
    for abbr in tqdm.tqdm(abbr_index):
        if "/" not in abbr:
            content_vector = pickle_reader(train_processed_path + '/content_vectors/%s_vector.pkl' % abbr)
            label2idx = {}
            label_idx = 0
            x = []
            y = []
            for instance_id, doc_id, pos, content_pos, content_vec, content, label in content_vector:
                if label not in label2idx:
                    label2idx[label] = label_idx
                    label_idx += 1
                x.append(content_vec)
                y.append(label2idx[label])

            abbr2idx_inventory[abbr] = label2idx
            # no need to train if only have 1 CUI
            if len(label2idx) > 1:
                x_train, y_train = train_sample(x, y, 2000)
                # train svm model
                model = SVC(kernel='rbf', gamma=0.01, C=1000).fit(x_train, y_train)
                pickle_writer(model, train_processed_path + '/svm_models/%s.pkl' % abbr)
    pickle_writer(abbr2idx_inventory, train_processed_path+'/abbr_cui_idx_inventory.pkl')


def test_svm(test_processed_path, train_processed_path):
    # Load abbr index
    abbr_index = AbbrIndex(test_processed_path+'/abbr_index_data.pkl')
    abbr2idx_inventory = pickle_reader(train_processed_path+'/abbr_cui_idx_inventory.pkl')

    TP = 0
    all = 0
    all_no_label = 0
    # generate testing data
    for abbr in tqdm.tqdm(abbr_index):
        if "/" not in abbr:
            content_vector = pickle_reader(test_processed_path + '/content_vectors/%s_vector.pkl' % abbr)
            if abbr not in abbr2idx_inventory:
                all += len(content_vector)
                all_no_label += len(content_vector)
            else:
                label2idx = abbr2idx_inventory[abbr]
                all += len(content_vector)
                count_no_label_instances = 0
                x = []
                y = []
                for instance_id, doc_id, pos, content_pos, content_vec, content, label in content_vector:
                    if label not in label2idx:
                        count_no_label_instances += 1
                    else:
                        x.append(content_vec)
                        y.append(label2idx[label])
                all_no_label += count_no_label_instances
                # if only have 1 CUI
                if len(label2idx) == 1:
                    TP += len(y)
                elif y==[]:
                    continue
                else:
                    model = pickle_reader(train_processed_path + '/svm_models/%s.pkl' % abbr)
                    y_pred = model.predict(np.vstack(x))
                    TP += sum(y == y_pred)

    print("Accuracy: ", TP/all)
    print("Num.instances: ", all)
    print("Num.no labels: ", all_no_label)


if __name__ == '__main__':
    train_processed_path = '/home/luoz3/data/mimic/processed/train/'

    mimic_test_path = '/home/luoz3/data/mimic/processed/test/'
    data_path = '/home/luoz3/data/'
    msh_test_path = data_path + 'msh/msh_processed/test/'
    share_test_path = data_path + 'share/processed/test/'

    # svm_cross_validation(train_processed_path)

    train_svm(train_processed_path)

    test_svm(mimic_test_path, train_processed_path)
    test_svm(msh_test_path, train_processed_path)
    test_svm(share_test_path, train_processed_path)

    # abbr2idx_inventory = pickle_reader(train_processed_path + '/abbr_cui_idx_inventory.pkl')

    print()
