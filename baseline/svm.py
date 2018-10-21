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
from baseline.dataset_helper import DataSetPaths, InstancePred, AbbrInstanceCollector, evaluation


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


def svm_cross_validation(train_processed_path, abbr_idx=0):
    """
    Tuning the parameters on largest abbr in train set.

    :param train_processed_path:
    :param abbr_idx:
    :return:
    """
    content_vector = pickle_reader(train_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
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
    abbr_idx_mapper = pickle_reader(train_processed_path+'/abbr_idx_mapper.pkl')
    abbr_cui2idx_inventory = {}
    os.makedirs(train_processed_path+'/svm_models/', exist_ok=True)
    # generate training data & train model
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(train_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        label2idx = {}
        label_idx = 0
        x = []
        y = []
        for global_instance_idx, doc_id, pos, content_pos, content_vec, content, label in content_vector:
            if label not in label2idx:
                label2idx[label] = label_idx
                label_idx += 1
            x.append(content_vec)
            y.append(label2idx[label])

        abbr_cui2idx_inventory[abbr] = label2idx
        # no need to train if only have 1 CUI
        if len(label2idx) > 1:
            x_train, y_train = train_sample(x, y, 2000)
            # train svm model
            model = SVC(kernel='rbf', gamma=0.01, C=100).fit(x_train, y_train)
            pickle_writer(model, train_processed_path + '/svm_models/%d_svm.pkl' % abbr_idx)
    pickle_writer(abbr_cui2idx_inventory, train_processed_path+'/abbr_cui_idx_inventory.pkl')


def test_svm(test_processed_path, train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(test_processed_path + '/abbr_idx_mapper.pkl')
    abbr_idx_mapper_train = pickle_reader(train_processed_path + '/abbr_idx_mapper.pkl')
    abbr2train_idx = abbr_idx_mapper_train['abbr2idx']
    abbr_cui2idx_inventory = pickle_reader(train_processed_path+'/abbr_cui_idx_inventory.pkl')

    TP = 0
    all = 0
    all_no_label = 0
    # generate testing data
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(test_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        if abbr not in abbr_cui2idx_inventory:
            all += len(content_vector)
            all_no_label += len(content_vector)
        else:
            label2idx = abbr_cui2idx_inventory[abbr]
            all += len(content_vector)
            count_no_label_instances = 0
            x = []
            y = []
            for global_instance_idx, doc_id, pos, content_pos, content_vec, content, label in content_vector:
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
                model = pickle_reader(train_processed_path + '/svm_models/%d_svm.pkl' % abbr2train_idx[abbr])
                y_pred = model.predict(np.vstack(x))
                TP += sum(y == y_pred)

    print("Accuracy: ", TP/all)
    print("Num.instances: ", all)
    print("Num.no labels: ", all_no_label)


def predict_svm(test_processed_path, train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(test_processed_path + '/abbr_idx_mapper.pkl')
    abbr_idx_mapper_train = pickle_reader(train_processed_path + '/abbr_idx_mapper.pkl')
    abbr2train_idx = abbr_idx_mapper_train['abbr2idx']
    abbr_cui2idx_inventory = pickle_reader(train_processed_path+'/abbr_cui_idx_inventory.pkl')

    instance_collection = []
    # generate testing data
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(test_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        if abbr not in abbr_cui2idx_inventory:
            for global_instance_idx, _, _, _, _, _, _ in content_vector:
                instance_collection.append(InstancePred(
                    index=global_instance_idx,
                    abbr=abbr,
                    sense_pred=None
                ))
        else:
            label2idx = abbr_cui2idx_inventory[abbr]
            x = []
            y = []
            global_idx_list = []
            for global_instance_idx, _, _, _, content_vec, _, label in content_vector:
                # if true label not in train collection
                if label not in label2idx:
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred='CUI-not-found'
                    ))
                # if only have 1 CUI
                elif len(label2idx) == 1:
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred=label
                    ))
                else:
                    x.append(content_vec)
                    y.append(label2idx[label])
                    global_idx_list.append(global_instance_idx)

            if len(y) > 0:
                model = pickle_reader(train_processed_path + '/svm_models/%d_svm.pkl' % abbr2train_idx[abbr])
                y_pred = model.predict(np.vstack(x))

                # get idx2label
                idx2label = {}
                for label, idx in label2idx.items():
                    idx2label[idx] = label

                for idx_pred, global_instance_idx in zip(y_pred, global_idx_list):
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred=idx2label[idx_pred]
                    ))

    # sort collection list based on global instance idx
    instance_collection = sorted(instance_collection, key=lambda x: x.index)
    return instance_collection


if __name__ == '__main__':
    dataset_paths = DataSetPaths()

    #####################################
    # train
    #####################################

    # svm_cross_validation(dataset_paths.mimic_train_folder, abbr_idx=40)

    train_svm(dataset_paths.mimic_train_folder)

    #####################################
    # testing (deprecated, but faster)
    #####################################
    # test_svm(dataset_paths.mimic_test_folder, dataset_paths.mimic_train_folder)
    # test_svm(dataset_paths.share_test_folder, dataset_paths.mimic_train_folder)
    # test_svm(dataset_paths.msh_test_folder, dataset_paths.mimic_train_folder)

    #####################################
    # testing (using standard evaluation pipeline)
    #####################################

    # load test sets
    mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_eval_txt)
    share_collector = AbbrInstanceCollector(dataset_paths.share_txt)
    msh_collector = AbbrInstanceCollector(dataset_paths.msh_txt)

    print("SVM on MIMIC test: ")
    mimic_test_collection_true = mimic_test_collector.generate_instance_collection()
    mimic_test_collection_pred = predict_svm(dataset_paths.mimic_test_folder, dataset_paths.mimic_train_folder)
    evaluation(mimic_test_collection_true, mimic_test_collection_pred)

    print("SVM on ShARe/CLEF: ")
    share_collection_true = share_collector.generate_instance_collection()
    share_collection_pred = predict_svm(dataset_paths.share_test_folder, dataset_paths.mimic_train_folder)
    evaluation(mimic_test_collection_true, mimic_test_collection_pred)

    print("SVM on MSH: ")
    msh_collection_true = msh_collector.generate_instance_collection()
    msh_collection_pred = predict_svm(dataset_paths.msh_test_folder, dataset_paths.mimic_train_folder)
    evaluation(msh_collection_true, msh_collection_pred)
