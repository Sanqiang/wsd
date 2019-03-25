"""
Train supervised WSD model by fastText.
"""
import os
import random
import errno
from tqdm import tqdm
from fastText import train_supervised, load_model, train_unsupervised
from baseline.word_embedding import AbbrIndex
from baseline.generate_content import AbbrCorpus, build_index_of_abbrs, Doc
from baseline.dataset_helper import DataSetPaths, compare_dataset_instances
from baseline.dataset_helper import InstancePred, evaluation, AbbrInstanceCollector
from preprocess.file_helper import txt_reader, txt_writer, pickle_writer, pickle_reader


###############################################################
# generate datasets (txt files) to feed the fastText model
###############################################################

def abbr_job(abbr, abbr_index, abbr_idx_mapper, docs, content_dir, window_size):
    corpus = AbbrCorpus(abbr, abbr_index, docs, window_size=window_size)
    corpus_content = corpus.content_generator()

    dataset = []
    for global_instance_idx, doc_id, pos, content_pos, content, label in corpus_content:
        content.insert(content_pos, abbr)
        content = " ".join(content)
        dataset.append("__label__{} {}".format(label, content))

    # save vector to pickle file
    index = abbr_idx_mapper['abbr2idx'][abbr]
    txt_writer(dataset, content_dir+'%d.txt' % index)


def generate_train_data(train_processed_path, window_size=5):
    # Load abbr index
    abbr_index = AbbrIndex(train_processed_path + '/abbr_index_data.pkl')
    train_docs = Doc(txt_reader(train_processed_path + "/train_no_mark.txt"))

    data_processed_path = train_processed_path + '/fasttext'
    os.makedirs(data_processed_path, exist_ok=True)

    # Build index for abbrs (for saving pickle files)
    abbr_idx_mapper = build_index_of_abbrs(abbr_index)
    pickle_writer(abbr_idx_mapper, data_processed_path + '/abbr_idx_mapper.pkl')

    content_dir = data_processed_path + '/dataset/'
    os.makedirs(content_dir, exist_ok=True)

    print("Building dataset for fastText...")
    print(len(abbr_index))

    for abbr in tqdm(abbr_index):
        abbr_job(abbr, abbr_index, abbr_idx_mapper, train_docs, content_dir, window_size)


def generate_test_data(test_processed_path, window_size=5):
    # Load abbr index
    abbr_index = AbbrIndex(test_processed_path + '/abbr_index_data.pkl')
    test_docs = Doc(txt_reader(test_processed_path + "/test_no_mark.txt"))

    data_processed_path = test_processed_path + '/fasttext'
    os.makedirs(data_processed_path, exist_ok=True)

    # Build index for abbrs (for saving pickle files)
    abbr_idx_mapper = build_index_of_abbrs(abbr_index)
    pickle_writer(abbr_idx_mapper, data_processed_path + '/abbr_idx_mapper.pkl')

    content_dir = data_processed_path + '/dataset/'
    os.makedirs(content_dir, exist_ok=True)

    print("Building dataset for fastText...")
    print(len(abbr_index))

    for abbr in tqdm(abbr_index):
        abbr_job(abbr, abbr_index, abbr_idx_mapper, test_docs, content_dir, window_size)


def generate_whole_dataset(processed_path, shuffle=False):
    abbr_idx_mapper = pickle_reader(processed_path + '/fasttext/abbr_idx_mapper.pkl')
    with open(processed_path+'/fasttext/dataset/all.txt', 'w') as f:
        total_dataset = []
        for abbr, abbr_idx in tqdm(abbr_idx_mapper['abbr2idx'].items()):
            total_dataset.extend(txt_reader(processed_path+'/fasttext/dataset/%d.txt' % abbr_idx))
        if shuffle:
            random.shuffle(total_dataset)
        f.write("\n".join(total_dataset))


###############################################################
# Train word embedding by FastText
###############################################################

def train_skipgram(processed_path):
    print("Train fastText...")
    model = train_unsupervised(
        input=processed_path+"/train_no_mark.txt",
        model='skipgram',
        dim=100,
        minCount=5,
        ws=10,
        lrUpdateRate=1000,
        epoch=50,
        thread=60
    )
    model.save_model(processed_path+'/fasttext.bin')


def comvert_bin_to_vec(bin_file, vec_file):
    f = load_model(bin_file)
    words = f.get_words()
    with open(vec_file, 'w') as file:
        file.write(str(len(words)) + " " + str(f.get_dimension())+'\n')
        for w in tqdm(words):
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file.write(w + vstr+'\n')
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass


###############################################################
# Single fastText model
###############################################################

def train_fasttext_classifier(train_processed_path, abbr=None, use_pretrain=False, use_softmax=False):
    train_path = train_processed_path + '/fasttext'
    if use_pretrain:
        model_path = train_path + '/model/pre_train'
    else:
        model_path = train_path + '/model'
    train_path = train_processed_path + '/fasttext'
    os.makedirs(model_path, exist_ok=True)
    if abbr is None:
        # train on whole dataset
        input_file = train_path+'/dataset/all.txt'
        if use_softmax:
            model_file = model_path+'/all_softmax.bin'
        else:
            model_file = model_path+'/all.bin'
    else:
        # Load abbr index
        abbr_idx_mapper = pickle_reader(train_path+'/abbr_idx_mapper.pkl')
        abbr_idx = abbr_idx_mapper['abbr2idx'][abbr]
        input_file = train_path+'/dataset/%d.txt' % abbr_idx
        model_file = model_path+'/%d.bin' % abbr_idx

    if use_softmax:
        loss = 'softmax'
    else:
        loss = 'hs'
    model_config = {
        'input': input_file,
        'epoch': 50,
        'lr': 0.1,
        'lrUpdateRate': 100,
        'dim': 100,
        'ws': 5,
        'wordNgrams': 2,
        'loss': loss,
        'thread': 60,
    }
    if use_pretrain:
        model_config['pretrainedVectors'] = train_processed_path+'/fasttext.vec'
    model = train_supervised(**model_config)
    model.save_model(model_file)
    return model


def predict_fasttext_classifier(train_processed_path, test_processed_path, use_pretrain=False, use_softmax=False):
    train_path = train_processed_path + '/fasttext'
    if use_pretrain:
        model_path = train_path + '/model/pre_train'
    else:
        model_path = train_path + '/model'
    # Load abbr index
    train_abbr_idx_mapper = pickle_reader(train_path + '/abbr_idx_mapper.pkl')
    train_abbr2idx = train_abbr_idx_mapper['abbr2idx']
    test_abbr_idx_mapper = pickle_reader(test_processed_path + '/fasttext/abbr_idx_mapper.pkl')
    test_abbr_index = AbbrIndex(test_processed_path + '/abbr_index_data.pkl')

    # Load model
    if use_softmax:
        model_file = model_path + '/all_softmax.bin'
    else:
        model_file = model_path + '/all.bin'
    model = load_model(model_file)
    label_set = set(map(lambda x: x.lstrip("__label__"), model.get_labels()))

    instance_collection = []
    # generate testing data
    for abbr, test_abbr_idx in tqdm(test_abbr_idx_mapper['abbr2idx'].items()):
        # if abbr not in train_abbr2idx:
        #     for doc_id, pos_list in test_abbr_index[abbr].items():
        #         for global_instance_idx, pos, label in pos_list:
        #             instance_collection.append(InstancePred(index=global_instance_idx, abbr=abbr, sense_pred=None))
        # else:
        eval_abbr_instance_list = txt_reader(test_processed_path + '/fasttext/dataset/%d.txt' % test_abbr_idx)
        abbr_instance_idx = 0
        for doc_id, pos_list in test_abbr_index[abbr].items():
            for global_instance_idx, pos, label in pos_list:
                if label not in label_set:
                    instance_collection.append(InstancePred(index=global_instance_idx, abbr=abbr, sense_pred=None))
                else:
                    # get instance
                    tokens = eval_abbr_instance_list[abbr_instance_idx].split()
                    label_in_txt = tokens[0].lstrip("__label__")
                    assert label == label_in_txt
                    context = " ".join(tokens[1:])
                    instance_collection.append(InstancePred(
                        index=global_instance_idx, abbr=abbr,
                        sense_pred=model.predict(context)[0][0].lstrip("__label__")))
                abbr_instance_idx += 1
    # sort collection list based on global instance idx
    instance_collection = sorted(instance_collection, key=lambda x: x.index)
    return instance_collection


###############################################################
# Multiple fastText models (one model per abbr)
###############################################################

def train_fasttext_classifier_multi_model(train_processed_path, use_pretrain=False, use_softmax=False):
    train_path = train_processed_path + '/fasttext'
    if use_pretrain:
        model_path = train_path + '/model/pre_train'
    else:
        model_path = train_path + '/model'
    if use_softmax:
        loss = 'softmax'
    else:
        loss = 'hs'
    os.makedirs(model_path, exist_ok=True)
    # Load abbr index
    abbr_idx_mapper = pickle_reader(train_path+'/abbr_idx_mapper.pkl')
    abbr_index = AbbrIndex(train_processed_path + '/abbr_index_data.pkl')
    abbr_label_set = {}
    # Load training data & train model
    for abbr, abbr_idx in tqdm(abbr_idx_mapper['abbr2idx'].items()):
        input_file = train_path + '/dataset/%d.txt' % abbr_idx
        model_file = model_path + '/%d.bin' % abbr_idx
        # load label list
        label_set = set()
        for doc_id, pos_list in abbr_index[abbr].items():
            for global_instance_idx, pos, label in pos_list:
                label_set.add(label)
        abbr_label_set[abbr] = label_set
        # no need to train if only have 1 CUI
        if len(label_set) > 1:
            model_config = {
                'input': input_file,
                'epoch': 50,
                'lr': 0.1,
                'lrUpdateRate': 100,
                'dim': 100,
                'ws': 5,
                'wordNgrams': 2,
                'loss': loss,
                'thread': 60,
            }
            if use_pretrain:
                model_config['pretrainedVectors'] = train_processed_path + '/fasttext.vec'
            model = train_supervised(**model_config)
            model.save_model(model_file)
    pickle_writer(abbr_label_set, train_path+'/abbr_label_set.pkl')


def predict_fasttext_classifier_multi_model(train_processed_path, test_processed_path, use_pretrain=False):
    train_path = train_processed_path + '/fasttext'
    if use_pretrain:
        model_path = train_path + '/model/pre_train'
    else:
        model_path = train_path + '/model'
    # Load abbr index
    test_abbr_idx_mapper = pickle_reader(test_processed_path + '/fasttext/abbr_idx_mapper.pkl')
    test_abbr_index = AbbrIndex(test_processed_path + '/abbr_index_data.pkl')
    train_abbr_idx_mapper = pickle_reader(train_processed_path + '/fasttext/abbr_idx_mapper.pkl')
    train_abbr2idx = train_abbr_idx_mapper['abbr2idx']
    train_abbr_label_set = pickle_reader(train_processed_path + '/fasttext/abbr_label_set.pkl')

    instance_collection = []
    # generate testing data
    for abbr, test_abbr_idx in tqdm(test_abbr_idx_mapper['abbr2idx'].items()):
        if abbr not in train_abbr_label_set:
            for doc_id, pos_list in test_abbr_index[abbr].items():
                for global_instance_idx, pos, label in pos_list:
                    instance_collection.append(InstancePred(index=global_instance_idx, abbr=abbr, sense_pred=None))
        else:
            train_label_set = train_abbr_label_set[abbr]
            eval_abbr_instance_list = txt_reader(test_processed_path + '/fasttext/dataset/%d.txt' % test_abbr_idx)

            abbr_instance_idx = 0
            context_list, global_idx_list = [], []
            for doc_id, pos_list in test_abbr_index[abbr].items():
                for global_instance_idx, pos, label in pos_list:
                    # if true label not in train collection
                    if label not in train_label_set:
                        instance_collection.append(InstancePred(index=global_instance_idx, abbr=abbr, sense_pred=None))
                    # if only have 1 CUI
                    elif len(train_label_set) == 1:
                        instance_collection.append(InstancePred(index=global_instance_idx, abbr=abbr, sense_pred=label))
                    # need predict
                    else:
                        # get instance
                        tokens = eval_abbr_instance_list[abbr_instance_idx].split()
                        label_in_txt = tokens[0].lstrip("__label__")
                        assert label == label_in_txt
                        context = " ".join(tokens[1:])
                        context_list.append(context)
                        global_idx_list.append(global_instance_idx)
                    abbr_instance_idx += 1
            # predict
            if len(context_list) > 0:
                # Load model
                model_file = model_path + '/%d.bin' % train_abbr2idx[abbr]
                model = load_model(model_file)
                predict_list = model.predict(context_list)[0]
                for idx, predict in zip(global_idx_list, predict_list):
                    instance_collection.append(InstancePred(index=idx, abbr=abbr, sense_pred=predict[0].lstrip("__label__")))

    # sort collection list based on global instance idx
    instance_collection = sorted(instance_collection, key=lambda x: x.index)
    return instance_collection


def train_evaluate_fasttext_on_datasets(dataset_paths, only_test=True, use_single_model=True, use_pretrain=False, use_softmax=False):
    # train
    # train_dataset = ("MIMIC Train", dataset_paths.mimic_train_folder)
    train_dataset = ("UPMC AB Train", dataset_paths.upmc_ab_train_folder)

    if not only_test:
        print("Train fastText on {}:".format(train_dataset[0]))
        if use_single_model:
            train_fasttext_classifier(train_dataset[1], use_pretrain=use_pretrain, use_softmax=use_softmax)
        else:
            train_fasttext_classifier_multi_model(train_dataset[1], use_pretrain=use_pretrain, use_softmax=use_softmax)
    # test
    datasets = [
        # ("MIMIC Test", dataset_paths.mimic_eval_txt, dataset_paths.mimic_test_folder),
        # ("ShARe/CLEF", dataset_paths.share_txt, dataset_paths.share_test_folder),
        # ("MSH", dataset_paths.msh_txt, dataset_paths.msh_test_folder),
        # ("UPMC example", dataset_paths.upmc_example_txt, dataset_paths.upmc_example_folder),
        ("UPMC AB test", dataset_paths.upmc_ab_test_txt, dataset_paths.upmc_ab_test_folder),
    ]
    for name, txt_path, test_folder in datasets:
        print("Test fastText on {}: ".format(name))
        test_collector = AbbrInstanceCollector(txt_path)
        test_collection_true = test_collector.generate_instance_collection()
        if use_single_model:
            test_collection_pred = predict_fasttext_classifier(train_dataset[1], test_folder, use_pretrain=use_pretrain, use_softmax=use_softmax)
        else:
            test_collection_pred = predict_fasttext_classifier_multi_model(train_dataset[1], test_folder, use_pretrain=use_pretrain)
        print(evaluation(test_collection_true, test_collection_pred))


if __name__ == '__main__':
    dataset_paths = DataSetPaths('luoz3_x1')

    #####################################
    # generate train & test data
    #####################################

    # generate_train_data(dataset_paths.mimic_train_folder)
    # generate_train_data(dataset_paths.upmc_ab_train_folder)

    # generate_test_data(dataset_paths.mimic_test_folder)
    # generate_test_data(dataset_paths.share_test_folder)
    # generate_test_data(dataset_paths.upmc_example_folder)
    # generate_test_data(dataset_paths.msh_test_folder)
    # generate_test_data(dataset_paths.upmc_ab_test_folder)

    # generate_whole_dataset(dataset_paths.mimic_train_folder, shuffle=True)
    # generate_whole_dataset(dataset_paths.mimic_test_folder)
    # generate_whole_dataset(dataset_paths.share_test_folder)
    # generate_whole_dataset(dataset_paths.msh_test_folder)
    # generate_whole_dataset(dataset_paths.upmc_example_folder)
    # generate_whole_dataset(dataset_paths.upmc_ab_train_folder, shuffle=True)
    # generate_whole_dataset(dataset_paths.upmc_ab_test_folder)

    #####################################
    # train word embedding
    #####################################

    # train_skipgram(dataset_paths.mimic_train_folder)
    # comvert_bin_to_vec(
    #     dataset_paths.mimic_train_folder+'/fasttext.bin',
    #     dataset_paths.mimic_train_folder+'/fasttext.vec'
    # )
    # train_skipgram(dataset_paths.upmc_all_no_mark_folder)
    # comvert_bin_to_vec(
    #     dataset_paths.upmc_all_no_mark_folder+'/fasttext.bin',
    #     dataset_paths.upmc_ab_train_folder+'/fasttext.vec'
    # )

    # #####################################
    # # train & test (single model)
    # #####################################

    # train_evaluate_fasttext_on_datasets(
    #     dataset_paths,
    #     only_test=False,
    #     use_single_model=True,
    #     use_pretrain=True,
    #     use_softmax=True
    # )

    #####################################
    # train & test (multiple model)
    #####################################

    train_evaluate_fasttext_on_datasets(
        dataset_paths,
        only_test=True,
        use_single_model=True,
        use_pretrain=True,
        use_softmax=True
    )
