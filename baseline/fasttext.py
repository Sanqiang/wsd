"""
Train supervised WSD model by fastText.
"""
import os
from tqdm import tqdm
import random
from fastText import train_supervised, load_model
from baseline.word_embedding import AbbrIndex
from baseline.generate_content import AbbrCorpus, build_index_of_abbrs, Doc
from baseline.dataset_helper import DataSetPaths
from preprocess.file_helper import txt_reader, txt_writer, pickle_writer, pickle_reader


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

    # Save all content vectors to pickle files
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

    # Save all content vectors to pickle files
    content_dir = data_processed_path + '/dataset/'
    os.makedirs(content_dir, exist_ok=True)

    print("Building dataset for fastText...")
    print(len(abbr_index))

    for abbr in tqdm(abbr_index):
        abbr_job(abbr, abbr_index, abbr_idx_mapper, test_docs, content_dir, window_size)


def train_fasttext_classifier(train_processed_path, abbr=None):
    train_path = train_processed_path + '/fasttext'
    os.makedirs(train_path+'/model', exist_ok=True)
    if abbr is None:
        # train on whole dataset
        input_file = train_path+'/dataset/all.txt'
        model_file = train_path+'/model/all.bin'
    else:
        # Load abbr index
        abbr_idx_mapper = pickle_reader(train_path+'/abbr_idx_mapper.pkl')
        abbr_idx = abbr_idx_mapper['abbr2idx'][abbr]
        input_file = train_path+'/dataset/%d.txt' % abbr_idx
        model_file = train_path+'/model/%d.bin' % abbr_idx

    model = train_supervised(
        input=input_file,
        epoch=100,
        lr=0.1,
        lrUpdateRate=500,
        dim=100,
        ws=5,
        loss='hs',
        thread=60
    )
    model.save_model(model_file)
    return model


def eval_fasttext_classifier(train_processed_path, test_processed_path, abbr=None):
    if abbr is None:
        # evaluate on whole dataset
        eval_file = test_processed_path+'/fasttext/dataset/all.txt'
        model_file = train_processed_path+'/fasttext/model/all.bin'
    else:
        # Load abbr index
        train_abbr_idx_mapper = pickle_reader(train_processed_path+'/fasttext/abbr_idx_mapper.pkl')
        test_abbr_idx_mapper = pickle_reader(test_processed_path + '/fasttext/abbr_idx_mapper.pkl')

        train_abbr_idx = train_abbr_idx_mapper['abbr2idx'][abbr]
        test_abbr_idx = test_abbr_idx_mapper['abbr2idx'][abbr]

        eval_file = test_processed_path + '/fasttext/dataset/%d.txt' % test_abbr_idx
        model_file = train_processed_path + '/fasttext/model/%d.bin' % train_abbr_idx

    model = load_model(model_file)
    print(model.test(eval_file))


def generate_whole_dataset(processed_path):
    abbr_idx_mapper = pickle_reader(processed_path + '/fasttext/abbr_idx_mapper.pkl')
    with open(processed_path+'/fasttext/dataset/all.txt', 'w') as f:
        total_dataset = []
        for abbr, abbr_idx in tqdm(abbr_idx_mapper['abbr2idx'].items()):
            total_dataset.extend(txt_reader(processed_path+'/fasttext/dataset/%d.txt' % abbr_idx))
        random.shuffle(total_dataset)
        f.write("\n".join(total_dataset))


def train_svm(train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(train_processed_path+'/abbr_idx_mapper.pkl')
    abbr_cui2idx_inventory = {}
    os.makedirs(train_processed_path+'/svm_models/', exist_ok=True)
    # generate training data & train model
    for abbr, abbr_idx in tqdm(abbr_idx_mapper['abbr2idx'].items()):
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


if __name__ == '__main__':
    dataset_paths = DataSetPaths('luoz3_x1')

    # generate_train_data(dataset_paths.mimic_train_folder)
    #
    # generate_test_data(dataset_paths.mimic_test_folder)
    # generate_test_data(dataset_paths.share_test_folder)
    # generate_test_data(dataset_paths.upmc_example_folder)

    # generate_whole_dataset(dataset_paths.mimic_train_folder)
    # generate_whole_dataset(dataset_paths.mimic_test_folder)

    abbr = 'ct'
    train_fasttext_classifier(dataset_paths.mimic_train_folder)

    eval_fasttext_classifier(dataset_paths.mimic_train_folder, dataset_paths.mimic_test_folder)
