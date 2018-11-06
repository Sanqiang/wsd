"""Simple majority vote impl"""
from preprocess.file_helper import pickle_writer, pickle_reader
from baseline.dataset_helper import DataSetPaths, AbbrInstanceCollector, process_abbr_token, evaluation, InstancePred


def predict_majority_vote(train_counter, test_path):
    """
    Make prediction using majority vote, return list of InstancePred.

    :param train_counter:
    :param test_path:
    :return: list of InstancePred
    """
    assign_map = {}
    for abbr in train_counter:
        assign_map[abbr] = train_counter[abbr].most_common(1)[0][0]

    instance_collection = []
    idx = 0
    for line in open(test_path):
        for token in line.rstrip('\n').split(" "):
            items = process_abbr_token(token)
            if items is not None:
                abbr, _, _ = items
                if abbr in assign_map:
                    sense_pred = assign_map[abbr]
                else:
                    sense_pred = None
                instance_collection.append(InstancePred(
                    index=idx,
                    abbr=abbr,
                    sense_pred=sense_pred))
                idx += 1
    return instance_collection


def evaluate_score_majority_vote(train_counter, test_path):
    assign_map = {}
    correct_cnt, total_cnt = 0.0, 0.0
    for abbr in train_counter:
        assign_map[abbr] = train_counter[abbr].most_common(1)[0][0]

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
    print()
    return acc


if __name__ == '__main__':
    dataset_paths = DataSetPaths('luoz3')
    train_counter_path = dataset_paths.mimic_train_folder+'train_abbr_counter.pkl'

    # # process train abbr info
    # mimic_train_collector = AbbrInstanceCollector(dataset_paths.mimic_train_txt)
    # mimic_train_counter = mimic_train_collector.generate_counter(train_counter_path)

    mimic_train_counter = pickle_reader(train_counter_path)

    #####################################
    # testing (directly compute score, not using standard pipeline)
    #####################################
    # print("Mvote on MIMIC test: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.mimic_eval_txt)
    # print("Mvote on ShARe/CLEF: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.share_txt)
    # print("Mvote on MSH: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.msh_txt)

    #####################################
    # testing (using standard evaluation pipeline)
    #####################################

    # load test sets
    mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_eval_txt)
    share_collector = AbbrInstanceCollector(dataset_paths.share_txt)
    msh_collector = AbbrInstanceCollector(dataset_paths.msh_txt)

    print("Mvote on MIMIC test: ")
    mimic_test_collection_true = mimic_test_collector.generate_instance_collection()
    mimic_test_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.mimic_eval_txt)
    evaluation(mimic_test_collection_true, mimic_test_collection_pred)

    print("Mvote on ShARe/CLEF: ")
    share_collection_true = share_collector.generate_instance_collection()
    share_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.share_txt)
    evaluation(share_collection_true, share_collection_pred)

    print("Mvote on MSH: ")
    msh_collection_true = msh_collector.generate_instance_collection()
    msh_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.msh_txt)
    evaluation(msh_collection_true, msh_collection_pred)
