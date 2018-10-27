"""
Test on test datasets.
"""
import os

from tensorflow.keras.utils import Progbar

from data_generator.data import TrainData, EvalData
from model.graph import Graph
from data_generator.data import get_feed, get_session_config
from util.checkpoint import copy_ckpt_to_modeldir

import tensorflow as tf
import numpy as np

import time
import glob
from datetime import datetime
from os.path import exists, join
from os import makedirs, listdir, remove

from model.model_config import get_args, BaseConfig
from baseline.dataset_helper import DataSetPaths, InstancePred, AbbrInstanceCollector, evaluation


def evaluate_and_write_to_disk(sess, graph, train_dataloader, output_file_path,
                               epoch=0, step=0, loss=0.0, perplexity=0.0):
    instance_collections = evaluate_on_testsets(sess, graph, train_dataloader)

    score_names_to_report = ['accuracy', 'accuracy_capable']
    with open(output_file_path, 'a') as score_csv_file:
        # write header
        if os.path.getsize(output_file_path) == 0:
            line = ','.join([str(i) for i in ['epoch', 'step', 'loss', 'perplexity']])
            for testset_name, (_, _, score_dict) in instance_collections.items():
                for score_name in score_names_to_report:
                    line += ',%s_%s' % (testset_name, score_name)
            line += '\n'
            score_csv_file.write(line)

        line = ','.join([str(i) for i in [epoch, step, loss, perplexity]])
        for testset_name, (_, _, score_dict) in instance_collections.items():
            for score_name in score_names_to_report:
                line += ',%.6f' % (score_dict[score_name])
        line += '\n'


def predict_from_model(sess, graph, test_dataloader, data_config):
    instance_collection_pred = []

    progbar = Progbar(target=test_dataloader.size)

    while True:
        input_feed, excluded_count, targets = get_feed(graph.data_feeds, test_dataloader, data_config, False)
        fetches = [graph.data_feeds[0]['pred'], graph.loss, graph.global_step,
                   graph.perplexity, graph.losses_eval]
        preds, loss, _, perplexity, losses_eval = sess.run(fetches, input_feed)

        progbar.update(current=targets[-1]['line_id'],
                       values=[('loss', loss), ('ppl', perplexity)])

        for example_id in range(data_config.batch_size - excluded_count):
            gt_target = targets[example_id]
            pred = preds[example_id]
            instance_collection_pred.append(
                InstancePred(
                    index=gt_target['inst_id'],
                    abbr=gt_target['abbr'],
                    sense_pred=test_dataloader.id2sense[pred[0]] if gt_target['abbr_id'] else None
                )
            )

        if excluded_count > 0:
            break

    # sort collection list by the global instance idx
    instance_collection_pred = sorted(instance_collection_pred, key=lambda x: x.index)

    '''
    # some instances might have been skipped, thus we add non-included instances before returning
    # Rui: not necessary now as no data is skipped
    instance_collection = []
    temp_idx = 0
    for idx in range(len(test_true)):
        temp_instance_pred = instance_collection_pred[temp_idx]
        if temp_instance_pred.index == idx:
            instance_collection.append(temp_instance_pred)
            temp_idx += 1
        else:
            instance_collection.append(InstancePred(index=idx, abbr=None, sense_pred=None))
    '''

    return instance_collection_pred


def evaluate_on_testsets(sess, graph, train_data):
    test_instance_collections = {}

    for test_dataset_name in ['msh', 'share', 'mimic']:
        print('Evaluating on %s' % test_dataset_name)

        if test_dataset_name == 'share':
            test_filepath = dataset_paths.share_txt
        elif test_dataset_name == 'msh':
            test_filepath = dataset_paths.msh_txt
        elif test_dataset_name == 'mimic':
            test_filepath = dataset_paths.mimic_eval_txt
        else:
            raise ValueError('Please type valid dataset name')

        # test file path
        data_config = BaseConfig()
        setattr(data_config, 'eval_file', test_filepath)

        test_collector = AbbrInstanceCollector(test_filepath)
        test_true_collection = test_collector.generate_instance_collection()
        test_dataloader = EvalData(train_data, data_config, dataset_name=test_dataset_name)

        test_pred_collection = predict_from_model(sess, graph, test_dataloader, data_config)

        score_dict = evaluation(test_true_collection, test_pred_collection)

        print('Performance of testset %s:' % test_dataset_name)
        for key, value in score_dict.items():
            print('\t%s = %.6f' % (key, value))

        test_instance_collections[test_dataset_name] = (test_true_collection, test_pred_collection, score_dict)

    return test_instance_collections


def predict_from_checkpoint(model_config, ckpt):
    """
    Make prediction using transformer, return list of InstancePred.

    :param model_config:
    :param ckpt:
    :return:
    """
    # Eval only uses single GPU
    assert model_config.num_gpus == 1

    train_data = TrainData(model_config)
    graph = Graph(False, model_config, train_data)
    tf.reset_default_graph()
    graph.create_model_multigpu()
    sess = tf.train.MonitoredTrainingSession(
        config=get_session_config()
    )
    graph.saver.restore(sess, ckpt)

    # instance_collections = evaluate_on_testsets(sess, graph, train_data)
    evaluate_and_write_to_disk(sess, graph, train_data, output_file_path=model_config.logdir)


def eval(model_config, ckpt):
    assert model_config.num_gpus == 1 # Eval only uses single GPU

    data = EvalData(model_config)
    graph = Graph(False, model_config, data)
    tf.reset_default_graph()
    graph.create_model_multigpu()
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_config.logdir,
        config=get_session_config()
    )
    graph.saver.restore(sess, ckpt)

    perplexitys = []
    total_cnt = 0.0
    correct_cnt, correct_cnt2, correct_cnt3, correct_cnt4, correct_cnt5 = 0.0, 0.0, 0.0, 0.0, 0.0
    report = []
    start_time = datetime.now()

    while True:
        input_feed, exclude_cnt, gt_targets = get_feed(graph.data_feeds, data, model_config, False)
        fetches = [graph.data_feeds[0]['pred'], graph.loss, graph.global_step,
                   graph.perplexity, graph.losses_eval]
        preds, loss, step, perplexity, losses_eval = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        for batch_id in range(model_config.batch_size - exclude_cnt):
            gt_target = gt_targets[batch_id]
            pred = preds[batch_id]

            if gt_target[2] == pred[0:1] :
                correct_cnt += 1
            if gt_target[2] in pred[0:2]:
                correct_cnt2 += 1
            if gt_target[2] in pred[0:3]:
                correct_cnt3 += 1
            if gt_target[2] in pred[0:4]:
                correct_cnt4 += 1
            if gt_target[2] in pred[0:5]:
                correct_cnt5 += 1
            total_cnt += 1

            abbr_id = gt_target[1]

            report.append('Abbr:%s\tPred:%s\tGt:%s\tline:%s\tinst id:%s with step %s with loss %s.' %
                          (data.id2abbr[abbr_id],
                           ';'.join([data.id2sense[loop] for loop in pred]),
                           data.id2sense[gt_target[2]], gt_target[3], gt_target[0], gt_target[4],
                           losses_eval[batch_id]))
            report.append('')

        if exclude_cnt > 0:
            break

    end_time = datetime.now()
    fmt = "%.5f"
    acc = fmt % (correct_cnt / total_cnt)
    acc2 = fmt % (correct_cnt2 / total_cnt)
    acc3 = fmt % (correct_cnt3 / total_cnt)
    acc4 = fmt % (correct_cnt4 / total_cnt)
    acc5 = fmt % (correct_cnt5 / total_cnt)
    perplexity = np.mean(perplexity)
    report = '\n'.join(report)
    filename = 'step%s_acc%s_acc2%s_acc3%s_acc4%s_acc5%s_pp%s.txt' % (step, acc, acc2, acc3, acc4, acc5, perplexity)
    span = end_time - start_time

    if not exists(model_config.resultdir):
        makedirs(model_config.resultdir)
    f = open(model_config.resultdir + filename, 'w')
    f.write(report)
    f.close()
    print('Eval Finished using %s.' % str(span))
    return float(acc)


def get_best_acc(model_config):
    if not exists(model_config.resultdir):
        makedirs(model_config.resultdir)
    best_acc_file = join(model_config.resultdir, 'best_acc')
    if exists(best_acc_file):
        return float(open(best_acc_file).readline())
    else:
        return 0.0


def write_best_acc(model_config, acc):
    best_acc_file = join(model_config.resultdir, 'best_acc')
    open(best_acc_file, 'w').write(str(acc))


def get_ckpt(modeldir, logdir, wait_second=60):
    while True:
        try:
            ckpt = copy_ckpt_to_modeldir(modeldir, logdir)
            return ckpt
        except FileNotFoundError as exp:
            if wait_second:
                print(str(exp) + '\nWait for 1 minutes.')
                time.sleep(wait_second)
            else:
                return None


class TestBaseConfig(BaseConfig):

    def __init__(self, test_file=None):
        self.eval_file = test_file

    zhaos5_data_path = '/home/zhaos5/projs/wsd/wsd_data'
    voc_file = zhaos5_data_path + '/mimic/subvocab'

    train_file = zhaos5_data_path + '/mimic/train'

    abbr_file = zhaos5_data_path + '/mimic/abbr'
    cui_file = zhaos5_data_path + '/mimic/cui'
    abbr_mask_file = zhaos5_data_path + '/mimic/abbr_mask'

    stype_voc_file = zhaos5_data_path + '/mimic/cui_extra_stype.voc'
    cui_extra_pkl = zhaos5_data_path + '/mimic/cui_extra.pkl'


if __name__ == '__main__':
    args = get_args()
    dataset_paths = DataSetPaths(args.environment)

    model_config = BaseConfig()

    # ckpt = '/home/zhaos5/projs/wsd/wsd_perf/0930_base_abbrabbr_train_extradef/model/model.ckpt-6434373'
    # ckpt_path = '/exp_data/20181021_base_abbrabbr/model/model.ckpt-4070595'
    ckpt_path = '/home/memray/Project/upmc/wsd/wsd_perf/1020_clas/log/model.ckpt-65'
    # ckpt_path = '/Users/memray/Project/upmc_wsd/wsd_perf/1020_clas/log/model.ckpt-2532'
    # #####################################
    # # testing (directly compute score, not using standard pipeline)
    # #####################################
    # acc = eval(model_config, ckpt)
    # write_best_acc(model_config, acc)

    #####################################
    # testing (using standard evaluation pipeline)
    #####################################
    predict_from_checkpoint(model_config, ckpt_path)
