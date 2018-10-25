from data_generator.data import EvalData, TrainData
from dataset_helper import DataSetPaths
from model.graph import Graph
from model.train import get_feed, get_session_config
from util.checkpoint import copy_ckpt_to_modeldir

from model.model_config import DummyConfig, BaseConfig, VocBaseConfig

import tensorflow as tf
import numpy as np

import time
import glob
from datetime import datetime
from os.path import exists, join
from os import makedirs, listdir, remove

from model.model_config import get_args

import sys

args = get_args()


def eval(model_config, ckpt):
    assert model_config.num_gpus == 1 # Eval only uses single GPU

    train_data = TrainData(model_config)
    graph = Graph(False, model_config, train_data)
    tf.reset_default_graph()
    graph.create_model_multigpu()
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_config.logdir,
        config=get_session_config()
    )
    graph.saver.restore(sess, ckpt)

    for test_dataset_name in ['share', 'msh', 'mimic']:
        if test_dataset_name == 'share':
            test_file = dataset_paths.share_txt
        elif test_dataset_name == 'msh':
            test_file = dataset_paths.msh_txt
        elif test_dataset_name == 'mimic:':
            test_file = dataset_paths.mimic_eval_txt

        model_config = BaseConfig()
        setattr(model_config, 'eval_file', test_file)
        eval_data = EvalData(model_config)

        perplexitys = []
        total_cnt = 0.0
        correct_cnt, correct_cnt2, correct_cnt3, correct_cnt4, correct_cnt5 = 0.0, 0.0, 0.0, 0.0, 0.0
        report = []
        start_time = datetime.now()

        while True:
            input_feed, exclude_cnt, gt_targets = get_feed(graph.data_feeds, eval_data, model_config, False)
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

                report.append('Abbr:%s\tPred:%s\tGt:%s\tline:%s with step %s with loss %s.' %
                              (eval_data.id2abbr[abbr_id],
                               ';'.join([eval_data.id2sense[loop] for loop in pred]),
                               eval_data.id2sense[gt_target[2]],
                               gt_target[3],
                               gt_target[0],
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


if __name__ == '__main__':
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

    dataset_paths = DataSetPaths()

    if args.mode == 'dummy':
        model_config = DummyConfig()
        best_acc = get_best_acc(model_config)
        while True:
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                acc = eval(model_config, ckpt)
                if acc > best_acc:
                    best_acc = acc
                    write_best_acc(model_config, best_acc)
                    for file in listdir(model_config.modeldir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            remove(model_config.modeldir + file)
                else:
                    for fl in glob.glob(ckpt + '*'):
                        remove(fl)
    elif args.mode == 'base':
        model_config = BaseConfig()
        best_acc = get_best_acc(model_config)
        while True:
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                acc = eval(model_config, ckpt)
                if acc > best_acc:
                    best_acc = acc
                    write_best_acc(model_config, best_acc)
                    for file in listdir(model_config.modeldir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            remove(model_config.modeldir + file)
                else:
                    for fl in glob.glob(ckpt + '*'):
                        remove(fl)
    elif args.mode == 'voc':
        model_config = VocBaseConfig()
        best_acc = get_best_acc(model_config)
        while True:
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                acc = eval(model_config, ckpt)
                if acc > best_acc:
                    best_acc = acc
                    write_best_acc(model_config, best_acc)
                    for file in listdir(model_config.modeldir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            remove(model_config.modeldir + file)
                else:
                    for fl in glob.glob(ckpt + '*'):
                        remove(fl)

