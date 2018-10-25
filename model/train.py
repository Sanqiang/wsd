from tensorflow.keras.utils import Progbar
from data_generator.data import TrainData
from model.graph import Graph

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
import time
import numpy as np
import random as rd

from model.model_config import get_args
from model.model_config import DummyConfig, BaseConfig, VocBaseConfig

import sys

from model.test import evaluate_on_testsets
from model.utils import get_feed, get_feed_cui, get_session_config

sys.path.insert(0,'/zfs1/hdaqing/saz31/wsd/wsd_code')

args = get_args()


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output


def train(model_config):
    print(list_config(model_config))
    train_data = TrainData(model_config)

    graph = Graph(True, model_config, train_data)
    graph.create_model_multigpu()
    print('Built Model Done!')

    if model_config.warm_start:
        ckpt_path = model_config.warm_start
        var_list = slim.get_variables_to_restore()
        available_vars = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_dict = {var.op.name: var for var in var_list}
        for var in var_dict:
            if reader.has_tensor(var):
                var_ckpt = reader.get_tensor(var)
                var_cur = var_dict[var]
                if any([var_cur.shape[i] != var_ckpt.shape[i] for i in range(len(var_ckpt.shape))]):
                    print('Variable %s missing due to shape.', var)
                else:
                    available_vars[var] = var_dict[var]
            else:
                print('Variable %s missing.', var)

            partial_restore_ckpt = slim.assign_from_checkpoint_fn(
                ckpt_path, available_vars,
                ignore_missing_vars=True, reshape_variables=False)

    print('Start training')

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_config.logdir,
        save_checkpoint_secs=model_config.save_model_secs,
        config=get_session_config(),
        hooks=[tf.train.CheckpointSaverHook(
            model_config.logdir, save_secs=model_config.save_model_secs, saver=graph.saver)]
    ) as sess:
        if model_config.warm_start:
            partial_restore_ckpt(sess)
            print('Warm start with ckpt %s' % model_config.warm_start)
        else:
            ckpt = tf.train.get_checkpoint_state(model_config.logdir)
            if ckpt:
                print('Loading previous checkpoint from: %s' % model_config.logdir)
                graph.saver.restore(sess, ckpt.model_checkpoint_path)

        if model_config.init_emb:
            sess.run(graph.embs_init_fn)
            print('init embedding from %s' % model_config.init_emb)

        perplexitys = []
        start_time = datetime.now()
        previous_step = 0
        previous_step_cui = 0
        while True:
            progbar = Progbar(target=train_data.size)
            # Train task
            for _ in range(model_config.task_iter_steps):
                input_feed, _, _ = get_feed(graph.data_feeds, train_data, model_config, True)
                fetches = [graph.train_op, graph.increment_global_step_task, graph.increment_global_step,
                           graph.global_step_task,
                           graph.perplexity,
                           graph.loss]

                _, _, _, step, perplexity, loss = sess.run(fetches, input_feed)
                perplexitys.append(perplexity)

                progbar.update(current=step * model_config.batch_size, values=[('loss', loss), ('ppl', perplexity)])

                if (step - previous_step) > model_config.model_print_freq:
                    end_time = datetime.now()
                    time_span = end_time - start_time
                    start_time = end_time
                    print('\nTASK: Perplexity:\t%f at step=%d using %s with loss=%s.'
                          % (perplexity, step, time_span,
                             np.mean(loss)))
                    perplexitys.clear()
                    previous_step = step

                # instance_collections = evaluate_on_testsets(sess, graph, train_data)

            # Fine tune CUI
            if model_config.extra_loss:
                for _ in range(model_config.cui_iter_steps):
                    input_feed = get_feed_cui(graph.obj_cui, train_data, model_config)
                    fetches = [graph.train_op_cui, graph.increment_global_step_cui, graph.increment_global_step,
                               graph.global_step_cui,
                               graph.perplexity_cui,
                               graph.loss_cui]
                    _, _, _, step, perplexity, loss = sess.run(fetches, input_feed)
                    if (step - previous_step_cui) > model_config.model_print_freq:
                        end_time = datetime.now()
                        time_span = end_time - start_time
                        start_time = end_time
                        print('CUI: Perplexity:\t%f at step %d using %s with loss:%s.'
                              % (perplexity, step, time_span,
                                 np.mean(loss)))
                        perplexitys.clear()
                        previous_step_cui = step


if __name__ == '__main__':
    if args.mode == 'dummy':
        train(DummyConfig())
    elif args.mode == 'base':
        train(BaseConfig())
    elif args.mode == 'voc':
        train(VocBaseConfig())

