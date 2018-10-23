from data_generator.data import TrainData
from model.graph import Graph

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
import time
import numpy as np
import random as rd

from model.model_config import get_args

import sys
sys.path.insert(0,'/zfs1/hdaqing/saz31/wsd/wsd_code')

args = get_args()


def get_feed(objs, data, model_config, is_train):
    input_feed = {}
    exclude_cnt = 0
    for obj in objs:
        tmp_contexts, tmp_targets, tmp_lines = [], [], []
        tmp_masked_contexts, tmp_masked_words = [], []
        cnt = 0
        while cnt < model_config.batch_size:
            if is_train:
                if model_config.it_train:
                    sample = next(data.data_it)
                else:
                    sample = data.get_sample()
            else:
                sample = data.get_sample()
                if sample is None:
                    # Only used in evaluation
                    sample = {}
                    sample['contexts'] = [0] * model_config.max_context_len
                    sample['target'] = [0, 0, 0, 0, -1]
                    sample['line'] = ''
                    # sample['def'] = [0] * model_config.max_def_len
                    # sample['stype'] = 0
                    exclude_cnt += 1 # Assume eval use single GPU

            tmp_contexts.append(sample['contexts'])
            tmp_targets.append(sample['target'])
            tmp_lines.append(sample['line'])
            # print('input:\t%s\t%s.' % (sample['line'], sample['target']))
            if model_config.lm_mask_rate and 'cur_masked_contexts' in sample:
                tmp_masked_contexts.append(sample['cur_masked_contexts'])
                tmp_masked_words.append(sample['masked_words'])

            cnt += 1

        for step in range(model_config.max_context_len):
            input_feed[obj['contexts'][step].name] = [
                tmp_contexts[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

        if model_config.hub_module_embedding:
            input_feed[obj['text_input'].name] = [
                tmp_lines[batch_idx]
                for batch_idx in range(model_config.batch_size)]

        input_feed[obj['abbr_inp'].name] = [
            tmp_targets[batch_idx][1]
            for batch_idx in range(model_config.batch_size)
        ]
        input_feed[obj['sense_inp'].name] = [
            tmp_targets[batch_idx][2]
            for batch_idx in range(model_config.batch_size)
        ]

        if model_config.lm_mask_rate and tmp_masked_contexts:
            i = 0
            while len(tmp_masked_contexts) < model_config.batch_size:
                tmp_masked_contexts.append(tmp_masked_contexts[i % len(tmp_masked_contexts)])
                tmp_masked_words.append(tmp_masked_words[i % len(tmp_masked_contexts)])
                i += 1

            for step in range(model_config.max_context_len):
                input_feed[obj['masked_contexts'][step].name] = [
                    tmp_masked_contexts[batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

            for step in range(model_config.max_subword_len):
                input_feed[obj['masked_words'][step].name] = [
                    tmp_masked_words[batch_idx][1][step]
                    for batch_idx in range(model_config.batch_size)]

    return input_feed, exclude_cnt, tmp_targets


def get_feed_cui(obj, data, model_config):
    """Feed the CUI model."""
    input_feed = {}
    tmp_extra_cui_def, tmp_extra_cui_stype, tmp_cuiid, tmp_abbrid = [], [], [], []
    cnt = 0
    while cnt < model_config.batch_size:
        sample = next(data.data_it_cui)
        tmp_cuiid.append(sample['cui_id'])
        tmp_abbrid.append(sample['abbr_id'])
        if 'def' in model_config.extra_loss:
            tmp_extra_cui_def.append(sample['def'])
        if 'stype' in model_config.extra_loss:
            tmp_extra_cui_stype.append(sample['stype'])
        cnt += 1

    input_feed[obj['abbr_inp'].name] = [
        tmp_abbrid[batch_idx]
        for batch_idx in range(model_config.batch_size)
    ]
    input_feed[obj['sense_inp'].name] = [
        tmp_cuiid[batch_idx]
        for batch_idx in range(model_config.batch_size)
    ]

    if 'def' in model_config.extra_loss:
        for step in range(model_config.max_def_len):
            input_feed[obj['def'][step].name] = [
                tmp_extra_cui_def[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

    if 'stype' in model_config.extra_loss:
        input_feed[obj['stype'].name] = [
            tmp_extra_cui_stype[batch_idx]
            for batch_idx in range(model_config.batch_size)
        ]

    return input_feed


def get_session_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.log_device_placement = True
    config.gpu_options.allocator_type = "BFC"
    config.gpu_options.allow_growth = True
    return config


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
    data = TrainData(model_config)

    graph = Graph(True, model_config, data)
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
                graph.saver.restore(sess, ckpt.model_checkpoint_path)

        if model_config.init_emb:
            sess.run(graph.embs_init_fn)
            print('init embedding from %s' % model_config.init_emb)

        perplexitys = []
        start_time = datetime.now()
        previous_step = 0
        previous_step_cui = 0
        while True:
            # Train task
            for _ in range(model_config.task_iter_steps):
                input_feed, _, _ = get_feed(graph.objs, data, model_config, True)
                fetches = [graph.train_op, graph.increment_global_step_task, graph.increment_global_step,
                           graph.global_step_task,
                           graph.perplexity,
                           graph.loss]
                _, _, _, step, perplexity, loss = sess.run(fetches, input_feed)
                perplexitys.append(perplexity)

                if (step - previous_step) > model_config.model_print_freq:
                    end_time = datetime.now()
                    time_span = end_time - start_time
                    start_time = end_time
                    print('TASK: Perplexity:\t%f at step %d using %s with loss:%s.'
                          % (perplexity, step, time_span,
                             np.mean(loss)))
                    perplexitys.clear()
                    previous_step = step

            # Fine tune CUI
            if model_config.extra_loss:
                for _ in range(model_config.cui_iter_steps):
                    input_feed = get_feed_cui(graph.obj_cui, data, model_config)
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
    from model.model_config import DummyConfig, BaseConfig, VocBaseConfig
    if args.mode == 'dummy':
        train(DummyConfig())
    elif args.mode == 'base':
        train(BaseConfig())
    elif args.mode == 'voc':
        train(VocBaseConfig())

