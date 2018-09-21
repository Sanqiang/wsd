from data_generator.data import TrainData
from model.graph import LSTMGraph

import tensorflow as tf
from datetime import datetime
import time

from model.model_config import get_args


args = get_args()


def get_feed(objs, data, model_config, is_train):
    input_feed = {}
    exclude_cnt = 0
    for obj in objs:
        tmp_contexts, tmp_targets, tmp_lines, tmp_sense_inps, tmp_abbr_sinps, tmp_abbr_einps = [], [], [], [], [], []
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
                    sample = {}
                    sample['contexts'] = [0] * model_config.max_context_len
                    sample['target'] = [0, 0, 0]
                    sample['line'] = ''
                    exclude_cnt += 1 # Assume eval use single GPU

            tmp_contexts.append(sample['contexts'])
            tmp_targets.append(sample['target'])
            tmp_lines.append(sample['line'])
            cnt += 1

        for step in range(model_config.max_context_len):
            input_feed[obj['contexts'][step].name] = [
                tmp_contexts[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

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
        input_feed[obj['abbr_sinp'].name] = [
            data.abbrs_pos[tmp_targets[batch_idx][1]]['s_i']
            for batch_idx in range(model_config.batch_size)
        ]
        input_feed[obj['abbr_einp'].name] = [
            data.abbrs_pos[tmp_targets[batch_idx][1]]['e_i']
            for batch_idx in range(model_config.batch_size)
        ]
    return input_feed, exclude_cnt, tmp_targets

def get_session_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.log_device_placement = True
    config.gpu_options.allocator_type = "BFC"
    # config.gpu_options.allow_growth = True
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
    graph = LSTMGraph(True, model_config, data)
    graph.create_model_multigpu()
    print('Built Model Done!')
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_config.logdir,
        save_checkpoint_secs=model_config.save_model_secs,
        config=get_session_config(),
    ) as sess:
        ckpt = tf.train.get_checkpoint_state(model_config.logdir)
        if ckpt:
            graph.saver.restore(sess, ckpt.model_checkpoint_path)
        perplexitys = []
        start_time = datetime.now()
        previous_step = 0
        while True:
            input_feed, _, _ = get_feed(graph.objs, data, model_config, True)
            fetches = [graph.train_op, graph.increment_global_step, graph.global_step,
                       graph.perplexity]
            _, _, step, perplexity = sess.run(fetches, input_feed)
            perplexitys.append(perplexity)

            if (step - previous_step) > model_config.model_print_freq:
                end_time = datetime.now()
                time_span = end_time - start_time
                start_time = end_time
                print('Perplexity:\t%f at step %d using %s.' % (perplexity, step, time_span))
                perplexitys.clear()
                previous_step = step


if __name__ == '__main__':
    from model.model_config import DummyConfig, BaseConfig
    if args.mode == 'dummy':
        train(DummyConfig())
    elif args.mode == 'base':
        train(BaseConfig())

