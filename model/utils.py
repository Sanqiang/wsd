import tensorflow as tf


def get_feed(data_feeds, data, model_config, is_train):
    input_feed = {}
    exclude_cnt = 0

    for data_feed in data_feeds:
        tmp_contexts, tmp_targets, tmp_lines = [], [], []
        tmp_masked_contexts, tmp_masked_words = [], []
        cnt = 0
        while cnt < model_config.batch_size:
            if model_config.it_train:
                example = next(data.data_it, None)
                print(cnt)
                print(example)
            else:
                example = data.get_sample()

            if example is None:
                # Only used in evaluation
                example = {}
                example['contexts'] = [0] * model_config.max_context_len
                example['target'] = [0, 0, 0, 0, -1]
                example['line'] = ''
                # sample['def'] = [0] * model_config.max_def_len
                # sample['stype'] = 0
                exclude_cnt += 1  # Assume eval use single GPU

            '''
            if is_train:
                if model_config.it_train:
                    example = next(data.data_it)
                else:
                    example = data.get_sample()
            else:
                example = data.get_sample()
                if example is None:
                    # Only used in evaluation
                    example = {}
                    example['contexts'] = [0] * model_config.max_context_len
                    example['target'] = [0, 0, 0, 0, -1]
                    example['line'] = ''
                    # sample['def'] = [0] * model_config.max_def_len
                    # sample['stype'] = 0
                    exclude_cnt += 1 # Assume eval use single GPU
            '''
            tmp_contexts.append(example['contexts'])
            tmp_targets.append(example['target'])
            tmp_lines.append(example['line'])
            # print('input:\t%s\t%s.' % (sample['line'], sample['target']))
            if model_config.lm_mask_rate and 'cur_masked_contexts' in example:
                tmp_masked_contexts.append(example['cur_masked_contexts'])
                tmp_masked_words.append(example['masked_words'])

            cnt += 1

        for step in range(model_config.max_context_len):
            input_feed[data_feed['contexts'][step].name] = [
                tmp_contexts[batch_idx][step]
                for batch_idx in range(model_config.batch_size)]

        if model_config.hub_module_embedding:
            input_feed[data_feed['text_input'].name] = [
                tmp_lines[batch_idx]
                for batch_idx in range(model_config.batch_size)]

        input_feed[data_feed['abbr_inp'].name] = [
            tmp_targets[batch_idx][1]
            for batch_idx in range(model_config.batch_size)
        ]
        input_feed[data_feed['sense_inp'].name] = [
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
                input_feed[data_feed['masked_contexts'][step].name] = [
                    tmp_masked_contexts[batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

            for step in range(model_config.max_subword_len):
                input_feed[data_feed['masked_words'][step].name] = [
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
