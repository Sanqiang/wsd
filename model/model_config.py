import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-mode', '--mode', default='dummy',
                        help='Mode?')
    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=2, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-env', '--environment', default='sys',
                        help='The environment machine?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-result', '--result_folder', default='result_val',
                        help='Result folder?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')

    parser.add_argument('-op', '--optimizer', default='adagrad',
                        help='Which optimizer to use?')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')
    parser.add_argument('-layer_drop', '--layer_prepostprocess_dropout', default=0.0, type=float,
                        help='Dropout rate for data input?')
    parser.add_argument('-model_print_freq', '--model_print_freq', default=50, type=int,
                        help='Print frequency of model training?')
    parser.add_argument('-task_iter_steps', '--task_iter_steps', default=5000, type=int,
                        help='The number for task interaction opposed to training cui ?')
    parser.add_argument('-cui_iter_steps', '--cui_iter_steps', default=800, type=int,
                        help='The number for cui interaction opposed to training task ?')


    # For Data
    parser.add_argument('-mc', '--min_count', default=0, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-eval_freq', '--model_eval_freq', default=10000, type=int,
                        help='The frequency of evaluation at training? not use if = 0.')
    parser.add_argument('-max_context_len', '--max_context_len', default=1000, type=int,
                        help='Max of context length?')
    parser.add_argument('-max_sense_cand', '--max_sense_cand', default=120, type=int,
                        help='Max of sense cands?')
    parser.add_argument('-max_subword_len', '--max_subword_len', default=10, type=int,
                        help='Max of subword length?')
    parser.add_argument('-vprocess', '--voc_process', default='',
                        help='Preprocess of vocab?')
    parser.add_argument('-it', '--it_train', default=False, type=bool,
                        help='Iteractive Processing Data?')

    # For Graph
    parser.add_argument('-dim', '--dimension', default=32, type=int,
                        help='Size of dimension?')
    parser.add_argument('-ns', '--number_samples', default=0, type=int,
                        help='Number of samples used in Softmax?')
    parser.add_argument('-ag_mode', '--aggregate_mode', default='avg',
                        help='The mode transform the encoder output to single hidden state')
    parser.add_argument('-pred_mode', '--predict_mode', default='match',
                        help='The mode for prediction, either [clas, match]')
    parser.add_argument('-ptr_mode', '--pointer_mode', default=None,
                        help='The mode for pointer network, either [none, first_dist]')
    parser.add_argument('-a_mode', '--abbr_mode', default='abbr',
                        help='The mode for feed abbr [abbr, sense]')
    parser.add_argument('-neg_cnt', '--negative_sampling_count', default=0, type=int,
                        help='The number of negative sampling for abbr?')
    parser.add_argument('-encoder_mode', '--encoder_mode', default='t2t',
                        help='The mode for encoder [t2t, ut2t]')

    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')
    parser.add_argument('-cprocess', '--enc_postprocess', default='',
                        help='Postprocess of the encoder?')
    parser.add_argument('-nhl', '--num_hidden_layers', default=2, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=2, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-nh', '--num_heads', default=2, type=int,
                        help='Number of multi-attention heads?')
    parser.add_argument('-hub_emb', '--hub_module_embedding', default='',
                        help='The hub module used for extra embedding?')

    parser.add_argument('-train_emb', '--train_emb', default=True,
                        help='Whether to train embedding?')
    parser.add_argument('-init_emb', '--init_emb', default=None,
                        help='The path of pretrained word embedding? If none to disable.')

    parser.add_argument('-lmr', '--lm_mask_rate', default=0.0, type=float,
                        help='The mask LM rate for learning LM for better word embedding '
                             '(https://arxiv.org/pdf/1810.04805.pdf)')

    # For Test
    parser.add_argument('-test_ckpt', '--test_ckpt', default='',
                        help='Path for test ckpt checkpoint?')

    # Extra Loss
    parser.add_argument('-eloss', '--extra_loss', default='',
                        help='Extra loss for for better sense understanidng? '
                             'choose from [def, stype], separate by :')
    parser.add_argument('-max_def_len', '--max_def_len', default=100, type=int,
                        help='Max of def length?')


    args = parser.parse_args()
    return args


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output


def get_path(file_path, env='sys'):
    if env == 'aws':
        return '/home/zhaos5/projs/wsd/wsd_perf/tmp/' + file_path
    elif env == 'crc':
        return '/zfs1/hdaqing/saz31/wsd/wsd_perf/tmp/' + file_path
    elif env == 'luoz3':
        return '/home/luoz3/wsd_result/tmp/' + file_path
    elif env == 'mengr':
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path
    elif env == 'aws_mengr':
        return '/exp_data/wsd_data/' + file_path
    else:
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


args = get_args()


class DummyConfig():
    mode = args.mode
    environment = args.environment
    architecture = args.architecture

    train_file = get_path('../wsd_data/dummy/train.txt', env=args.environment)
    eval_file = get_path('../wsd_data/dummy/eval.txt', env=args.environment)
    voc_file = get_path('../wsd_data/dummy/subvocab', env=args.environment)

    # abbr_common_file = get_path('../wsd_data/medline/abbr_common.txt')
    # abbr_rare_file = get_path('../wsd_data/medline/abbr_rare.txt')
    abbr_file = get_path('../wsd_data/dummy/abbr')
    cui_file = get_path('../wsd_data/dummy/cui')
    abbr_mask_file = get_path('../wsd_data/dummy/abbr_mask')

    stype_voc_file = get_path('../wsd_data/dummy/cui_extra_stype.voc')
    cui_extra_pkl = get_path('../wsd_data/dummy/cui_extra.pkl')

    max_context_len = args.max_context_len
    max_sense_cand = args.max_sense_cand
    max_subword_len = args.max_subword_len
    max_def_len = args.max_def_len
    predict_mode = args.predict_mode
    abbr_mode = args.abbr_mode
    # TODO(sanqiang): add neg sampling when new data comes
    negative_sampling_count = args.negative_sampling_count
    aggregate_mode = args.aggregate_mode
    pointer_mode = args.pointer_mode
    if aggregate_mode is not None:
        aggregate_mode = aggregate_mode.split(':')
    subword_vocab_size = 1

    num_heads = args.num_heads
    num_hidden_layers = args.num_hidden_layers
    num_encoder_layers = args.num_encoder_layers
    hparams_pos = args.hparams_pos
    enc_postprocess = args.enc_postprocess.split(':')

    it_train = args.it_train
    voc_process = args.voc_process.split(':')

    hub_module_embedding = args.hub_module_embedding

    dimension = args.dimension
    layer_prepostprocess_dropout = args.layer_prepostprocess_dropout
    save_model_secs = 30
    model_print_freq = 10

    learning_rate = 0.001
    batch_size = args.batch_size
    optimizer = args.optimizer
    environment = args.environment
    num_gpus = args.num_gpus
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/test1/', environment)
    modeldir = get_path('../' + output_folder + '/model/', environment)
    logdir = get_path('../' + output_folder + '/log/', environment)

    extra_loss = [v for v in args.extra_loss.split(':') if len(v)>0]

    task_iter_steps = args.task_iter_steps
    cui_iter_steps = args.cui_iter_steps
    task_iter_steps = 50
    cui_iter_steps = 10

    warm_start = args.warm_start
    lm_mask_rate = args.lm_mask_rate

    train_emb = args.train_emb
    init_emb = args.init_emb

    encoder_mode = args.encoder_mode


class BaseConfig(DummyConfig):
    learning_rate = args.learning_rate
    subword_vocab_size = 8000

    voc_file = get_path('../wsd_data/mimic/subvocab', env=args.environment)

    train_file = get_path('../wsd_data/mimic/train', env=args.environment)
    # train_pickle = get_path('../wsd_data/mimic/train.pkl')
    eval_file = get_path('../wsd_data/mimic/eval', env=args.environment)

    abbr_file = get_path('../wsd_data/mimic/abbr', env=args.environment)
    cui_file = get_path('../wsd_data/mimic/cui', env=args.environment)
    abbr_mask_file = get_path('../wsd_data/mimic/abbr_mask', env=args.environment)
    # abbr_rare_file = get_path('../wsd_data/medline/abbr_rare.txt')

    stype_voc_file = get_path('../wsd_data/mimic/cui_extra_stype.voc', env=args.environment)
    cui_extra_pkl = get_path('../wsd_data/mimic/cui_extra.pkl', env=args.environment)

    # save_model_secs = 600
    # model_print_freq = 1000
    voc_process = args.voc_process.split(':')

    save_model_secs = 1200
    model_print_freq = args.model_print_freq
    task_iter_steps = args.task_iter_steps
    cui_iter_steps = args.cui_iter_steps


class VocBaseConfig(BaseConfig):
    # Use ptr-trained word embedding (rather than subtoken)
    subword_vocab_size = 0
    voc_file = get_path('../wsd_data/mimic/w2v_vocab')

    train_emb = args.train_emb
    init_emb = get_path('../wsd_data/mimic/w2v_vectors.npy')
