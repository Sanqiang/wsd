import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-mode', '--mode', default='dummy',
                        help='Mode?')
    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=3, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-env', '--environment', default='crc',
                        help='The environment machine?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-result', '--result_folder', default='result_val',
                        help='Result folder?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')

    parser.add_argument('-op', '--optimizer', default='adam',
                        help='Which optimizer to use?')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Value of learning rate?')
    parser.add_argument('-layer_drop', '--layer_prepostprocess_dropout', default=0.0, type=float,
                        help='Dropout rate for data input?')
    parser.add_argument('-model_print_freq', '--model_print_freq', default=100, type=int,
                        help='Print frequency of model training?')


    # For Data
    parser.add_argument('-mc', '--min_count', default=0, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-eval_freq', '--model_eval_freq', default=10000, type=int,
                        help='The frequency of evaluation at training? not use if = 0.')
    parser.add_argument('-max_context_len', '--max_context_len', default=300, type=int,
                        help='Max of context length?')

    # For Graph
    parser.add_argument('-dim', '--dimension', default=16, type=int,
                        help='Size of dimension?')
    parser.add_argument('-ns', '--number_samples', default=0, type=int,
                        help='Number of samples used in Softmax?')
    parser.add_argument('-max_abbrs', '--max_abbrs', default=3, type=int,
                        help='Size of targets?')

    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')
    parser.add_argument('-cprocess', '--enc_postprocess', default='',
                        help='Postprocess of the encoder?')
    parser.add_argument('-nhl', '--num_hidden_layers', default=2, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=2, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-ndl', '--num_decoder_layers', default=2, type=int,
                        help='Number of decoder layer?')
    parser.add_argument('-nh', '--num_heads', default=2, type=int,
                        help='Number of multi-attention heads?')

    # For Our Idea
    parser.add_argument('-ag_mode', '--aggregate_mode', default='selfattn',
                        help='The mode transform the encoder output to single hidden state')

    # For Test
    parser.add_argument('-test_ckpt', '--test_ckpt', default='',
                        help='Path for test ckpt checkpoint?')

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
    if env == 'crc':
        return "/zfs1/hdaqing/saz31/wsd/tmp/" + file_path
    elif env == 'aws':
        return '/home/zhaos5/projs/wsd/wsd_perf/tmp/' + file_path
    else:
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


args = get_args()


class DummyConfig():
    mode = args.mode

    voc_file = get_path('../wsd_data/medline/subvoc.txt')

    train_file = get_path('../wsd_data/dummy/train.txt')
    eval_file = get_path('../wsd_data/dummy/eval.txt')

    abbr_common_file = get_path('../wsd_data/medline/abbr_common.txt')
    abbr_rare_file = get_path('../wsd_data/medline/abbr_rare.txt')

    max_context_len = args.max_context_len
    aggregate_mode = args.aggregate_mode
    max_abbrs = args.max_abbrs
    subword_vocab_size = 1

    num_heads = args.num_heads
    num_hidden_layers = args.num_hidden_layers
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    hparams_pos = args.hparams_pos
    enc_postprocess = args.enc_postprocess.split(':')
    dimension = args.dimension
    layer_prepostprocess_dropout = args.layer_prepostprocess_dropout
    save_model_secs = 600
    model_print_freq = args.model_print_freq

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    optimizer = args.optimizer
    environment = args.environment
    num_gpus = args.num_gpus
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/test1/', environment)
    modeldir = get_path('../' + output_folder + '/model/', environment)
    logdir = get_path('../' + output_folder + '/log/', environment)


class BaseConfig(DummyConfig):
    voc_file = get_path('../wsd_data/medline/subvoc.txt')

    train_file = get_path('../wsd_data/medline/train.txt')
    eval_file = get_path('../wsd_data/medline/eval.txt')

    abbr_common_file = get_path('../wsd_data/medline/abbr_common.txt')
    abbr_rare_file = get_path('../wsd_data/medline/abbr_rare.txt')

    save_model_secs = 600
    model_print_freq = 1000


