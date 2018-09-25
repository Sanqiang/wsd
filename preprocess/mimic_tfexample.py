"""
Prepare TfRecord
"""
import tensorflow as tf

PATH_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
PATH_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'
PATH_TF_TRAIN = '/home/zhaos5/projs/wsd/wsd_data/mimic/train.examples'
PATH_TF_EVAL = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval.examples'

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tf_record(data_file, tf_data_file):
    writer = tf.python_io.TFRecordWriter(tf_data_file)
    reader = open(data_file)
    #TODO(sanqiang): tf example

