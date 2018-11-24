import tensorflow as tf


class ATTN:
    """Implementation of Attn based LSTM http://www.aclweb.org/anthology/P16-1044"""
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data= data

    def get_logits(self, query, values):
        return None


class NTN:
    """Implementation of Neural Tensor Network in https://www.ijcai.org/Proceedings/15/Papers/188.pdf"""
    def __init__(self, model_config, data, num_slice=3):
        self.num_slice = num_slice
        self.model_config = model_config
        self.data= data

        self.W = tf.get_variable(
            'ntn_bilinear_w', [self.model_config.dimension, self.model_config.dimension, self.num_slice], tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        self.V = tf.get_variable(
            'ntn_proj_v', [self.model_config.dimension*2, self.num_slice], tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        self.b = tf.get_variable(
            'ntn_bias', [self.num_slice], tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        self.U = tf.get_variable(
            'ntn_u', [self.num_slice], tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

    def get_logits(self, query, values):
        # query: [bs, dim] values: [dim, #cui]
        energies = []
        for i in range(self.num_slice):
            # bilinear term
            cur_W = self.W[:, :, i]
            bilinear_term = tf.matmul(query, cur_W)
            bilinear_term = tf.matmul(bilinear_term, values)

            # proj term
            cur_V = tf.expand_dims(tf.expand_dims(self.V[:, i], axis=-1), axis=0)
            query_stack = tf.tile(tf.expand_dims(query, axis=1),
                                  [1, self.data.sen_cnt, 1])
            values_stack = tf.tile(tf.expand_dims(tf.transpose(values), axis=0),
                                   [self.model_config.batch_size, 1, 1])
            proj_term = tf.squeeze(
                tf.nn.conv1d(tf.concat([query_stack, values_stack], axis=-1), cur_V, 1, 'SAME'), axis=-1)

            # bias term
            bias_term = self.b[i]

            energy = bilinear_term + proj_term + bias_term
            energies.append(energy)

        energy_concat = tf.tanh(tf.stack(energies, axis=-1))
        return tf.squeeze(
            tf.nn.conv1d(energy_concat, tf.expand_dims(tf.expand_dims(self.U, axis=-1), axis=0), 1, 'SAME'), axis=-1)


