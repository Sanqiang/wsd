import tensorflow as tf
from tensor2tensor.models import transformer
from util import constant
from tensor2tensor.layers import common_attention, common_layers


class Graph:
    def __init__(self, is_train, model_config, data):
        self.is_train = is_train
        self.model_config = model_config
        self.data = data
        self.hparams = transformer.transformer_base()
        self.setup_hparams()

    def create_model_multigpu(self):
        # with tf.device('/cpu:0'):
            losses = []
            grads = []
            optim = self.get_optim()
            self.objs = []

            with tf.device('/cpu:0'):
                self.global_step = tf.train.get_or_create_global_step()
                self.embs = tf.get_variable(
                    'embs', [self.data.voc.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_id in range(self.model_config.num_gpus):
                    with tf.device('/device:GPU:%d' % gpu_id):
                        with tf.name_scope('%s_%d' % ('gpu_scope', gpu_id)):
                            loss, obj = self.create_model()
                            grad = optim.compute_gradients(loss)
                            tf.get_variable_scope().reuse_variables()
                            losses.append(loss)
                            grads.append(grad)
                            self.objs.append(obj)

            with tf.variable_scope('optimization'):
                self.loss = tf.divide(tf.add_n(losses), self.model_config.num_gpus)
                self.perplexity = tf.exp(tf.reduce_mean(self.loss))

                if self.is_train:
                    avg_grad = self.average_gradients(grads)
                    grads = [g for (g, v) in avg_grad]
                    clipped_grads, _ = tf.clip_by_global_norm(grads, 4.0)
                    self.train_op = optim.apply_gradients(zip(clipped_grads, tf.trainable_variables()),
                                                          global_step=self.global_step)
                    self.increment_global_step = tf.assign_add(self.global_step,
                                                               self.model_config.batch_size * self.model_config.num_gpus)

                self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def create_model(self):
        with tf.variable_scope('variables'):
            contexts = []
            for _ in range(self.model_config.max_context_len):
                contexts.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='context_input'))

            sense_inps, abbr_sinps, abbr_einps = [], [], []
            for _ in range(self.model_config.max_abbrs):
                sense_inps.append(tf.zeros(self.model_config.batch_size, tf.int32, name='sense_input'))
                abbr_sinps.append(tf.zeros([self.model_config.batch_size], tf.int32, name='sense__sinput'))
                abbr_einps.append(tf.zeros([self.model_config.batch_size], tf.int32, name='sense_einput'))

            num_abbr = tf.zeros([self.model_config.batch_size], tf.float32, name='num_abbr')

        with tf.variable_scope('model'):
            contexts_emb = tf.stack(self.embedding_fn(contexts, self.embs), axis=1)
            contexts_emb_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(tf.stack(contexts, axis=1),
                                     self.data.voc.encode(constant.PAD))))
            contexts_emb = tf.nn.dropout(contexts_emb,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            encoder_outputs = transformer.transformer_encoder(
                contexts_emb, contexts_emb_bias, self.hparams)

            if self.model_config.aggregate_mode == 'selfattn':
                selfattn_w = tf.get_variable(
                    'selfattn_w', [1, self.model_config.dimension, 1], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                selfattn_b = tf.get_variable(
                    'selfattn_b', [1, 1, 1], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                weight = tf.nn.tanh(tf.nn.conv1d(encoder_outputs, selfattn_w, 1, 'SAME') + selfattn_b)
                encoder_outputs *= weight
                aggregate_state = tf.reduce_mean(encoder_outputs, axis=1)
            else:
                aggregate_state = tf.reduce_mean(encoder_outputs, axis=1)

        with tf.variable_scope('pred'):
            proj_w = tf.get_variable('proj_w', [self.model_config.dimension, self.data.sen_cnt], tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            proj_b = tf.get_variable('proj_b', [self.data.sen_cnt], tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            losses = []
            preds = []
            for abbr_id in range(self.model_config.max_abbrs):
                abbr_sinp = abbr_sinps[abbr_id]
                abbr_einp = abbr_einps[abbr_id]
                sense_inp = sense_inps[abbr_id]

                mask = tf.to_float(tf.sequence_mask(abbr_einp, self.data.sen_cnt)) - tf.to_float(tf.sequence_mask(abbr_sinp, self.data.sen_cnt))
                logits = tf.matmul(aggregate_state, proj_w) + proj_b
                logits *= mask

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_inp)
                loss_mask = tf.to_float(tf.not_equal(sense_inp, 0))
                loss *= loss_mask

                losses.append(loss)
                preds.append(tf.nn.top_k(logits, k=5, sorted=True)[1])

        preds = tf.stack(preds, axis=1)
        obj = {
            'contexts': contexts,
            'sense_inp': sense_inps,
            'abbr_sinp': abbr_sinps,
            'abbr_einp': abbr_einps,
            'num_abbr': num_abbr,
            'preds': preds,
        }
        return tf.add_n(losses)/num_abbr, obj

    def get_optim(self):
        learning_rate = tf.constant(self.model_config.learning_rate)

        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.model_config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        else:
            raise Exception('Not Implemented Optimizer!')
        return opt

    def embedding_fn(self, inputs, embedding):
        if type(inputs) == list:
            if not inputs:
                return []
            else:
                return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]
        else:
            return tf.nn.embedding_lookup(embedding, inputs)

    # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0
