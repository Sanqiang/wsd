import tensorflow as tf
from tensor2tensor.models import transformer
from util import constant
from tensor2tensor.layers import common_attention, common_layers
import tensorflow_hub as hub
import numpy as np


class BaseGraph:
    def __init__(self, is_train, model_config, data):
        self.is_train = is_train
        self.model_config = model_config
        self.data = data
        self.hparams = transformer.transformer_base()
        self.setup_hparams()

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

    def get_aggregate_state(self, encoder_outputs):
        if 'selfattn' in self.model_config.aggregate_mode:
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
        return aggregate_state


class ContextEncoder(BaseGraph):
    def __init__(self, is_train, model_config, data, embs):
        BaseGraph.__init__(self, is_train, model_config, data)
        self.embs = embs

    def embed_context(self, contexts, abbr_inp_emb):
        with tf.variable_scope('context_embed'):
            contexts_emb = self.embedding_fn(contexts, self.embs)
            return contexts_emb

    def context_encoder(self, contexts_emb, contexts, abbr_inp_emb):
        """

        :param contexts_emb: a tensor of [batch_size, max_context_len, emb_dim]
        :param contexts: a list of [max_context_len, batch_size]
        :param abbr_inp_emb: a tensor of [batch_size, num_abbr=1, emb_dim]
        :return:
        """
        weights = {}
        # Create an bias tensor as mask (big neg values for padded part), input=[batch_size, context_len], output=[batch_size, 1, 1, context_len]
        contexts_bias = common_attention.attention_bias_ignore_padding(
            tf.to_float(tf.equal(tf.stack(contexts, axis=1),
                                 self.data.voc.encode(constant.PAD))))
        # add dropout to context input [batch_size, max_context_len, emb_dim]
        contexts_emb = tf.nn.dropout(contexts_emb,
                                     1.0 - self.hparams.layer_prepostprocess_dropout)
        # get the output vector of transformer
        encoder_ouput = transformer.transformer_encoder_abbr(
            contexts_emb, contexts_bias, abbr_inp_emb,
            abbr_bias = tf.zeros([self.model_config.batch_size,1,1,1]),
            hparams=self.hparams,
            save_weights_to=weights)

        return encoder_ouput, weights


class PointerNetwork(BaseGraph):
    def __init__(self, is_train, model_config, data, weights, contexts):
        BaseGraph.__init__(self, is_train, model_config, data)
        self.weights = weights
        self.contexts = tf.stack(contexts, axis=1)

    def getLogitFromFirstSelfAttnDist(self):
        attn_dist = tf.reduce_sum(list(self.weights.values())[0][:, 0, :, :], axis=1)

        batch_nums = tf.range(0, limit=self.model_config.batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        batch_nums = tf.tile(batch_nums, [1, self.model_config.max_context_len])
        indices = tf.stack((batch_nums, self.contexts), axis=2)
        shape = [self.model_config.batch_size, self.data.voc.vocab_size()]
        projected_attn_dist = tf.scatter_nd(indices, attn_dist, shape)
        return projected_attn_dist


class Graph(BaseGraph):
    def __init__(self, is_train, model_config, data):
        BaseGraph.__init__(self, is_train, model_config, data)

    def create_model_multigpu(self):
        with tf.device('/cpu:0'):
            losses = []
            grads = []
            optim = self.get_optim()
            self.objs = []

            with tf.device('/cpu:0'):
                self.global_step = tf.train.get_or_create_global_step()

                # Vocab embedding
                self.embs = tf.get_variable(
                    'embs', [self.data.voc.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

                # Abbr embedding
                if self.model_config.abbr_mode == 'abbr':
                    self.abbr_embs = tf.get_variable(
                        'abbr_embs', [len(self.data.id2abbr), self.model_config.dimension], tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

                # Semantic type embedding
                if 'stype' in self.model_config.extra_loss:
                    self.stype_embs = tf.get_variable(
                        'stype_embs', [len(self.data.id2stype), self.model_config.dimension], tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

                # Mask for only predict candidate senses
                np_mask = np.loadtxt(self.model_config.abbr_mask_file)
                self.mask_embs = tf.convert_to_tensor(np_mask, dtype=tf.float32)

            # Use tf.hub text modeling
            self.embed_hub_module = None
            if self.model_config.hub_module_embedding:
                self.embed_hub_module = hub.Module(
                    'https://tfhub.dev/google/universal-sentence-encoder-large/3', name='embed_hub')

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_id in range(self.model_config.num_gpus):
                    with tf.device('/device:GPU:%d' % gpu_id):
                        with tf.name_scope('%s_%d' % ('gpu_scope', gpu_id)):
                            loss, obj = self.create_model()
                            print('Built Model for GPU%s' % gpu_id)
                            grad = optim.compute_gradients(loss)
                            print('Built Grads for GPU%s' % gpu_id)
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
                else:
                    self.losses_eval = losses[0] # In eval, single cpu/gpu is used.

                self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def create_model(self):
        with tf.variable_scope('variables'):
            contexts_inp = []
            for _ in range(self.model_config.max_context_len):
                contexts_inp.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='context_input'))

            abbr_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='abbr_input')
            sense_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='sense_input')

            # Generate mask that masks the candidate sense to be predicted as 1 and others to 0, mask embedding is a one-hot matrix of [num_abbr, num_sense]
            # The mask is 2 dimension vector with size [batch_size, sense_size]
            mask = tf.nn.embedding_lookup(self.mask_embs, abbr_inp)

            if self.model_config.hub_module_embedding:
                text_input = tf.zeros([self.model_config.batch_size], tf.string, name='text_input')

            if 'def' in self.model_config.extra_loss:
                defs = []
                for _ in range(self.model_config.max_def_len):
                    defs.append(
                        tf.zeros(self.model_config.batch_size, tf.int32, name='def_input'))

            if 'stype' in self.model_config.extra_loss:
                stype_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='stype_input')

        with tf.variable_scope('model'):
            context_encoder = ContextEncoder(self.is_train, self.model_config, self.data, self.embs)

        with tf.variable_scope('pred'):
            # tf.hub text modeling always has 512 dimension vector
            project_size = self.model_config.dimension + (512 if self.model_config.hub_module_embedding else 0)
            sense_embs = tf.get_variable('proj_w', [project_size, self.data.sen_cnt], tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            if self.model_config.predict_mode == 'clas':
                sense_bias = tf.get_variable('proj_b', [self.data.sen_cnt], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())

            if self.model_config.abbr_mode == 'abbr':
                # learn a new embedding for each abbr
                abbr_inp_emb = self.embedding_fn(abbr_inp, self.abbr_embs)
            elif self.model_config.abbr_mode == 'sense':
                # take the weighted-averaged sense embedding of this abbr as the abbr embedding
                sense_weight = tf.get_variable('sense_weight',
                                               [1, 1, self.data.sen_cnt], tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
                mask_exp = tf.expand_dims(mask, 1)
                abbr_inp_emb = tf.reduce_mean(tf.expand_dims(sense_embs, 0) * mask_exp * sense_weight, axis=-1)
            else:
                raise ValueError('Unsupported abbr mode.')

            # get the embedding of abbr word
            abbr_inp_emb = tf.expand_dims(abbr_inp_emb, axis=1)
            # get the embedding of context words
            contexts_emb = tf.stack(context_encoder.embed_context(contexts_inp, abbr_inp_emb), axis=1)

            # get the output of transformer. If mode='match', the vector is the predicted sense embedding
            encoder_outputs, weights = context_encoder.context_encoder(contexts_emb, contexts_inp, abbr_inp_emb)
            # encoder_outputs = transformer.transformer_encoder_addabbr(
            #     contexts_emb, contexts_emb_bias, abbr_inp_emb, self.hparams)

            aggregate_state = self.get_aggregate_state(encoder_outputs)
            if self.model_config.hub_module_embedding:
                # Append embedding from hub text model
                embed_hub_state = self.embed_hub_module(text_input)
                aggregate_state = tf.concat([aggregate_state, embed_hub_state], axis=-1)

            if self.model_config.predict_mode == 'clas':
                # Instead mask logit, mask proj_w and proj_b for efficiency
                proj_w_stack = tf.stack([sense_embs for _ in range(self.model_config.batch_size)], axis=0)
                masked_proj_w = proj_w_stack * tf.expand_dims(mask, 1)
                proj_b_stack = tf.stack([sense_bias for _ in range(self.model_config.batch_size)], axis=0)
                masked_proj_b = proj_b_stack * mask
                logits = tf.squeeze(tf.matmul(tf.expand_dims(aggregate_state, axis=1), masked_proj_w), axis=1) + masked_proj_b
                # logits = tf.matmul(aggregate_state, proj_w) + proj_b
                # logits *= mask
            elif self.model_config.predict_mode == 'match':
                aggregate_state_exp = tf.expand_dims(aggregate_state, axis=-1)
                mask_exp = tf.expand_dims(mask, 1)
                cur_embs = tf.expand_dims(sense_embs, 0) * mask_exp
                logits = tf.reduce_sum(
                    cur_embs * tf.tile(
                        aggregate_state_exp, [1, 1, self.data.sen_cnt]),
                    1)
            else:
                raise ValueError("Unsupported prediction mode.")

            if self.model_config.pointer_mode:
                ptr_network = PointerNetwork(self.is_train, self.model_config, self.data, weights, contexts_inp)
                if self.model_config.pointer_mode == 'first_dist':
                    ptr_logits = ptr_network.getLogitFromFirstSelfAttnDist()
                    # TODO(sanqiang): ptr logit
                    print('Use Ptr Network with first attn distribution.')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_inp)
            loss_mask = tf.to_float(tf.not_equal(sense_inp, 0))
            loss *= loss_mask

            pred = tf.nn.top_k(logits, k=5, sorted=True)[1]
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('extra'):
            if self.is_train:
                self.def_loss = tf.constant(0.0)
                if 'def' in self.model_config.extra_loss:
                    aggregate_state_def = tf.contrib.layers.fully_connected(
                        aggregate_state, self.model_config.dimension)
                    defs_stack = tf.stack(defs, axis=1)
                    defs_embed = self.embedding_fn(defs_stack, self.embs)
                    defs_bias = common_attention.attention_bias_ignore_padding(
                        tf.to_float(tf.equal(defs_stack,
                                             self.data.voc.encode(constant.PAD))))
                    defs_embed = tf.nn.dropout(defs_embed,
                                               1.0 - self.hparams.layer_prepostprocess_dropout)
                    defs_output = transformer.transformer_encoder(
                        defs_embed, defs_bias, self.hparams)
                    defs_output = tf.reduce_mean(defs_output, axis=1)
                    self.def_loss = tf.losses.absolute_difference(defs_output, aggregate_state_def)
                    loss += self.def_loss

                self.style_loss = tf.constant(0.0)
                if 'stype' in self.model_config.extra_loss:
                    aggregate_state_stype = tf.contrib.layers.fully_connected(
                        aggregate_state, self.model_config.dimension)
                    style_emb = self.embedding_fn(stype_inp, self.stype_embs)
                    self.style_loss = tf.losses.absolute_difference(style_emb, aggregate_state_stype)
                    loss += self.style_loss

        obj = {
            'contexts': contexts_inp,
            'abbr_inp': abbr_inp,
            'sense_inp': sense_inp,
            'pred': pred,
        }
        if self.model_config.hub_module_embedding:
            obj['text_input'] = text_input

        if 'def' in self.model_config.extra_loss:
            obj['def'] = defs

        if 'stype' in self.model_config.extra_loss:
            obj['stype'] = stype_inp

        return loss, obj
