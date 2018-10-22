import tensorflow as tf
from tensor2tensor.models import transformer
from util import constant
from tensor2tensor.layers import common_attention, common_layers
from tensor2tensor.models.research import universal_transformer_util, universal_transformer
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

        if self.model_config.encoder_mode == 'ut2t':
            # hparams for universal t2t
            self.hparams = universal_transformer.update_hparams_for_universal_transformer(self.hparams)
            self.hparams.recurrence_type = "act"

    def get_aggregate_state(self, encoder_outputs, bias_mask):
        bias_cnt = 1.0 + tf.expand_dims(tf.reduce_sum(bias_mask, axis=-1), axis=-1)
        bias_mask = tf.expand_dims(bias_mask, axis=-1)
        if 'selfattn' in self.model_config.aggregate_mode:
            selfattn_w = tf.get_variable(
                'selfattn_w', [1, self.model_config.dimension, 1], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            selfattn_b = tf.get_variable(
                'selfattn_b', [1, 1, 1], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            weight = tf.nn.tanh(tf.nn.conv1d(encoder_outputs, selfattn_w, 1, 'SAME') + selfattn_b)
            encoder_outputs *= weight
        aggregate_state = tf.reduce_sum(encoder_outputs*bias_mask, axis=1)
        aggregate_state /= bias_cnt
        return aggregate_state


class ContextEncoder(BaseGraph):
    def __init__(self, is_train, model_config, data, embs):
        BaseGraph.__init__(self, is_train, model_config, data)
        self.embs = embs

    def embed_context(self, contexts):
        with tf.variable_scope('context_embed'):
            contexts_emb = self.embedding_fn(contexts, self.embs)
            return contexts_emb

    def context_encoder(self, contexts_emb, contexts, abbr_inp_emb=None):
        weights = {}
        contexts_bias = common_attention.attention_bias_ignore_padding(
            tf.to_float(tf.equal(tf.stack(contexts, axis=1),
                                 self.data.voc.encode(constant.PAD))))
        contexts_emb = tf.nn.dropout(contexts_emb,
                                     1.0 - self.hparams.layer_prepostprocess_dropout)
        # encoder_ouput = transformer.transformer_encoder_abbr(
        #     contexts_emb, contexts_bias, abbr_inp_emb,
        #     tf.zeros([self.model_config.batch_size,1,1,1]), self.hparams,
        #     save_weights_to=weights)
        if self.model_config.encoder_mode == 't2t':
            encoder_ouput = transformer.transformer_encoder(
                contexts_emb, contexts_bias, self.hparams,
                save_weights_to=weights)
        elif self.model_config.encoder_mode == 'ut2t':
            encoder_ouput, _ = universal_transformer_util.universal_transformer_encoder(
                contexts_emb, contexts_bias, self.hparams,
                save_weights_to=weights)
        else:
            raise ValueError('Unknow encoder_mode.')
        return encoder_ouput, weights


# TODO(sanqiang): PTR
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
            optim = self.get_optim()
            self.objs = []

            with tf.variable_scope('shared'):
                with tf.device('/cpu:0'):
                    # Vocab embedding
                    self.embs = tf.get_variable(
                        'embs', [self.data.voc.vocab_size(), self.model_config.dimension], tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=self.model_config.train_emb) # tf.random_uniform_initializer(-0.1, 0.1)
                    if self.model_config.init_emb:
                        print('Use init embedding from %s' % self.model_config.init_emb)
                        self.embs_init_fn = self.embs.assign(
                            tf.convert_to_tensor(np.load(self.model_config.init_emb), dtype=tf.float32))

                    # Mask for only predict candidate senses
                    np_mask = np.loadtxt(self.model_config.abbr_mask_file)
                    self.mask_embs = tf.convert_to_tensor(np_mask, dtype=tf.float32)

                with tf.device('/gpu:0'):
                    self.global_step = tf.train.get_or_create_global_step()
                    self.increment_global_step = tf.assign_add(self.global_step, 1)

                    self.global_step_task = tf.get_variable(
                        'global_step_task', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

                    # tf.hub text modeling always has 512 dimension vector
                    project_size = self.model_config.dimension + (512 if self.model_config.hub_module_embedding else 0)
                    self.sense_embs = tf.get_variable('proj_w', [project_size, self.data.sen_cnt], tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer()) # tf.random_uniform_initializer(-0.1, 0.1)
                    if self.model_config.predict_mode == 'clas':
                        self.sense_bias = tf.get_variable('proj_b', [self.data.sen_cnt], tf.float32,
                                                          initializer=tf.contrib.layers.xavier_initializer())

                # Use tf.hub text modeling
                self.embed_hub_module = None
                if self.model_config.hub_module_embedding:
                    import tensorflow_hub as hub
                    self.embed_hub_module = hub.Module(
                        'https://tfhub.dev/google/universal-sentence-encoder-large/3', name='embed_hub')

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_id in range(self.model_config.num_gpus):
                    with tf.device('/device:GPU:%d' % gpu_id):
                        with tf.name_scope('%s_%d' % ('gpu_scope', gpu_id)):
                            loss, obj = self.create_model()
                            print('Built Model for GPU%s' % gpu_id)
                            tf.get_variable_scope().reuse_variables()
                            losses.append(loss)
                            self.objs.append(obj)

            with tf.device('/gpu:0'):
                # Add graph for cui
                if self.model_config.extra_loss and self.is_train:
                    self.create_model_cui()

            with tf.variable_scope('optimization'):
                self.loss = tf.divide(tf.add_n(losses), self.model_config.num_gpus)
                self.perplexity = tf.exp(tf.reduce_mean(self.loss))

                if self.is_train:
                    self.train_op = optim.minimize(self.loss)
                    self.increment_global_step_task = tf.assign_add(
                        self.global_step_task, 1)
                else:
                    self.losses_eval = losses[0] # In eval, single cpu/gpu is used.

                self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def get_logits(self, query_vector, cand_mask):
        def entry_stop_gradients(target, mask):
            # Copied from https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
            mask_inverse = tf.abs(mask - 1)
            return tf.stop_gradient(mask_inverse * target) + mask * target

        if self.model_config.predict_mode == 'clas':
            # Deprecated
            proj_w_stack = entry_stop_gradients(
                tf.stack([self.sense_embs for _ in range(self.model_config.batch_size)], axis=0),
                tf.expand_dims(cand_mask, 1))
            proj_b_stack = entry_stop_gradients(
                tf.stack([self.sense_bias for _ in range(self.model_config.batch_size)], axis=0),
                cand_mask)
            logits = tf.squeeze(tf.matmul(tf.expand_dims(query_vector, axis=1), proj_w_stack),
                                axis=1) + proj_b_stack
            raise NotImplementedError('CLAS mode is not finished')
        elif self.model_config.predict_mode == 'match':
            aggregate_state_exp = tf.expand_dims(query_vector, axis=-1)
            cur_embs = tf.expand_dims(self.sense_embs, 0)
            logits = tf.reduce_sum(
                cur_embs * tf.tile(
                    aggregate_state_exp, [1, 1, self.data.sen_cnt]), 1)
            logits = entry_stop_gradients(logits, cand_mask)
        else:
            raise ValueError("Unsupported prediction mode.")
        return logits

    def create_model(self):
        with tf.variable_scope('task'):
            with tf.variable_scope('variables'):
                contexts = []
                for _ in range(self.model_config.max_context_len):
                    contexts.append(
                        tf.zeros(self.model_config.batch_size, tf.int32, name='context_input'))

                abbr_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='abbr_input')
                sense_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='sense_input')

                if self.model_config.hub_module_embedding:
                    text_input = tf.zeros([self.model_config.batch_size], tf.string, name='text_input')

                # Abbr embedding
                if self.model_config.abbr_mode == 'abbr':
                    self.abbr_embs = tf.get_variable(
                        'abbr_embs', [len(self.data.id2abbr), self.model_config.dimension], tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer()) # tf.random_uniform_initializer(-0.08, 0.08)

                if self.model_config.lm_mask_rate:
                    masked_contexts = []
                    for _ in range(self.model_config.max_context_len):
                        masked_contexts.append(
                            tf.zeros(self.model_config.batch_size, tf.int32, name='masked_contexts_input'))

                    masked_words = []
                    for _ in range(self.model_config.max_subword_len):
                        masked_words.append(
                            tf.zeros(self.model_config.batch_size, tf.int32, name='masked_words_input'))

                # Generate mask that mask the candidate sense to be predicted as 1 and others to 0
                # The mask is 2 dimension vector with size [batch_size, sense_size]
                mask = tf.nn.embedding_lookup(self.mask_embs, abbr_inp)

            with tf.variable_scope('model'):
                context_encoder = ContextEncoder(self.is_train, self.model_config, self.data, self.embs)

            with tf.variable_scope('pred'):
                if self.model_config.abbr_mode == 'abbr':
                    abbr_inp_emb = self.embedding_fn(abbr_inp, self.abbr_embs)
                elif self.model_config.abbr_mode == 'sense':
                    sense_weight = tf.get_variable('sense_weight',
                                                   [1, 1, self.data.sen_cnt], tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    abbr_inp_emb = tf.reduce_mean(tf.expand_dims(self.sense_embs, 0) * sense_weight, axis=-1)
                else:
                    raise ValueError('Unsupported abbr mode.')

                abbr_inp_emb = tf.expand_dims(abbr_inp_emb, axis=1)
                contexts_emb = tf.stack(context_encoder.embed_context(contexts), axis=1)

                encoder_outputs, weights = context_encoder.context_encoder(contexts_emb, contexts, abbr_inp_emb)
                bias_mask = tf.to_float(tf.not_equal(tf.stack(contexts, axis=1), self.data.voc.encode(constant.PAD)))
                aggregate_state = self.get_aggregate_state(encoder_outputs, bias_mask)
                if self.model_config.hub_module_embedding:
                    # Append embedding from hub text model
                    embed_hub_state = self.embed_hub_module(text_input)
                    aggregate_state = tf.concat([aggregate_state, embed_hub_state], axis=-1)

                logits = self.get_logits(aggregate_state, mask)
                mask_inverse = tf.abs(mask - 1) * -1e12
                logits = logits + mask_inverse

                if self.model_config.pointer_mode:
                    ptr_network = PointerNetwork(self.is_train, self.model_config, self.data, weights, contexts)
                    if self.model_config.pointer_mode == 'first_dist':
                        ptr_logits = ptr_network.getLogitFromFirstSelfAttnDist()
                        # TODO(sanqiang): ptr logit
                        print('Use Ptr Network with first attn distribution.')

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_inp)

                if self.model_config.lm_mask_rate:
                    tf.get_variable_scope().reuse_variables()
                    masked_contexts_emb = tf.stack(context_encoder.embed_context(masked_contexts), axis=1)
                    masked_contexts_outputs, _ = context_encoder.context_encoder(masked_contexts_emb, masked_contexts)
                    masked_contexts_output = self.get_aggregate_state(
                        masked_contexts_outputs, tf.to_float(
                            tf.not_equal(tf.stack(masked_contexts, axis=1), self.data.voc.encode(constant.PAD))))

                    tf.get_variable_scope().reuse_variables()
                    masked_words_emb = tf.stack(context_encoder.embed_context(masked_words), axis=1)
                    masked_words_outputs, _ = context_encoder.context_encoder(masked_words_emb, masked_words)
                    masked_words_output = self.get_aggregate_state(
                        masked_words_outputs, tf.to_float(
                            tf.not_equal(tf.stack(masked_words, axis=1), self.data.voc.encode(constant.PAD))))

                    loss_lm = -tf.reduce_mean(tf.reduce_sum(masked_contexts_output*masked_words_output, axis=-1))
                    loss += loss_lm

                if not self.is_train:

                    if self.model_config.mode == 'dummy':
                        pred = tf.nn.top_k(logits, k=3, sorted=True)[1]
                    else:
                        pred = tf.nn.top_k(logits, k=5, sorted=True)[1]
                tf.get_variable_scope().reuse_variables()

        obj = {
            'contexts': contexts,
            'abbr_inp': abbr_inp,
            'sense_inp': sense_inp,
        }
        if not self.is_train:
            obj['pred'] = pred
        if self.model_config.hub_module_embedding:
            obj['text_input'] = text_input
        if self.model_config.lm_mask_rate:
            obj['masked_words'] = masked_words
            obj['masked_contexts'] = masked_contexts

        return loss, obj

    def create_model_cui(self):
        assert self.model_config.extra_loss
        self.global_step_cui = tf.get_variable(
            'global_step_cui', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

        with tf.variable_scope('cui'):
            # Semantic type embedding
            if 'stype' in self.model_config.extra_loss:
                self.stype_embs = tf.get_variable(
                    'stype_embs', [len(self.data.id2stype), self.model_config.dimension], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

            abbr_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='abbr_input')
            sense_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='sense_input')

            inputs = []
            if 'def' in self.model_config.extra_loss:
                defs = []
                for _ in range(self.model_config.max_def_len):
                    defs.append(
                        tf.zeros(self.model_config.batch_size, tf.int32, name='def_input'))

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
                inputs.append(defs_output)

            if 'stype' in self.model_config.extra_loss:
                stype_inp = tf.zeros(self.model_config.batch_size, tf.int32, name='stype_input')
                style_emb = self.embedding_fn(stype_inp, self.stype_embs)
                inputs.append(style_emb)

            if len(inputs) > 1:
                inputs = tf.concat(inputs, axis=1)
            elif len(inputs) == 1:
                inputs = inputs[0]
            aggregate_state = tf.contrib.layers.fully_connected(
                inputs, self.model_config.dimension, activation_fn=None)
            logits = self.get_logits(aggregate_state,
                                     tf.ones([self.model_config.batch_size, len(self.data.id2sense)]))
            self.loss_cui = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_inp)

            with tf.variable_scope('cui_optimization'):
                optim = self.get_optim()
                self.perplexity_cui = tf.exp(tf.reduce_mean(self.loss_cui))
                self.train_op_cui = optim.minimize(self.loss_cui)
                self.increment_global_step_cui = tf.assign_add(self.global_step_cui, 1)

            self.obj_cui = {
                'abbr_inp': abbr_inp,
                'sense_inp': sense_inp
            }

            if 'def' in self.model_config.extra_loss:
                self.obj_cui['def'] = defs

            if 'stype' in self.model_config.extra_loss:
                self.obj_cui['stype'] = stype_inp

