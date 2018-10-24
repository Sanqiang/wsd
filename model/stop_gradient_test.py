import tensorflow as tf
import numpy as np

def entry_stop_gradients(target, mask):
    '''
    Copied from https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
    Mask the gradients of specific entries from target
    :param target: input tensor
    :param mask: matrix mask, 1 denotes to which entry I would like to apply gradient, 0 denotes to which entry I don't want to apply gradient(set gradient to 0)
    :return:
        a tensor whose shape and value is same to target, but only entries where mask value is 1 are allowed to apply gradient
    '''
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target

mask = np.array([1., 0, 1, 1, 0, 0, 1, 1, 0, 1])
# mask_h = np.abs(mask - 1)

emb = tf.constant(np.ones([10, 5]))

matrix = entry_stop_gradients(emb, tf.expand_dims(mask, 1))

parm = np.random.randn(5, 1)
t_parm = tf.constant(parm)

loss = tf.reduce_sum(tf.matmul(matrix, t_parm))

# loss to emb has no grad
grad1 = tf.gradients(loss, emb)
# loss to matrix still has grad
grad2 = tf.gradients(loss, matrix)
print(matrix)
with tf.Session() as sess:
    print(sess.run(loss))
    print(sess.run([grad1]))
    print(sess.run([grad2]))
