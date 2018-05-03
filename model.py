import math

import tensorflow as tf

from config import *


def _get_random_abs_max_val(i_node_number, o_node_number):
    return 4.0 * math.sqrt(6.0) / math.sqrt(i_node_number + o_node_number)


def _build_mlp(theta, Ws, bs):
    layer = theta
    for W, b in zip(Ws[:-1], bs[:-1]):
        layer = tf.nn.sigmoid(tf.matmul(layer, W) + b)
    return tf.nn.sigmoid(tf.matmul(layer, Ws[-1]) + bs[-1])


def init_models(user_number, item_number):
    # Latent Variables
    U = tf.Variable(
        tf.random_normal(
            [user_number, D], mean=INITIAL_MEAN, stddev=INITIAL_STDDEV))
    U_prime = tf.Variable(
        tf.random_normal(
            [user_number, D_prime], mean=INITIAL_MEAN, stddev=INITIAL_STDDEV))
    V = tf.Variable(
        tf.random_normal(
            [item_number, D], mean=INITIAL_MEAN, stddev=INITIAL_STDDEV))
    V_prime = tf.Variable(
        tf.random_normal(
            [item_number, D_prime], mean=INITIAL_MEAN, stddev=INITIAL_STDDEV))
    # QUESTION: U_prime과 V_prime은 왜 등장하는 것인가?

    # Lookups
    # QUESTION: embedding_sparse_lookup 쓰면 더 퍼포먼스가 좋아지지 않을까?
    user_ids = tf.placeholder(tf.int32, [None])
    item_ids = tf.placeholder(tf.int32, [None])
    X = tf.placeholder(tf.float32, [None])  # Real rating
    U_lookup = tf.nn.embedding_lookup(U, user_ids)

    U_prime_lookup = tf.nn.embedding_lookup(U_prime, user_ids)
    V_lookup = tf.nn.embedding_lookup(V, item_ids)
    V_prime_lookup = tf.nn.embedding_lookup(V_prime, item_ids)
    theta = tf.concat(
        [U_lookup, V_lookup,
         tf.multiply(U_prime_lookup, V_prime_lookup)],
        axis=1)

    # MLP(Multi Layer Perceptrons)
    Ws, bs = [], []
    for layer_idx in range(HIDDEN_LAYER_NUMBER):
        if layer_idx == 0:
            i_node_number = int(theta.shape[1])
            o_node_number = HIDDEN_LAYER_NODE_NUMBER
        elif layer_idx == HIDDEN_LAYER_NUMBER - 1:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = 1
        else:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = HIDDEN_LAYER_NODE_NUMBER

        Ws.append(
            tf.Variable(
                tf.random_uniform(
                    [i_node_number, o_node_number],
                    minval=-_get_random_abs_max_val(i_node_number,
                                                    o_node_number),
                    maxval=_get_random_abs_max_val(i_node_number,
                                                   o_node_number))))
        bs.append(tf.Variable(tf.zeros([o_node_number])))
    X_hat = _build_mlp(theta, Ws, bs)  # predicted rating of our network.

    # Define Losses
    loss = tf.nn.l2_loss(tf.subtract(X, X_hat)) + LAMBDA * (
        tf.norm(U_lookup, ord='fro') + tf.norm(V_lookup, ord='fro') + tf.norm(
            U_prime_lookup, ord='fro') + tf.norm(V_prime_lookup, ord='fro'))
    train = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss, var_list=[U, U_prime, V, V_prime] + Ws + bs)
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X, X_hat))))

    return {
        'X': X,
        'train': train,
        'loss': loss,
        'RMSE': RMSE,
        'user_ids': user_ids,
        'item_ids': item_ids,
    }


# import tensorflow as tf
#
# from utils import KL, build_mlp, get_kl_weight
#
#
# class _NNMFBase(object):
#     def __init__(self,
#                  num_users,
#                  num_items,
#                  D=10,
#                  Dprime=60,
#                  hidden_units_per_layer=50,
#                  latent_normal_init_params={'mean': 0.0,
#                                             'stddev': 0.1},
#                  model_filename='model/nnmf.ckpt'):
#         self.num_users = num_users
#         self.num_items = num_items
#         self.D = D
#         self.Dprime = Dprime
#         self.hidden_units_per_layer = hidden_units_per_layer
#         self.latent_normal_init_params = latent_normal_init_params
#         self.model_filename = model_filename
#
#         # Internal counter to keep track of current iteration
#         self._iters = 0
#
#         # Input
#         self.user_index = tf.placeholder(tf.int32, [None])
#         self.item_index = tf.placeholder(tf.int32, [None])
#         self.r_target = tf.placeholder(tf.float32, [None])
#
#         # Call methods to initialize variables and operations (to be implemented by children)
#         self._init_vars()
#         self._init_ops()
#
#         # RMSE
#         self.rmse = tf.sqrt(
#             tf.reduce_mean(tf.square(tf.sub(self.r, self.r_target))))
#
#     def _train_iteration(self, data, additional_feed=None):
#         user_ids = data['user_id']
#         item_ids = data['item_id']
#         ratings = data['rating']
#
#         feed_dict = {
#             self.user_index: user_ids,
#             self.item_index: item_ids,
#             self.r_target: ratings
#         }
#
#         if additional_feed:
#             feed_dict.update(additional_feed)
#
#         for step in self.optimize_steps:
#             self.sess.run(step, feed_dict=feed_dict)
#
#         self._iters += 1
#
#     def train_iteration(self, data):
#         self._train_iteration(data)
#
#     def eval_loss(self, data):
#         raise NotImplementedError
#
#     def eval_rmse(self, data):
#         user_ids = data['user_id']
#         item_ids = data['item_id']
#         ratings = data['rating']
#
#         feed_dict = {
#             self.user_index: user_ids,
#             self.item_index: item_ids,
#             self.r_target: ratings
#         }
#         return self.sess.run(self.rmse, feed_dict=feed_dict)
#
#     def predict(self, user_id, item_id):
#         rating = self.sess.run(
#             self.r,
#             feed_dict={self.user_index: [user_id],
#                        self.item_index: [item_id]})
#         return rating[0]
#
#
# class NNMF(_NNMFBase):
#     def __init__(self, *args, **kwargs):
#         if 'lam' in kwargs:
#             self.lam = float(kwargs['lam'])
#             del kwargs['lam']
#         else:
#             self.lam = 0.01
#
#         super(NNMF, self).__init__(*args, **kwargs)
#
#     def _init_vars(self):
#         # Latents
#         self.U = tf.Variable(
#             tf.truncated_normal([self.num_users, self.D], **
#                                 self.latent_normal_init_params))
#         self.Uprime = tf.Variable(
#             tf.truncated_normal([self.num_users, self.Dprime], **
#                                 self.latent_normal_init_params))
#         self.V = tf.Variable(
#             tf.truncated_normal([self.num_items, self.D], **
#                                 self.latent_normal_init_params))
#         self.Vprime = tf.Variable(
#             tf.truncated_normal([self.num_items, self.Dprime], **
#                                 self.latent_normal_init_params))
#
#         # Lookups
#         self.U_lu = tf.nn.embedding_lookup(self.U, self.user_index)
#         self.Uprime_lu = tf.nn.embedding_lookup(self.Uprime, self.user_index)
#         self.V_lu = tf.nn.embedding_lookup(self.V, self.item_index)
#         self.Vprime_lu = tf.nn.embedding_lookup(self.Vprime, self.item_index)
#
#         # MLP ("f")
#         f_input_layer = tf.concat(
#             concat_dim=1,
#             values=[
#                 self.U_lu, self.V_lu,
#                 tf.mul(self.Uprime_lu, self.Vprime_lu)
#             ])
#
#         _r, self.mlp_weights = build_mlp(
#             f_input_layer, hidden_units_per_layer=self.hidden_units_per_layer)
#         self.r = tf.squeeze(_r, squeeze_dims=[1])
#
#     def _init_ops(self):
#         # Loss
#         reconstruction_loss = tf.reduce_sum(
#             tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])
#         reg = tf.add_n([
#             tf.reduce_sum(tf.square(self.Uprime), reduction_indices=[0, 1]),
#             tf.reduce_sum(tf.square(self.U), reduction_indices=[0, 1]),
#             tf.reduce_sum(tf.square(self.V), reduction_indices=[0, 1]),
#             tf.reduce_sum(tf.square(self.Vprime), reduction_indices=[0, 1])
#         ])
#         self.loss = reconstruction_loss + (self.lam * reg)
#
#         # Optimizer
#         self.optimizer = tf.train.AdamOptimizer()
#         # Optimize the MLP weights
#         f_train_step = self.optimizer.minimize(
#             self.loss, var_list=self.mlp_weights.values())
#         # Then optimize the latents
#         latent_train_step = self.optimizer.minimize(
#             self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])
#
#         self.optimize_steps = [f_train_step, latent_train_step]
#
#     def eval_loss(self, data):
#         user_ids = data['user_id']
#         item_ids = data['item_id']
#         ratings = data['rating']
#
#         feed_dict = {
#             self.user_index: user_ids,
#             self.item_index: item_ids,
#             self.r_target: ratings
#         }
#         return self.sess.run(self.loss, feed_dict=feed_dict)