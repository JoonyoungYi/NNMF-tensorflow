import os
import time

import tensorflow as tf

from .utils.mlp import build as build_mlp
from .utils.dataset import get_N_and_M


def _init_model_file_path(kind):
    folder_path = 'logs/{}'.format(int(time.time() * 1000))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return os.path.join(folder_path, 'model.ckpt')


class NNMF(object):
    def __init__(self,
                 kind,
                 D=10,
                 D_prime=60,
                 K=1,
                 hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0,
                                            'stddev': 0.1},
                 lambda_value=0.01):
        self.lambda_value = lambda_value
        self.N, self.M = get_N_and_M(kind)
        self.D = D
        self.D_prime = D_prime
        self.K = K
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params
        self.model_file_path = _init_model_file_path(kind)

        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Call methods to initialize variables and operations (to be implemented by children)
        self._init_vars()
        self._init_ops()

        # RMSE
        self.rmse = tf.sqrt(
            tf.reduce_mean(tf.square(tf.subtract(self.r, self.r_target))))

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _init_vars(self):
        # Latents
        self.U = tf.Variable(
            tf.truncated_normal([self.N, self.D], **
                                self.latent_normal_init_params))
        self.U_prime = tf.Variable(
            tf.truncated_normal([self.N, self.D_prime, self.K], **
                                self.latent_normal_init_params))
        self.V = tf.Variable(
            tf.truncated_normal([self.M, self.D], **
                                self.latent_normal_init_params))
        self.V_prime = tf.Variable(
            tf.truncated_normal([self.M, self.D_prime, self.K], **
                                self.latent_normal_init_params))

        # Lookups
        self.U_lookup = tf.nn.embedding_lookup(self.U, self.user_index)
        self.U_prime_lookup = tf.nn.embedding_lookup(self.U_prime,
                                                     self.user_index)
        self.V_lookup = tf.nn.embedding_lookup(self.V, self.item_index)
        self.V_prime_lookup = tf.nn.embedding_lookup(self.V_prime,
                                                     self.item_index)

        # MLP ("f")
        prime = tf.reduce_sum(
            tf.multiply(self.U_prime_lookup, self.V_prime_lookup), axis=2)
        f_input_layer = tf.concat(
            values=[self.U_lookup, self.V_lookup, prime], axis=1)

        _r, self.mlp_weights = build_mlp(
            f_input_layer,
            hidden_unit_number=self.hidden_units_per_layer,
            output_unit_number=1,
            hidden_layer_number=3)
        # self.r = _r
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(
            tf.square(tf.subtract(self.r_target, self.r)),
            reduction_indices=[0])
        regularizer_loss = tf.add_n([
            tf.reduce_sum(tf.square(self.U_prime)),
            tf.reduce_sum(tf.square(self.U)),
            tf.reduce_sum(tf.square(self.V)),
            tf.reduce_sum(tf.square(self.V_prime))
        ])
        self.loss = reconstruction_loss + (
            self.lambda_value * regularizer_loss)

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(1e-3)

        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(
            self.loss, var_list=self.mlp_weights)
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(
            self.loss, var_list=[self.U, self.U_prime, self.V, self.V_prime])

        self.optimize_steps = [f_train_step, latent_train_step]

    def train_iteration(self, data, additional_feed=None):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {
            self.user_index: user_ids,
            self.item_index: item_ids,
            self.r_target: ratings
        }

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._iters += 1

    def eval_loss(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {
            self.user_index: user_ids,
            self.item_index: item_ids,
            self.r_target: ratings
        }
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(
            self.r,
            feed_dict={self.user_index: [user_id],
                       self.item_index: [item_id]})
        return rating[0]

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {
            self.user_index: user_ids,
            self.item_index: item_ids,
            self.r_target: ratings
        }
        return self.sess.run(self.rmse, feed_dict=feed_dict)
