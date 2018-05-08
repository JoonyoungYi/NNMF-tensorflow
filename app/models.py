import tensorflow as tf

from .utils.mlp import build as build_mlp


class _NNMFBase(object):
    def __init__(self,
                 num_users,
                 num_items,
                 D=10,
                 Dprime=60,
                 hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0,
                                            'stddev': 0.1},
                 model_filename='logs/1/nnmf.ckpt'):
        self.num_users = num_users
        self.num_items = num_items
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params
        self.model_filename = model_filename

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

    def _init_vars(self):
        raise NotImplementedError

    def _init_ops(self):
        raise NotImplementedError

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _train_iteration(self, data, additional_feed=None):
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

    def train_iteration(self, data):
        self._train_iteration(data)

    def eval_loss(self, data):
        raise NotImplementedError

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

    def predict(self, user_id, item_id):
        rating = self.sess.run(
            self.r,
            feed_dict={self.user_index: [user_id],
                       self.item_index: [item_id]})
        return rating[0]


class NNMF(_NNMFBase):
    def __init__(self, *args, **kwargs):
        if 'lam' in kwargs:
            self.lam = float(kwargs['lam'])
            del kwargs['lam']
        else:
            self.lam = 0.01

        super(NNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Latents
        self.U = tf.Variable(
            tf.truncated_normal([self.num_users, self.D], **
                                self.latent_normal_init_params))
        self.Uprime = tf.Variable(
            tf.truncated_normal([self.num_users, self.Dprime], **
                                self.latent_normal_init_params))
        self.V = tf.Variable(
            tf.truncated_normal([self.num_items, self.D], **
                                self.latent_normal_init_params))
        self.Vprime = tf.Variable(
            tf.truncated_normal([self.num_items, self.Dprime], **
                                self.latent_normal_init_params))

        # Lookups
        self.U_lu = tf.nn.embedding_lookup(self.U, self.user_index)
        self.Uprime_lu = tf.nn.embedding_lookup(self.Uprime, self.user_index)
        self.V_lu = tf.nn.embedding_lookup(self.V, self.item_index)
        self.Vprime_lu = tf.nn.embedding_lookup(self.Vprime, self.item_index)

        # MLP ("f")
        f_input_layer = tf.concat(
            values=[
                self.U_lu, self.V_lu,
                tf.multiply(self.Uprime_lu, self.Vprime_lu)
            ],
            axis=1)

        _r, self.mlp_weights = build_mlp(
            f_input_layer, hidden_units_per_layer=self.hidden_units_per_layer)
        # self.r = _r
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(
            tf.square(tf.subtract(self.r_target, self.r)),
            reduction_indices=[0])
        reg = tf.add_n([
            tf.reduce_sum(tf.square(self.Uprime), reduction_indices=[0, 1]),
            tf.reduce_sum(tf.square(self.U), reduction_indices=[0, 1]),
            tf.reduce_sum(tf.square(self.V), reduction_indices=[0, 1]),
            tf.reduce_sum(tf.square(self.Vprime), reduction_indices=[0, 1])
        ])
        self.loss = reconstruction_loss + (self.lam * reg)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer()
        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(
            self.loss,
            var_list=[self.mlp_weights[key] for key in self.mlp_weights])
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(
            self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])

        self.optimize_steps = [f_train_step, latent_train_step]

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
