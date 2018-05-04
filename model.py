import math

import tensorflow as tf

from config import *


def _get_random_abs_max_val(i_node_number, o_node_number):
    return 4.0 * math.sqrt(6.0) / math.sqrt(i_node_number + o_node_number)


def _build_mlp(theta, Ws, bs):
    layer = theta
    for W, b in zip(Ws[:-1], bs[:-1]):
        layer = tf.nn.sigmoid(tf.matmul(layer, W) + b)
    return tf.matmul(layer, Ws[-1]) + bs[-1]


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

        random_abs_max_val = _get_random_abs_max_val(i_node_number,
                                                     o_node_number)
        Ws.append(
            tf.Variable(
                tf.random_uniform(
                    [i_node_number, o_node_number],
                    minval=-random_abs_max_val,
                    maxval=random_abs_max_val)))
        bs.append(tf.Variable(tf.zeros([o_node_number])))
    X_hat = _build_mlp(theta, Ws, bs)  # predicted rating of our network.

    # Define Losses
    loss = tf.reduce_sum(tf.square(tf.subtract(X, X_hat))) + LAMBDA * tf.add_n(
        [
            tf.reduce_sum(tf.square(U)),
            tf.reduce_sum(tf.square(V)),
            tf.reduce_sum(tf.square(U_prime)),
            tf.reduce_sum(tf.square(V_prime))
        ])
    trains = [
        tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(
            loss, var_list=[U, U_prime, V, V_prime]),
        tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(
            loss, var_list=Ws + bs)
    ]
    # train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
    #     loss, var_list=[U, U_prime, V, V_prime] + Ws + bs)
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X, X_hat))))

    return {
        'X': X,
        'trains': trains,
        'loss': loss,
        'RMSE': RMSE,
        'user_ids': user_ids,
        'item_ids': item_ids,
    }
