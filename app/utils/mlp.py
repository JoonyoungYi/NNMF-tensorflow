import math

import tensorflow as tf


def _get_weight_init_range(n_in, n_out):
    """
        Calculates range for picking initial weight values from a uniform distribution.
    """
    return 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)


def build(layer,
          hidden_unit_number,
          hidden_layer_number,
          output_unit_number,
          activation=tf.nn.sigmoid,
          final_activation=tf.nn.sigmoid):
    """
        Builds a feed-forward NN (MLP)
    """
    prev_layer_unit_number = layer.get_shape().as_list()[1]
    Ws, bs = [], []
    for unit_number in [hidden_unit_number] * (
            hidden_layer_number - 1) + [output_unit_number]:
        # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
        range = _get_weight_init_range(prev_layer_unit_number, unit_number)
        W = tf.Variable(
            tf.random_uniform(
                [prev_layer_unit_number, unit_number],
                minval=-range,
                maxval=range))
        b = tf.Variable(tf.zeros([unit_number]))
        Ws.append(W)
        bs.append(b)
        layer = activation(tf.matmul(layer, W) + b)

        prev_layer_unit_number = unit_number

    return layer * 4 + 1, Ws + bs
