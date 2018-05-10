import math

import tensorflow as tf


def _get_weight_init_range(n_in, n_out):
    """
        Calculates range for picking initial weight values from a uniform distribution.
    """
    return 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)


def build(layer,
          training,
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

    unit_numbers = [hidden_unit_number] * (
        hidden_layer_number - 1) + [output_unit_number]
    for i, unit_number in enumerate(unit_numbers):
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

        layer = tf.matmul(layer, W) + b
        if i < len(unit_numbers) - 1:
            # layer = tf.layers.batch_normalization(layer, training=training)
            layer = activation(layer)
            # layer = tf.nn.dropout(layer, 0.5)
        else:
            if final_activation:
                layer = final_activation(layer)
        prev_layer_unit_number = unit_number

    return layer, Ws + bs
