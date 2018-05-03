import tensorflow as tf
import numpy as np

import dataset
from model import init_models
from config import *


def main(session):
    # Define computation graph & Initialize
    print('Building network & initializing variables')
    dataset.init_data_sets()

    models = init_models(user_number=943, item_number=1682)

    session.run(tf.global_variables_initializer())

    train_user_ids, train_item_ids, train_xs = dataset.get_train_data()
    test_user_ids, test_item_ids, test_xs = dataset.get_test_data()

    for epoch_idx in range(EPOCH_NUMBER):
        batch_user_ids, batch_item_ids, batch_xs = dataset.next_batch()
        for batch_idx in range(dataset.get_total_batch_number()):
            _ = session.run(
                (models['loss'], models['train']),
                feed_dict={
                    models['X']: batch_xs,
                    models['user_ids']: batch_user_ids,
                    models['item_ids']: batch_item_ids
                }, )

        train_rmse, train_loss = session.run(
            (models['RMSE'], models['loss']),
            feed_dict={
                models['X']: train_xs,
                models['user_ids']: train_user_ids,
                models['item_ids']: train_item_ids
            }, )

        test_rmse, test_loss = session.run(
            (models['RMSE'], models['loss']),
            feed_dict={
                models['X']: test_xs,
                models['user_ids']: test_user_ids,
                models['item_ids']: test_item_ids
            }, )
        print('>>', epoch_idx, train_loss, train_rmse, test_loss, test_rmse)


if __name__ == '__main__':
    with tf.Session() as session:
        main(session)
