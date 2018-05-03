import math

import tensorflow as tf
import numpy as np

import dataset
from model import init_models
from config import *

# def load_data(train_filename,
#               valid_filename,
#               test_filename,
#               delimiter='\t',
#               col_names=['user_id', 'item_id', 'rating']):
#     """Helper function to load in/preprocess dataframes"""
#     train_data = pd.read_csv(
#         train_filename, delimiter=delimiter, header=None, names=col_names)
#     train_data['user_id'] = train_data['user_id'] - 1
#     train_data['item_id'] = train_data['item_id'] - 1
#     valid_data = pd.read_csv(
#         valid_filename, delimiter=delimiter, header=None, names=col_names)
#     valid_data['user_id'] = valid_data['user_id'] - 1
#     valid_data['item_id'] = valid_data['item_id'] - 1
#     test_data = pd.read_csv(
#         test_filename, delimiter=delimiter, header=None, names=col_names)
#     test_data['user_id'] = test_data['user_id'] - 1
#     test_data['item_id'] = test_data['item_id'] - 1
#
#     return train_data, valid_data, test_data
#
#
# def train(model, sess, saver, train_data, valid_data, batch_size, max_iters,
#           use_early_stop, early_stop_max_iter):
#     # Print initial values
#     batch = train_data.sample(batch_size) if batch_size else train_data
#     train_error = model.eval_loss(batch)
#     train_rmse = model.eval_rmse(batch)
#     valid_rmse = model.eval_rmse(valid_data)
#     print("{:3f} {:3f}, {:3f}".format(train_error, train_rmse, valid_rmse))
#
#     # Optimize
#     prev_valid_rmse = float("Inf")
#     early_stop_iters = 0
#     for i in xrange(max_iters):
#         # Run SGD
#         batch = train_data.sample(batch_size) if batch_size else train_data
#         model.train_iteration(batch)
#
#         # Evaluate
#         train_error = model.eval_loss(batch)
#         train_rmse = model.eval_rmse(batch)
#         valid_rmse = model.eval_rmse(valid_data)
#         print("{:3f} {:3f}, {:3f}".format(train_error, train_rmse, valid_rmse))
#
#         # Checkpointing/early stopping
#         if use_early_stop:
#             early_stop_iters += 1
#             if valid_rmse < prev_valid_rmse:
#                 prev_valid_rmse = valid_rmse
#                 early_stop_iters = 0
#                 saver.save(sess, model.model_filename)
#             elif early_stop_iters == early_stop_max_iter:
#                 print("Early stopping ({} vs. {})...".format(
#                     prev_valid_rmse, valid_rmse))
#                 break
#         else:
#             saver.save(sess, model.model_filename)
#
#
# def test(model, sess, saver, test_data, train_data=None, log=True):
#     if train_data:
#         train_rmse = model.eval_rmse(train_data)
#         if log:
#             print("Final train RMSE: {}".format(train_rmse))
#
#     test_rmse = model.eval_rmse(test_data)
#     if log:
#         print("Final test RMSE: {}".format(test_rmse))
#
#     return test_rmse


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
            loss_value, _ = session.run(
                (models['loss'], models['train']),
                feed_dict={
                    models['X']: batch_xs,
                    models['user_ids']: batch_user_ids,
                    models['item_ids']: batch_item_ids
                }, )
        train_rmse = session.run(
            models['RMSE'],
            feed_dict={
                models['X']: train_xs,
                models['user_ids']: train_user_ids,
                models['item_ids']: train_item_ids
            }, )

        test_rmse = session.run(
            models['RMSE'],
            feed_dict={
                models['X']: test_xs,
                models['user_ids']: test_user_ids,
                models['item_ids']: test_item_ids
            }, )
        print('>>', epoch_idx, loss_value, train_rmse, test_rmse)

    # assert False
    #
    # _, D_loss_value = session.run(
    #     (models['D_train'], models['D_loss']),
    #     feed_dict={
    #         models['X']: batch_xs,
    #         models['Z']: noise,
    #     }, )
    #
    # assert False
    # model = NNMF(num_users, num_items, **model_params)
    #
    # model.init_sess(sess)
    # saver = tf.train.Saver()
    #
    # # Train
    # if mode in ('train', 'test'):
    #     # Process data
    #     print("Reading in data")
    #     train_data, valid_data, test_data = load_data(
    #         train_filename,
    #         valid_filename,
    #         test_filename,
    #         delimiter=delimiter,
    #         col_names=col_names)
    #
    #     if mode == 'train':
    #         train(
    #             model,
    #             sess,
    #             saver,
    #             train_data,
    #             valid_data,
    #             batch_size=batch_size,
    #             max_iters=max_iters,
    #             use_early_stop=use_early_stop,
    #             early_stop_max_iter=early_stop_max_iter)
    #
    #     print('Loading best checkpointed model')
    #     saver.restore(sess, model.model_filename)
    #     test(model, sess, saver, test_data, train_data=train_data)


if __name__ == '__main__':
    with tf.Session() as session:
        main(session)
