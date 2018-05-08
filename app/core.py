import argparse, json, time, os

import tensorflow as tf
import pandas as pd
import numpy as np

from .models import NNMF
from .utils import dataset


def train(model, sess, saver, train_data, valid_data, batch_size, max_iters,
          use_early_stop, early_stop_max_iter):
    # Print initial values
    batch = train_data.sample(batch_size) if batch_size else train_data
    print(batch.shape)
    print(valid_data.shape)
    train_error = model.eval_loss(batch)
    train_rmse = model.eval_rmse(batch)
    valid_rmse = model.eval_rmse(valid_data)
    print("{:3f} {:3f}, {:3f}".format(train_error, train_rmse, valid_rmse))

    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0
    for i in range(max_iters):
        # Run SGD
        batch = train_data.sample(batch_size) if batch_size else train_data
        model.train_iteration(batch)

        # Evaluate
        train_error = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_rmse = model.eval_rmse(valid_data)
        print("{:3f} {:3f}, {:3f}".format(train_error, train_rmse, valid_rmse))

        # Checkpointing/early stopping
        if use_early_stop:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(sess, model.model_filename)
            elif early_stop_iters == early_stop_max_iter:
                print("Early stopping ({} vs. {})...".format(
                    prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(sess, model.model_filename)


def test(model, sess, saver, test_data, train_data=pd.DataFrame(), log=True):
    if train_data.empty:
        train_rmse = model.eval_rmse(train_data)
        if log:
            print("Final train RMSE: {}".format(train_rmse))

    test_rmse = model.eval_rmse(test_data)
    if log:
        print("Final test RMSE: {}".format(test_rmse))

    return test_rmse


def run():
    # Set up command line params
    parser = argparse.ArgumentParser(
        description='Trains/evaluates NNMF models.')
    parser.add_argument(
        '--train',
        metavar='TRAIN_INPUT_FILE',
        type=str,
        default='data/ml-100k/split/u.data.train',
        help='the location of the training set\'s input file')
    parser.add_argument(
        '--valid',
        metavar='VALID_INPUT_FILE',
        type=str,
        default='data/ml-100k/split/u.data.valid',
        help='the location of the validation set\'s input file')
    parser.add_argument(
        '--test',
        metavar='TEST_INPUT_FILE',
        type=str,
        default='data/ml-100k/split/u.data.test',
        help='the location of the test set\'s input file')
    parser.add_argument(
        '--users',
        metavar='NUM_USERS',
        type=int,
        default=943,  # ML 100K has 943 users
        help='the number of users in the data set')
    parser.add_argument(
        '--movies',
        metavar='NUM_MOVIES',
        type=int,
        default=1682,  # ML 100K has 1682 movies
        help='the number of movies in the data set')
    parser.add_argument(
        '--model-params',
        metavar='MODEL_PARAMS_JSON',
        type=str,
        default='{}',
        help='JSON string containing model params')
    parser.add_argument(
        '--delim',
        metavar='DELIMITER',
        type=str,
        default='\t',
        help='the delimiter to use when parsing input files')
    parser.add_argument(
        '--cols',
        metavar='COL_NAMES',
        type=str,
        default=['user_id', 'item_id', 'rating'],
        help='the column names of the input data',
        nargs='+')
    parser.add_argument(
        '--no-early',
        default=False,
        action='store_true',
        help='disable early stopping')
    parser.add_argument(
        '--early-stop-max-iter',
        metavar='EARLY_STOP_MAX_ITER',
        type=int,
        default=40,
        help=
        'the maximum number of iterations to let the model continue training after reaching a '
        'minimum validation error')
    parser.add_argument(
        '--max-iters',
        metavar='MAX_ITERS',
        type=int,
        default=10000,
        help='the maximum number of iterations to allow the model to train for'
    )
    parser.add_argument(
        '--hyperparam-search-size',
        metavar='HYPERPARAM_SEARCH_SIZE',
        type=int,
        default=50,
        help=
        'when in "select" mode, the number of times to sample for random search'
    )

    # Parse args
    args = parser.parse_args()
    # Global args
    train_filename = args.train
    valid_filename = args.valid
    test_filename = args.test
    num_users = args.users
    num_items = args.movies
    model_params = json.loads(args.model_params)
    delimiter = args.delim
    col_names = args.cols
    batch_size = None
    use_early_stop = not (args.no_early)
    early_stop_max_iter = args.early_stop_max_iter
    max_iters = args.max_iters

    with tf.Session() as sess:
        # Define computation graph & Initialize
        print('Building network & initializing variables')

        model = NNMF(num_users, num_items, **model_params)
        model.init_sess(sess)
        saver = tf.train.Saver()

        # Process data
        print("Reading in data")
        train_data, valid_data, test_data = dataset.load_data(
            train_filename,
            valid_filename,
            test_filename,
            delimiter=delimiter,
            col_names=col_names)

        train(
            model,
            sess,
            saver,
            train_data,
            valid_data,
            batch_size=batch_size,
            max_iters=max_iters,
            use_early_stop=use_early_stop,
            early_stop_max_iter=early_stop_max_iter)

        print('Loading best checkpointed model')
        saver.restore(sess, model.model_filename)
        test(model, sess, saver, test_data, train_data=train_data)
