import os
import random

import pandas as pd

_COL_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
_DELIMITER = '\t'
_VALIDATION_PORTION = 0.1


def _get_file_path(file_name):
    return os.path.join('data/ml-100k/{}'.format(file_name))


def _is_file_exists(file_name):
    file_path = _get_file_path(file_name)
    return os.path.exists(file_path)


def _get_train_valid_file_name(file_name):
    train_file_name = file_name + '.train'
    valid_file_name = file_name + '.valid'
    return train_file_name, valid_file_name


def _split_data(file_name):
    train_file_name, valid_file_name = _get_train_valid_file_name(file_name)
    file_path = _get_file_path(file_name)
    train_file_path = _get_file_path(train_file_name)
    valid_file_path = _get_file_path(valid_file_name)

    with open(file_path, 'r') as f, \
         open(train_file_path, 'w') as train_f, \
         open(valid_file_path, 'w') as valid_f:
        for line in f:
            if random.random() < 0.1:
                valid_f.write(line)
            else:
                train_f.write(line)


def _split_data_if_not_splitted(file_name):
    train_file_name, valid_file_name = _get_train_valid_file_name(file_name)

    if not _is_file_exists(train_file_name) or not _is_file_exists(
            valid_file_name):
        _split_data(file_name)

    return train_file_name, valid_file_name


def load_ml_100k_data():
    train_file_name, valid_file_name = _split_data_if_not_splitted('u1.base')
    test_file_name = 'u1.test'

    train_data = pd.read_csv(
        _get_file_path(train_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    train_data['user_id'] = train_data['user_id'] - 1
    train_data['item_id'] = train_data['item_id'] - 1

    valid_data = pd.read_csv(
        _get_file_path(valid_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    valid_data['user_id'] = valid_data['user_id'] - 1
    valid_data['item_id'] = valid_data['item_id'] - 1

    test_data = pd.read_csv(
        _get_file_path(test_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    test_data['user_id'] = test_data['user_id'] - 1
    test_data['item_id'] = test_data['item_id'] - 1

    return {'train': train_data, 'valid': valid_data, 'test': test_data}
