import os

import pandas as pd

ML_100K = 'ml-100k'


def _make_dir_if_not_exists(kind):
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/{}'.format(kind)):
        os.system(
            'wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O data/ml-100k.zip'
        )
        os.system('unzip data/{}.zip -d data'.format(kind))


def load_data(kind):
    _make_dir_if_not_exists(kind)

    if kind == ML_100K:
        train_filename = 'data/ml-100k/u1.base'
        valid_filename = 'data/ml-100k/u1.base'
        test_filename = 'data/ml-100k/u1.test'
        delimiter = '\t'
        col_names = ['user_id', 'item_id', 'rating', 'timestamp']

        train_data = pd.read_csv(
            train_filename, delimiter=delimiter, header=None, names=col_names)
        train_data['user_id'] = train_data['user_id'] - 1
        train_data['item_id'] = train_data['item_id'] - 1

        valid_data = pd.read_csv(
            valid_filename, delimiter=delimiter, header=None, names=col_names)
        valid_data['user_id'] = valid_data['user_id'] - 1
        valid_data['item_id'] = valid_data['item_id'] - 1

        test_data = pd.read_csv(
            test_filename, delimiter=delimiter, header=None, names=col_names)
        test_data['user_id'] = test_data['user_id'] - 1
        test_data['item_id'] = test_data['item_id'] - 1

        return train_data, valid_data, test_data
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))


# import tensorflow as tf
# import numpy as np

# train_user = None
# train_item = None
# train_rating = None
#
#
# def _gen_and_split_from_txt(*args, **kwargs):
#     data = np.genfromtxt(*args, **kwargs)
#     np.random.shuffle(data)
#
#     user = data[:, 0]
#     item = data[:, 1]
#     rating = data[:, 2]
#     del data
#     return user, item, rating
#
#
# def init_data_sets():
#     global train_user
#     global train_item
#     global train_rating
#
#     global test_user
#     global test_item
#     global test_rating
#
#     if not os.path.exists('data'):
#         os.mkdir('data')
#     if not os.path.exists('data/ml-100k'):
#         os.system(
#             'wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O data/ml-100k.zip'
#         )
#         os.system('unzip data/ml-100k.zip -d data')
#
#     user_ids = np.genfromtxt(
#         'data/ml-100k/u.user', delimiter='|', usecols=(0), dtype=int)
#     f = open('data/ml-100k/u.item', 'r', errors='replace')
#     item_ids = np.genfromtxt(f, usecols=0, delimiter='|', dtype=int)
#     f.close()
#
#     train_user, train_item, train_rating = _gen_and_split_from_txt(
#         'data/ml-100k/u1.base', delimiter='\t', usecols=(0, 1, 2), dtype=int)
#     train_user = _encode(train_user, user_ids)
#     train_item = _encode(train_item, item_ids)
#
#     test_user, test_item, test_rating = _gen_and_split_from_txt(
#         'data/ml-100k/u1.test', delimiter='\t', usecols=(0, 1, 2), dtype=int)
#     test_user = _encode(test_user, user_ids)
#     test_item = _encode(test_item, item_ids)
#
#     del user_ids
#     del item_ids
#
#     if not os.path.exists('data/ml-100k/np'):
#         os.mkdir('data/ml-100k/np')
#
#
# def _encode(data, ids):
#     for i in range(data.shape[0]):
#         data[i] = np.where(ids == data[i])[0][0]
#     return data
#
#
# def get_total_batch_number():
#     return train_user.shape[0] // BATCH_SIZE
#
#
# idx = 0
#
#
# def get_train_data():
#     return train_user[:20000], train_item[:20000], train_rating[:20000]
#
#
# def get_test_data():
#     return test_user, test_item, test_rating
#
#
# def next_batch():
#     assert train_user.shape[0] % BATCH_SIZE == 0
#
#     global idx
#     start_idx = idx
#     end_idx = idx + BATCH_SIZE
#     idx += BATCH_SIZE
#     idx %= train_user.shape[0]
#
#     return train_user[start_idx:end_idx], \
#         train_item[start_idx:end_idx], \
#         train_rating[start_idx:end_idx]