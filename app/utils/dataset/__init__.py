import os

import pandas as pd

from .ml_100k import load_ml_100k_data
from .ml_1m import load_ml_1m_data
from . import utils
from .utils import ML_100K
from .utils import ML_1M


def load_data(kind):
    utils.make_dir_if_not_exists('data')

    if kind == ML_100K:
        return load_ml_100k_data()
    elif kind == ML_1M:
        return load_ml_1m_data()
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))


def get_N_and_M(kind):
    if kind == ML_100K:
        return 943, 1682
    elif kind == ML_1M:
        return 6040, 3952
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
