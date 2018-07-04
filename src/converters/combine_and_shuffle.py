# coding=utf-8

"""
    组合初赛数据 和随机打乱
"""

import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import math
import pandas as pd

sys.path.append('../')
from utils.tencent_data_func import *
from utils.donal_args import args


nums = 100000000
print "reading train data"
train_dict, train_num = read_raw_data(args.combine_train_path, nums)
nums = 100000000
chusai_train_dict, chusai_train_num = read_raw_data(args.chusai_combine_train_path, nums)

rng_state = np.random.get_state()

for key in train_dict:
    train_dict[key].extend(chusai_train_dict[key])
    np.random.set_state(rng_state)
    np.random.shuffle(train_dict[key])


def write_data(data_dict, path):
    f = open(path,'wb')
    headers = [key for key in data_dict]
    f.write(','.join(headers)+'\n')
    for i,d in enumerate(data_dict['label']):
        row = []
        for key in headers:
            row.append(data_dict[key][i])
        f.write(','.join(row)+'\n')
    f.close()

write_data(train_dict, args.random_combine_train_path_with_chusai)
print "combine and shuffle data finished"





