# coding=utf-8

import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import math
import pandas as pd
part = int(sys.argv[1])

sys.path.append('../../')
from utils.tencent_data_func import *
from utils.donal_args import args

nums = 100000000
print "reading train data"
print args.random_combine_train_path_with_chusai
train_dict, train_num = read_raw_data(args.random_combine_train_path_with_chusai, nums)
print "reading test1 data"
test1_dict, test1_num = read_raw_data(args.combine_test1_path, nums)
print "reading test2 data"
test2_dict, test2_num = read_raw_data(args.combine_test2_path, nums)


one_hot_feature=['aid', 'advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId',
                 'productId', 'productType', 'LBS','age','carrier','consumptionAbility','education','gender','house']
combine_onehot_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house']
vector_feature=['os','ct', 'marriageStatus', 'interest1','interest2','interest3','interest4','interest5',
                'kw1','kw2','kw3','topic1','topic2','topic3']
combine_vector_feature = ['os','ct', 'marriageStatus', 'interest1','interest2','interest3','interest4','interest5']
app_feature = ['appIdAction','appIdInstall']

if part == 0:
    print "label preparing"
    labels = []
    for d in train_dict['label']:
        labels.append(int(d))
    labels = np.array(labels)
    write_data_into_parts(labels, args.dnn_data_path + 'train_with_chusai/train-label')
    # labels = []
    # for d in test1_dict['label']:
    #     labels.append(int(d))
    # labels = np.array(labels)
    # write_data_into_parts(labels, args.dnn_data_path+'test1/test-label')
    labels = []
    for d in test2_dict['label']:
        labels.append(int(d))
    labels = np.array(labels)
    write_data_into_parts(labels, args.dnn_data_path + 'test2_with_chusai/test-label')

    begin_num = 1
    for feature in one_hot_feature:
        print feature, "preparing"
        train_res, test1_res, test2_res, f_dict = onehot_feature_process(train_dict[feature], test1_dict[feature],
                                                                         test2_dict[feature], begin_num)
        write_dict(args.dnn_data_path + 'dict_with_chusai/' + feature + '.csv', f_dict)
        write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + feature)
        # write_data_into_parts(test1_res, args.dnn_data_path + 'test1/test-' + feature)
        write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + feature)

    for feature in combine_onehot_feature:
        print feature + '_aid', "preparing"
        train_res, test1_res, test2_res, f_dict = onehot_combine_process(train_dict[feature], train_dict['aid'],
                                                                         test1_dict[feature], test1_dict['aid'],
                                                                         test2_dict[feature], test2_dict['aid'],
                                                                         begin_num)
        write_dict(args.dnn_data_path + 'dict_with_chusai/' + feature + '_aid.csv', f_dict)
        write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + feature + '_aid')
        # write_data_into_parts(test1_res, args.dnn_data_path + 'test1/test-' + feature + '_aid')
        write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + feature + '_aid')

    print "static max_len :", begin_num
begin_num = 1
if part == 1:
    for feature in vector_feature:
        print feature, "preparing"
        max_len = args.dynamic_dict[feature]
        train_res, test1_res, test2_res, f_dict = vector_feature_process(train_dict[feature], test1_dict[feature],
                                                                         test2_dict[feature],
                                                                         begin_num, max_len)
        write_dict(args.dnn_data_path + 'dict_with_chusai/' + feature + '.csv', f_dict)
        write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + feature)
        # write_data_into_parts(test1_res, args.dnn_data_path + 'test1/test-' + feature)
        write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + feature)
        train_res_lengths = get_vector_feature_len(train_res)
        # test1_res_lengths = get_vector_feature_len(test1_res)
        test2_res_lengths = get_vector_feature_len(test2_res)
        write_data_into_parts(train_res_lengths, args.dnn_data_path + 'train_with_chusai/train-' + feature + '_lengths')
        # write_data_into_parts(test1_res_lengths, args.dnn_data_path + 'test1_with_chusai/test-' + feature + '_lengths')
        write_data_into_parts(test2_res_lengths, args.dnn_data_path + 'test2_with_chusai/test-' + feature + '_lengths')

if part == 2:
    for feature in ['interest1', 'interest2','marriageStatus']:
        print feature + '_aid', "preparing"
        max_len = args.dynamic_dict[feature]
        train_res, test1_res, test2_res, f_dict = vector_combine_process(train_dict[feature], train_dict['aid'],
                                                                         test1_dict[feature], test1_dict['aid'],
                                                                         test2_dict[feature], test2_dict['aid'],
                                                                         begin_num, max_len)
        write_dict(args.dnn_data_path + 'dict_with_chusai/' + feature + '_aid.csv', f_dict)
        write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + feature + '_aid')
        # write_data_into_parts(test1_res, args.dnn_data_path + 'test1_with_chusai/test-' + feature + '_aid')
        write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + feature + '_aid')
        train_res_lengths = get_vector_feature_len(train_res)
        # test1_res_lengths = get_vector_feature_len(test1_res)
        test2_res_lengths = get_vector_feature_len(test2_res)
        write_data_into_parts(train_res_lengths, args.dnn_data_path + 'train_with_chusai/train-' + feature + '_aid' + '_lengths')
        # write_data_into_parts(test1_res_lengths, args.dnn_data_path + 'test1_with_chusai/test-' + feature + '_aid' + '_lengths')
        write_data_into_parts(test2_res_lengths, args.dnn_data_path + 'test2_with_chusai/test-' + feature + '_aid' + '_lengths')

    print "dynamic max_len", begin_num



if part == 3:
    """
        len features
    """
    len_features = ['interest1', 'interest2', 'interest5']
    for f in len_features:
        print f + '_len adding'
        train_res, test1_res, test2_res, f_dict = add_len(train_dict[f], test1_dict[f], test2_dict[f])
        write_dict(args.dnn_data_path + 'dict_with_chusai/' + f + '_len' + '.csv', f_dict)
        write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + f + '_len')
        # write_data_into_parts(test1_res, args.dnn_data_path + 'test1/test-' + f + '_len')
        write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + f + '_len')

    """
        uid combine counts
    """
    print "uid counts adding"
    ad_feartures = ['uid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                    'adCategoryId', 'productId', 'productType']

    dnn_data_path = args.root_data_path + 'dnn/'
    for f in ad_feartures:
        print f, 'uid times counting'
        train_res, test1_res, test2_res, f_dict = count_combine_feature_times(train_dict['uid'], train_dict[f],
                                                                              test1_dict['uid'], test1_dict[f],
                                                                              test2_dict['uid'], test2_dict[f])
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'uid_' + f + '_times_count.csv', f_dict)
        write_data_into_parts(train_res, dnn_data_path + 'train_with_chusai/train-' + 'uid_' + f + '_times_count')
        # write_data_into_parts(test1_res, dnn_data_path + 'test1/test-' + 'uid_' + f + '_times_count')
        write_data_into_parts(test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'uid_' + f + '_times_count')

        print "log uid counts adding"
        log_train_res = []
        # log_test1_res = []
        log_test2_res = []
        log_f_dict = {}
        for val in train_res:
            log_train_res.append(int(math.log(1 + val * val)))
        # for val in test1_res:
        #     log_test1_res.append(int(math.log(1 + val * val)))
        for val in test2_res:
            log_test2_res.append(int(math.log(1 + val * val)))
        for key in f_dict:
            new_key = int(math.log(1 + key * key))
            if not log_f_dict.has_key(new_key):
                log_f_dict[new_key] = 0
            log_f_dict[new_key] += f_dict[key]
        log_train_res = np.array(log_train_res)
        # log_test1_res = np.array(log_test1_res)
        log_test2_res = np.array(log_test2_res)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'log_uid_' + f + '_times_count.csv', log_f_dict)
        write_data_into_parts(log_train_res, dnn_data_path + 'train_with_chusai/train-' + 'log_uid_' + f + '_times_count')
        # write_data_into_parts(log_test1_res, dnn_data_path + 'test1/test-' + 'log_uid_' + f + '_times_count')
        write_data_into_parts(log_test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'log_uid_' + f + '_times_count')

if part == 4:
    """
        uid pos cimbine counts
    """
    print "uid pos combine counts"
    ad_feartures = ['uid','advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                    'adCategoryId', 'productId', 'productType']
    dnn_data_path = args.root_data_path + 'dnn/'
    print "uid pos combine counts adding"
    for f in ad_feartures:
        print f,'uid pos counting'
        new_train_data = combine_to_one(train_dict['uid'], train_dict[f])
        new_test1_data = combine_to_one(test1_dict['uid'], test1_dict[f])
        new_test2_data = combine_to_one(test2_dict['uid'], test2_dict[f])
        train_res, test1_res, test2_res, f_dict = count_pos_feature(new_train_data, new_test1_data,
                                                                    new_test2_data, train_dict['label'], 5,
                                                                    is_val=False)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'uid_' + f + '_pos_times_count_5_fold_all.csv', f_dict)
        write_data_into_parts(train_res, dnn_data_path + 'train_with_chusai/train-' + 'uid_'+ f +'_pos_times_count_5_fold_all')
        # write_data_into_parts(test1_res, dnn_data_path + 'test1/test-' + 'uid_'+ f + '_pos_times_count_5_fold_all')
        write_data_into_parts(test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'uid_'+ f + '_pos_times_count_5_fold_all')

        print "log uid counts adding"
        log_train_res = []
        # log_test1_res = []
        log_test2_res = []
        log_f_dict = {}
        for val in train_res:
            log_train_res.append(int(math.log(1 + val * val)))
        # for val in test1_res:
        #     log_test1_res.append(int(math.log(1 + val * val)))
        for val in test2_res:
            log_test2_res.append(int(math.log(1 + val * val)))
        for key in f_dict:
            new_key = int(math.log(1 + key * key))
            if key == -1:
                new_key = -1
            if not log_f_dict.has_key(new_key):
                log_f_dict[new_key] = 0
            log_f_dict[new_key] += f_dict[key]
        log_train_res = np.array(log_train_res)
        # log_test1_res = np.array(log_test1_res)
        log_test2_res = np.array(log_test2_res)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'log_uid_' + f + '_pos_times_count_5_fold_all.csv', f_dict)
        write_data_into_parts(log_train_res,
                                     dnn_data_path + 'train_with_chusai/train-' + 'log_uid_' + f + '_pos_times_count_5_fold_all')
        # write_data_into_parts(test1_res,
        #                              dnn_data_path + 'test1_with_chusai/test-' + 'log_uid_' + f + '_pos_times_count_5_fold_all')
        write_data_into_parts(log_test2_res,
                                     dnn_data_path + 'test2_with_chusai/test-' + 'log_uid_' + f + '_pos_times_count_5_fold_all')

if part == 5:
    """
        uid pos combine counts sample
    """
    print "uid pos combine counts"
    ad_feartures = ['uid','advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                    'adCategoryId', 'productId', 'productType']
    dnn_data_path = args.root_data_path + 'dnn/'
    print "uid pos combine counts sample adding"
    for f in ad_feartures:
        print f, 'uid pos counting'
        new_train_data = combine_to_one(train_dict['uid'], train_dict[f])
        new_test1_data = combine_to_one(test1_dict['uid'], test1_dict[f])
        new_test2_data = combine_to_one(test2_dict['uid'], test2_dict[f])
        train_res, test1_res, test2_res, f_dict = count_pos_feature(new_train_data, new_test1_data,
                                                                    new_test2_data, train_dict['label'], 5,
                                                                    is_val=True)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'uid_' + f + '_pos_times_count_5_fold.csv', f_dict)
        write_data_into_parts(train_res, dnn_data_path + 'train_with_chusai/train-' + 'uid_'+ f +'_pos_times_count_5_fold')
        # write_data_into_parts(test1_res, dnn_data_path + 'test1_with_chusai/test-' + 'uid_'+ f + '_pos_times_count_5_fold')
        write_data_into_parts(test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'uid_'+ f + '_pos_times_count_5_fold')

        print "log uid counts adding"
        log_train_res = []
        # log_test1_res = []
        log_test2_res = []
        log_f_dict = {}
        for val in train_res:
            log_train_res.append(int(math.log(1 + val * val)))
        # for val in test1_res:
        #     log_test1_res.append(int(math.log(1 + val * val)))
        for val in test2_res:
            log_test2_res.append(int(math.log(1 + val * val)))
        for key in f_dict:
            new_key = int(math.log(1 + key * key))
            if key == -1:
                new_key = -1
            if not log_f_dict.has_key(new_key):
                log_f_dict[new_key] = 0
            log_f_dict[new_key] += f_dict[key]
        log_train_res = np.array(log_train_res)
        # log_test1_res = np.array(log_test1_res)
        log_test2_res = np.array(log_test2_res)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'log_uid_' + f + '_pos_times_count_5_fold.csv', f_dict)
        write_data_into_parts(log_train_res,
                                     dnn_data_path + 'train_with_chusai/train-' + 'log_uid_' + f + '_pos_times_count_5_fold')
        # write_data_into_parts(log_test1_res,
        #                              dnn_data_path + 'test1/test-' + 'log_uid_' + f + '_pos_times_count_5_fold')
        write_data_into_parts(log_test2_res,
                                     dnn_data_path + 'test2_with_chusai/test-' + 'log_uid_' + f + '_pos_times_count_5_fold')

combine_features = [('topic1','topic2'),('LBS','kw1'),('interest2','aid'),('kw2','creativeSize'),
                    ('kw2','aid'),('marriageStatus','aid'),('interest1','aid'),('LBS','kw2'),
                    ('topic2','aid'),('LBS','aid'),('interest3','aid'),('interest4','aid'),('interest5','aid'),
                    ('topic1','aid'),('kw3','aid'),('topic3','aid'),('kw1','aid')]
if part == 6:
    """
        seq feat counting
    """
    print "uid seq feat counts"
    dnn_data_path = args.root_data_path + 'dnn/'
    # raw_train_dict, raw_train_num = read_raw_data(args.raw_train_path, nums)
    train_res, test1_res, test2_res, f_dict = uid_seq_feature(train_dict['uid'], test1_dict['uid'],
                                                              test2_dict['uid'], train_dict['label'])
    # new_raw_train_data = combine_to_one(raw_train_dict['uid'], raw_train_dict['aid'])
    # new_train_data = combine_to_one(train_dict['uid'], train_dict['aid'])
    # train_res = resort_data(new_raw_train_data, raw_train_res, new_train_data)

    write_dict(dnn_data_path + 'dict_with_chusai/' + 'uid_seq.csv',f_dict)
    write_data_into_parts(train_res, dnn_data_path + 'train_with_chusai/train-' + 'uid_seq')
    # write_data_into_parts(test1_res, dnn_data_path + 'test1/test-' + 'uid_seq')
    write_data_into_parts(test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'uid_seq')

    for f1 in ['age', 'gender']:
        for f2 in ['advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId','productId', 'productType']:
            print f1 + '_' + f2, "preparing"
            train_res, test1_res, test2_res, f_dict = onehot_combine_process(train_dict[f1], train_dict[f2],
                                                                             test1_dict[f1], test1_dict[f2],
                                                                             test2_dict[f1], test2_dict[f2],
                                                                             begin_num)
            write_dict(args.dnn_data_path + 'dict_with_chusai/' + f1 + '_' + f2 + '.csv', f_dict)
            write_data_into_parts(train_res, args.dnn_data_path + 'train_with_chusai/train-' + f1 + '_' + f2)
            # write_data_into_parts(test1_res, args.dnn_data_path + 'test1/test-' + feature + '_aid')
            write_data_into_parts(test2_res, args.dnn_data_path + 'test2_with_chusai/test-' + f1 + '_' + f2)


if part == 7:
    """
        uid combine counts
    """
    print "uid counts adding"
    ad_feartures = ['uid','advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                    'adCategoryId', 'productId', 'productType']
    print "loading chusai test1"
    chusai_test1_dict, test1_num_chusai = read_raw_data(args.chusai_combine_test1_path, nums)
    print "loading chusai test2"
    chusai_test2_dict, test2_num_chusai = read_raw_data(args.chusai_combine_test2_path, nums)
    for key in chusai_test1_dict:
        chusai_test1_dict[key].extend(chusai_test2_dict[key])

    dnn_data_path = args.root_data_path + 'dnn/'
    for f in ad_feartures:
        print f, 'uid times counting'
        train_res, test1_res, test2_res, f_dict = count_combine_feature_times_with_chusai(train_dict['uid'], train_dict[f],
                                                                                          test1_dict['uid'], test1_dict[f],
                                                                                          test2_dict['uid'], test2_dict[f],
                                                                                          chusai_test1_dict['uid'],
                                                                                          chusai_test1_dict[f])
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'uid_' + f + '_times_count_with_chusai_test.csv', f_dict)
        write_data_into_parts(train_res, dnn_data_path + 'train_with_chusai/train-' + 'uid_' + f + '_times_count_with_chusai_test')
        # write_data_into_parts(test1_res, dnn_data_path + 'test1_with_chusai/test-' + 'uid_' + f + '_times_count_with_chusai_test')
        write_data_into_parts(test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'uid_' + f + '_times_count_with_chusai_test')

        print "log uid counts adding"
        log_train_res = []
        log_test1_res = []
        log_test2_res = []
        log_f_dict = {}
        for val in train_res:
            log_train_res.append(int(math.log(1 + val * val)))
        for val in test1_res:
            log_test1_res.append(int(math.log(1 + val * val)))
        for val in test2_res:
            log_test2_res.append(int(math.log(1 + val * val)))
        for key in f_dict:
            new_key = int(math.log(1 + key * key))
            if not log_f_dict.has_key(new_key):
                log_f_dict[new_key] = 0
            log_f_dict[new_key] += f_dict[key]
        log_train_res = np.array(log_train_res)
        log_test1_res = np.array(log_test1_res)
        log_test2_res = np.array(log_test2_res)
        write_dict(dnn_data_path + 'dict_with_chusai/' + 'log_uid_' + f + '_times_count_with_chusai_test.csv', log_f_dict)
        write_data_into_parts(log_train_res, dnn_data_path + 'train_with_chusai/train-' + 'log_uid_' + f + '_times_count_with_chusai_test')
        # write_data_into_parts(log_test1_res, dnn_data_path + 'test1_with_chusai/test-' + 'log_uid_' + f + '_times_count_with_chusai_test')
        write_data_into_parts(log_test2_res, dnn_data_path + 'test2_with_chusai/test-' + 'log_uid_' + f + '_times_count_with_chusai_test')


