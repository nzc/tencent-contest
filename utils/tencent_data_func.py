# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import random

from donal_args import args

"""
some functions to process tencent data
"""

def read_raw_data(path, max_num):
    f = open(path, 'rb')
    features = f.readline().strip().split(',')
    dict = {}
    num = 0
    for line in f:
        if num >= max_num:
            break
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.has_key(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
        num += 1
    f.close()
    return dict,num

def read_raw_data_2(path, begin, end):
    f = open(path, 'rb')
    features = f.readline().strip().split(',')
    dict = {}
    num = 0
    for line in f:
        num += 1
        if num <= begin:
            continue
        if num > end:
            break
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.has_key(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
    f.close()
    return dict,num

def onehot_feature_process(train_data, test1_data, test2_data, begin_num, filter_num = 100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    for d in train_data:
        if not count_dict.has_key(d):
            count_dict[d] = 0
        count_dict[d] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)
    train_res = []
    for d in train_data:
        if d in filter_set:
            d = '-2'
        if not index_dict.has_key(d):
            index_dict[d] = begin_index
            begin_index += 1
        train_res.append(index_dict[d])
    if not index_dict.has_key('-2'):
        index_dict['-2'] = begin_index
    test1_res = []
    for d in test1_data:
        if d in filter_set or not index_dict.has_key(d):
            d = '-2'
        test1_res.append(index_dict[d])
    test2_res = []
    for d in test2_data:
        if d in filter_set or not index_dict.has_key(d):
            d = '-2'
        test2_res.append(index_dict[d])
    return np.array(train_res), np.array(test1_res), np.array(test2_res), index_dict

def vector_feature_process(train_data, test1_data, test2_data, begin_num, max_len = 30, filter_num = 100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    print "dict counting"
    for d in train_data:
        xs = d.split(' ')
        for x in xs:
            if not count_dict.has_key(x):
                count_dict[x] = 0
            count_dict[x] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)
    train_res = []
    for d in train_data:
        xs = d.split(' ')
        row = [0] * max_len
        for i, x in enumerate(xs):
            if x in filter_set:
                x = '-2'
            if not index_dict.has_key(x):
                index_dict[x] = begin_index
                begin_index += 1
            row[i] = index_dict[x]
        train_res.append(row)
    if not index_dict.has_key('-2'):
        index_dict['-2'] = begin_index
    test1_res = []
    for d in test1_data:
        row = [0] * max_len
        xs = d.split(' ')
        for i, x in enumerate(xs):
            if x in filter_set or not index_dict.has_key(x):
                x = '-2'
            row[i] = index_dict[x]
        test1_res.append(row)
    test2_res = []
    for d in test2_data:
        row = [0] * max_len
        xs = d.split(' ')
        for i, x in enumerate(xs):
            if x in filter_set or not index_dict.has_key(x):
                x = '-2'
            row[i] = index_dict[x]
        test2_res.append(row)
    return np.array(train_res), np.array(test1_res), np.array(test2_res), index_dict

def onehot_combine_process(train_data_1, train_data_2, test1_data_1, test1_data_2, test2_data_1, test2_data_2, begin_num, filter_num = 100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    for i, d in enumerate(train_data_1):
        id_1 = train_data_1[i]
        id_2 = train_data_2[i]
        t_id = id_1 + '|' + id_2
        if not count_dict.has_key(t_id):
            count_dict[t_id] = 0
        count_dict[t_id] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)
    train_res = []
    for i, d in enumerate(train_data_1):
        id_1 = train_data_1[i]
        id_2 = train_data_2[i]
        t_id = id_1 + '|' + id_2
        if t_id in filter_set:
            t_id = '-2'
        if not index_dict.has_key(t_id):
            index_dict[t_id] = begin_index
            begin_index += 1
        train_res.append(index_dict[t_id])
    if not index_dict.has_key('-2'):
        index_dict['-2'] = begin_index
    test1_res = []
    for i, d in enumerate(test1_data_1):
        id_1 = test1_data_1[i]
        id_2 = test1_data_2[i]
        t_id = id_1 + '|' + id_2
        if t_id in filter_set or not index_dict.has_key(t_id):
            t_id = '-2'
        test1_res.append(index_dict[t_id])
    test2_res = []
    for i, d in enumerate(test2_data_1):
        id_1 = test2_data_1[i]
        id_2 = test2_data_2[i]
        t_id = id_1 + '|' + id_2
        if t_id in filter_set or not index_dict.has_key(t_id):
            t_id = '-2'
        test2_res.append(index_dict[t_id])
    return np.array(train_res), np.array(test1_res), np.array(test2_res), index_dict

def vector_combine_process(train_data_1, train_data_2, test1_data_1, test1_data_2,
                          test2_data_1, test2_data_2, begin_num, max_len = 30, filter_num=100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    for i, d in enumerate(train_data_1):
        xs_1 = train_data_1[i].split(' ')
        xs_2 = train_data_2[i].split(' ')
        for x_1 in xs_1:
            for x_2 in xs_2:
                t_id = x_1 + '|' + x_2
                if not count_dict.has_key(t_id):
                    count_dict[t_id] = 0
                count_dict[t_id] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)
    train_res = []
    for i, d in enumerate(train_data_1):
        xs_1 = train_data_1[i].split(' ')
        xs_2 = train_data_2[i].split(' ')
        row = [0] * max_len
        j = 0
        for x_1 in xs_1:
            for x_2 in xs_2:
                t_id = x_1 + '|' + x_2
                if t_id in filter_set:
                    t_id = '-2'
                if not index_dict.has_key(t_id):
                    index_dict[t_id] = begin_index
                    begin_index += 1
                row[j] = index_dict[t_id]
                j += 1
        train_res.append(row)
    if not index_dict.has_key('-2'):
        index_dict['-2'] = begin_index
    test1_res = []
    for i, d in enumerate(test1_data_1):
        xs_1 = test1_data_1[i].split(' ')
        xs_2 = test1_data_2[i].split(' ')
        row = [0]*max_len
        j = 0
        for x_1 in xs_1:
            for x_2 in xs_2:
                t_id = x_1 + '|' + x_2
                if t_id in filter_set or not index_dict.has_key(t_id):
                    t_id = '-2'
                row[j] = index_dict[t_id]
                j += 1
        test1_res.append(row)
    test2_res = []
    for i, d in enumerate(test2_data_1):
        xs_1 = test2_data_1[i].split(' ')
        xs_2 = test2_data_2[i].split(' ')
        row = [0] * max_len
        j = 0
        for x_1 in xs_1:
            for x_2 in xs_2:
                t_id = x_1 + '|' + x_2
                if t_id in filter_set or not index_dict.has_key(t_id):
                    t_id = '-2'
                row[j] = index_dict[t_id]
                j += 1
        test2_res.append(row)
    return np.array(train_res), np.array(test1_res), np.array(test2_res), index_dict

def read_indexs(path):
    f = open(path,'rb')
    res = []
    for line in f:
        res.append(int(line.strip())-1)
    return res

def write_data_into_parts(data, root_path, nums = 5100000):
    l = data.shape[0] // nums
    for i in range(l+1):
        begin = i * nums
        end = min(nums*(i+1), data.shape[0])
        t_data  = data[begin:end]
        t_data.tofile(root_path+'_'+str(i)+'.bin')

def write_data_into_parts_to_npy(data, root_path, nums = 5100000):
    l = data.shape[0] // nums
    for i in range(l+1):
        begin = i * nums
        end = min(nums*(i+1), data.shape[0])
        t_data  = data[begin:end]
        np.save(root_path+'_'+str(i)+'.npy',t_data)

def get_vector_feature_len(data):
    res = []
    for d in data:
        cnt = 0
        for item in d:
            if item != 0:
                cnt += 1
        res.append(cnt)
    return np.array(res)

def read_csv_helper(path):
    data = []
    with open(path) as f:
        tmp = []
        for line in f:
            items = line.strip().split(',')
            for item in items:
                tmp.append(int(item))
        data.append(tmp)
    return np.array(data)

def read_data_from_bin(path, col_names, part, sizes):
    data_arr = []
    for col_name in col_names:
        data_arr.append(np.fromfile(path + col_name + '_' + str(part) +'.bin', dtype=np.int).reshape(sizes))
    return np.concatenate(data_arr,axis=1)

def read_data_from_npy(path, col_names, part, sizes):
    data_arr = []
    for col_name in col_names:
        data_arr.append(np.load(path + col_name + '_' + str(part) +'.npy').reshape(sizes))
    return np.concatenate(data_arr,axis=1)

def read_data_from_csv(path, col_names, part, sizes):
    data_arr = []
    for col_name in col_names:
        data_arr.append(read_csv_helper(path + col_name + '_' + str(part) +'.csv').reshape(sizes))
    return np.concatenate(data_arr,axis=1)

def read_data_from_bin_to_dict(path, col_names, part, sizes=None):
    data_arr = {}
    t_sizes = sizes
    for col_name in col_names:
        if sizes == None:
            t_sizes = [-1, args.dynamic_features_max_len_dict[col_name]]
        data_arr[col_name] = np.fromfile(path + col_name + '_' + str(part) + '.bin', dtype=np.int).reshape(t_sizes)
    return data_arr

def read_data_from_npy_to_dict(path, col_names, part, sizes=None):
    data_arr = {}
    t_sizes = sizes
    bin_cols = ['log_uid_adCategoryId_times_count','log_uid_advertiserId_times_count',
                'log_uid_campaignId_times_count','log_uid_creativeId_times_count',
                'log_uid_creativeSize_times_count','log_uid_productId_times_count',
                'log_uid_productType_times_count']
    for col_name in col_names:
        if sizes == None:
            t_sizes = [-1, args.dynamic_features_max_len_dict[col_name]]
        if col_name not in bin_cols:
            data_arr[col_name] = np.load(path + col_name + '_' + str(part) + '.npy').reshape(t_sizes)
        else:
            data_arr[col_name] = np.fromfile(path + col_name + '_' + str(part) + '.bin', dtype=np.int).reshape(t_sizes)
    return data_arr

def read_data_from_csv_to_dict(path, col_names, part, sizes=None):
    data_arr = {}
    t_sizes = sizes
    for col_name in col_names:
        if sizes == None:
            t_sizes = [-1, args.dynamic_features_max_len_dict[col_name]]
        data_arr[col_name] = read_csv_helper(path + col_name + '_' + str(part) + '.csv').reshape(t_sizes)
    return data_arr

def load_tencent_data_to_dict(path, part):
    static_features = args.static_features
    dynamic_features = args.dynamic_features
    dynamic_lengths = []
    for f in dynamic_features:
        dynamic_lengths.append(f + '_lengths')
    labels = read_data_from_bin(path, ['label'], part, [-1, 1]).reshape([-1])
    static_index_dict = read_data_from_bin_to_dict(path, static_features, part, [-1])
    dynamic_index_dict = read_data_from_bin_to_dict(path, dynamic_features, part)
    dynamic_lengths_dict = read_data_from_bin_to_dict(path, dynamic_lengths, part, [-1])
    return labels, static_index_dict, dynamic_index_dict, dynamic_lengths_dict, None

def load_concatenate_tencent_data_to_dict(data_root_path, parts, static_features, dynamic_features, is_csv=False):
    labels = []
    static_ids = []
    dynamic_ids = []
    dynamic_lengths = []
    extern_lr_ids = None
    dynamic_lens = []
    for f in dynamic_features:
        dynamic_lens.append(f + '_lengths')
    for part in parts:
        print part, "part loading"
        b_time = time()
        extern_lr_features = args.extern_lr_features
        if not is_csv:
            labels.append(read_data_from_bin(data_root_path, ['label'], part, [-1, 1]).reshape([-1]))
        else:
            labels.append(read_data_from_csv(data_root_path, ['label'], part, [-1, 1]).reshape([-1]))
        if not is_csv:
            static_ids.append(read_data_from_bin_to_dict(data_root_path, static_features, part, [-1]))
            dynamic_ids.append(read_data_from_bin_to_dict(data_root_path, dynamic_features, part))
            dynamic_lengths.append(read_data_from_bin_to_dict(data_root_path, dynamic_lens, part, [-1]))
        else:
            static_ids.append(read_data_from_csv_to_dict(data_root_path, static_features, part, [-1]))
            dynamic_ids.append(read_data_from_csv_to_dict(data_root_path, dynamic_features, part))
            dynamic_lengths.append(read_data_from_csv_to_dict(data_root_path, dynamic_lens, part, [-1]))
        extern_lr_ids = None
        print "%d part loading costs %.1f s" % (part, time() - b_time)
        if len(extern_lr_features) != 0:
            extern_lr_ids = read_data_from_bin(data_root_path, extern_lr_features, part)
    static_index_dict = {}
    dynamic_index_dict = {}
    dynamic_lengths_dict = {}
    for key in static_features:
        static_index_dict[key] = np.concatenate([item[key] for item in static_ids], axis = 0)
    for key in dynamic_features:
        dynamic_index_dict[key] = np.concatenate([item[key] for item in dynamic_ids], axis = 0)
        dynamic_lengths_dict[key] = np.concatenate([item[key+'_lengths'] for item in dynamic_lengths], axis = 0)
    return np.concatenate(labels, axis=0), static_index_dict, \
            dynamic_index_dict, dynamic_lengths_dict, \
            extern_lr_ids

def load_concatenate_tencent_data_from_npy_to_dict(data_root_path, parts, static_features,dynamic_features,is_csv=False):
    labels = []
    static_ids = []
    dynamic_ids = []
    dynamic_lengths = []
    extern_lr_ids = None
    dynamic_lens = []
    for f in dynamic_features:
        dynamic_lens.append(f + '_lengths')
    for part in parts:
        print part, "part loading"
        b_time = time()
        extern_lr_features = args.extern_lr_features
        if not is_csv:
            labels.append(read_data_from_npy(data_root_path, ['label'], part, [-1, 1]).reshape([-1]))
        else:
            labels.append(read_data_from_csv(data_root_path, ['label'], part, [-1, 1]).reshape([-1]))
        if not is_csv:
            static_ids.append(read_data_from_npy_to_dict(data_root_path, static_features, part, [-1]))
            dynamic_ids.append(read_data_from_npy_to_dict(data_root_path, dynamic_features, part))
            dynamic_lengths.append(read_data_from_npy_to_dict(data_root_path, dynamic_lens, part, [-1]))
        else:
            static_ids.append(read_data_from_csv_to_dict(data_root_path, static_features, part, [-1]))
            dynamic_ids.append(read_data_from_csv_to_dict(data_root_path, dynamic_features, part))
            dynamic_lengths.append(read_data_from_csv_to_dict(data_root_path, dynamic_lens, part, [-1]))
        extern_lr_ids = None
        print "%d part loading costs %.1f s" % (part, time() - b_time)
        if len(extern_lr_features) != 0:
            extern_lr_ids = read_data_from_bin(data_root_path, extern_lr_features, part)
    static_index_dict = {}
    dynamic_index_dict = {}
    dynamic_lengths_dict = {}
    for key in static_features:
        static_index_dict[key] = np.concatenate([item[key] for item in static_ids], axis = 0)
    for key in dynamic_features:
        dynamic_index_dict[key] = np.concatenate([item[key] for item in dynamic_ids], axis = 0)
        dynamic_lengths_dict[key] = np.concatenate([item[key+'_lengths'] for item in dynamic_lengths], axis = 0)
    return np.concatenate(labels, axis=0), static_index_dict, \
            dynamic_index_dict, dynamic_lengths_dict, \
            extern_lr_ids

def load_dynamic_total_size_dict(path, dynamic_features):
    dynamic_max_len_dict = {}
    for key in dynamic_features:
        f = open(path + key+'.csv', 'rb')
        dynamic_max_len_dict[key] = len(f.readlines()) + 1
        f.close()
    return dynamic_max_len_dict

def load_static_total_size_dict(path, static_features):
    static_max_len_dict = {}
    for key in static_features:
        f = open(path + key+'.csv', 'rb')
        static_max_len_dict[key] = len(f.readlines()) + 1
        f.close()
    return static_max_len_dict

def load_tencent_data(data_root_path, part, test=False):
    static_features = args.static_features
    dynamic_features = args.dynamic_features
    dynamic_lengths = []
    for f in dynamic_features:
        dynamic_lengths.append(f+'_lengths')
    extern_lr_features = args.extern_lr_features
    if not test:
        labels = read_data_from_bin(data_root_path, ['label'], part, [-1,1]).reshape([-1])
    static_ids = read_data_from_bin(data_root_path, static_features, part,[-1,1])
    dynamic_ids = read_data_from_bin(data_root_path, dynamic_features, part, [-1,args.dynamic_max_len])
    dynamic_lengths = read_data_from_bin(data_root_path, dynamic_lengths, part, [-1,1])
    extern_lr_ids = None
    if len(extern_lr_features) != 0:
        extern_lr_ids = read_data_from_bin(data_root_path, extern_lr_features, part)
    if test:
        labels = np.array([0] * static_ids.shape[0])
    return labels, static_ids, dynamic_ids, dynamic_lengths, extern_lr_ids


def load_concatenate_tencent_data(data_root_path, parts, test=False):
    labels = []
    static_ids = []
    dynamic_ids = []
    dynamic_lengths = []
    extern_lr_ids = None
    num = 0
    for part in range(parts):
        print part, "part loading"
        b_time = time()
        static_features = args.static_features
        dynamic_features = args.dynamic_features
        dynamic_lens = []
        for f in dynamic_features:
            dynamic_lens.append(f+'_lengths')
        extern_lr_features = args.extern_lr_features
        if not test:
            labels.append(read_data_from_bin(data_root_path, ['label'], part, [-1,1]).reshape([-1]))
        static_ids.append(read_data_from_bin(data_root_path, static_features, part, [-1, 1]))
        dynamic_ids.append(read_data_from_bin(data_root_path, dynamic_features, part, [-1, args.dynamic_max_len]))
        dynamic_lengths.append(read_data_from_bin(data_root_path, dynamic_lens, part, [-1, 1]))
        num += len(static_ids[part])
        extern_lr_ids = None
        print "%d part loading costs %.1f s" % (part, time()-b_time)
        if len(extern_lr_features) != 0:
            extern_lr_ids = read_data_from_bin(data_root_path, extern_lr_features, part)
    if not test:
        return np.concatenate(labels,axis=0), np.concatenate(static_ids,axis=0), \
           np.concatenate(dynamic_ids, axis=0), np.concatenate(dynamic_lengths, axis=0),\
           extern_lr_ids
    else:
        return np.array([0]*num), np.concatenate(static_ids,axis=0), \
           np.concatenate(dynamic_ids, axis=0), np.concatenate(dynamic_lengths, axis=0),\
           extern_lr_ids


def bin_to_libffm(data_root_path, dir_data_root_path, parts, test=False):
    fw = open(dir_data_root_path, 'wb')
    for part in range(parts):
        print part, "preparing"
        labels, static_ids, dynamic_ids, dynamic_lengths, extern_lr_ids = load_tencent_data(data_root_path, part, test)
        for i, d in enumerate(labels):
            row = []
            row.append(str(labels[i]))
            st_size = static_ids.shape[1]
            for j in range(static_ids.shape[1]):
                row.append(str(j) + ':' + str(static_ids[i][j]) +':1')
            st_total_feature_size = 54634
            for j in range(dynamic_ids.shape[1]):
                ind = j // args.dynamic_max_len + st_size
                if dynamic_ids[i][j] != 0:
                    row.append(str(ind) + ':' + str(dynamic_ids[i][j] + st_total_feature_size) + ':1')
            fw.write(' '.join(row) + '\n')
    fw.close()



def write_dict(data_path, data):
    fw = open(data_path, 'wb')
    for key in data:
        fw.write(str(key)+','+str(data[key])+'\n')
    fw.close()

def count_feature_times(train_data, test1_data, test2_data):
    total_dict = {}
    count_dict = {}
    for i, d in enumerate(train_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for i, d in enumerate(test1_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for i, d in enumerate(test2_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for key in total_dict:
        if not count_dict.has_key(total_dict[key]):
            count_dict[total_dict[key]] = 0
        count_dict[total_dict[key]] += 1
    train_res = []
    for d in train_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        train_res.append(max(t))
    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        test1_res.append(max(t))
    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        test2_res.append(max(t))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def count_combine_feature_times(train_data_1, train_data_2, test1_data_1, test1_data_2, test2_data_1, test2_data_2):
    total_dict = {}
    count_dict = {}
    for i, d in enumerate(train_data_1):
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test1_data_1):
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test2_data_1):
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for key in total_dict:
        if not count_dict.has_key(total_dict[key]):
            count_dict[total_dict[key]] = 0
        count_dict[total_dict[key]] += 1

    train_res = []
    for i, d in enumerate(train_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        train_res.append(max(t))
    test1_res = []
    for i, d in enumerate(test1_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test1_res.append(max(t))
    test2_res = []
    for i, d in enumerate(test2_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test2_res.append(max(t))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def add_len(train_data, test1_data, test2_data):
    count_dict = {}
    train_res = []
    for i, d in enumerate(train_data):
        if d=='-1':
            train_res.append(0)
            if not count_dict.has_key(0):
                count_dict[0] = 0
            count_dict[0] += 1
        else:
            xs = d.split(' ')
            train_res.append(len(xs))
            if not count_dict.has_key(len(xs)):
                count_dict[len(xs)] = 0
            count_dict[len(xs)] += 1
    test1_res = []
    for i, d in enumerate(test1_data):
        if d=='-1':
            test1_res.append(0)
        else:
            xs = d.split(' ')
            test1_res.append(len(xs))
    test2_res = []
    for i, d in enumerate(test2_data):
        if d == '-1':
            test2_res.append(0)
        else:
            xs = d.split(' ')
            test2_res.append(len(xs))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            if not pos_dict.has_key(x):
                pos_dict[x] = 0
            total_dict[x] += 1
            if labels[i] == '1':
                pos_dict[x] += 1
    return total_dict, pos_dict

def combine_to_one(data1, data2):
    assert  len(data1) == len(data2)
    new_res = []
    for i, d in enumerate(data1):
        x1 = data1[i]
        x2 = data2[i]
        new_x = x1 + '|' + x2
        new_res.append(new_x)
    return new_res

def count_pos_feature(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last = nums-4739700
        assert last > 0
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    if not test_only:
        for i in range(k):
            print i,"part counting"
            print split_points[i], split_points[i+1]
            tmp = []
            total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data[j].split(' ')
                t = []
                for x in xs:
                    if not pos_dict.has_key(x):
                        t.append(0)
                        continue
                    t.append(pos_dict[x] + 1)
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
    count_dict = {-1:0}
    for key in pos_dict:
        if not count_dict.has_key(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.has_key(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x] + 1)
            train_res.append(max(t))

    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def count_pos_feature_2(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last  = nums - 4739700
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    if not test_only:
        for i in range(k):
            print i,"part counting"
            print split_points[i], split_points[i+1]
            tmp = []
            total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = count_train_data[j].split(' ')
                t = []
                for x in xs:
                    if not total_dict.has_key(x):
                        t.append(0)
                        continue
                    t.append(pos_dict[x]+1)
                tmp.append(max(t))
            train_res.extend(tmp)

    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(count_train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(count_labels)
    e = last*(k-1)/k

    total_dict, pos_dict = gen_count_dict(count_train_data[0:e], count_labels[0:e], 1, 0)
    count_dict = {-1:0}
    for key in pos_dict:
        if not count_dict.has_key(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.has_key(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x] + 1)
            train_res.append(max(t))


    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not total_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not total_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict


def uid_seq_feature(train_data, test1_data, test2_data, label):
    count_dict = {}
    seq_dict = {}
    seq_emb_dict = {}
    train_seq = []
    ind = 0
    for i, d in enumerate(train_data):
        if not count_dict.has_key(d):
            count_dict[d] = []
        seq_key = ' '.join(count_dict[d][-4:])
        if not seq_dict.has_key(seq_key):
            seq_dict[seq_key] = 0
            seq_emb_dict[seq_key] = ind
            ind += 1
        seq_dict[seq_key] += 1
        train_seq.append(seq_emb_dict[seq_key])
        count_dict[d].append(label[i])
    test1_seq = []
    for d in test1_data:
        if not count_dict.has_key(d):
            seq_key = ''
        else:
            seq_key = ' '.join(count_dict[d][-4:])
        if seq_emb_dict.has_key(seq_key):
            key = seq_emb_dict[seq_key]
        else:
            key = 0
        test1_seq.append(key)
    test2_seq = []
    for d in test2_data:
        if not count_dict.has_key(d):
            seq_key = ''
        else:
            seq_key = ' '.join(count_dict[d][-4:])
        if seq_emb_dict.has_key(seq_key):
            key = seq_emb_dict[seq_key]
        else:
            key = 0
        test2_seq.append(key)

    return np.array(train_seq), np.array(test1_seq), np.array(test2_seq), seq_emb_dict

def resort_data(raw_train_data, train_feat, ran_train_data):
    feat_dict = {}
    for i,d in enumerate(raw_train_data):
        feat_dict[d] = train_feat[i]
    train_res = []
    for d in ran_train_data:
        train_res.append(feat_dict[d])
    return np.array(train_res)

def count_combine_feature_times_with_chusai(train_data_1, train_data_2,
                                            test1_data_1, test1_data_2,
                                            test2_data_1, test2_data_2,
                                            chusai_train_data_1, chusai_train_data_2):
    total_dict = {}
    count_dict = {}
    for i, d in enumerate(chusai_train_data_1):
        xs1 = d.split(' ')
        xs2 = chusai_train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1

    for i, d in enumerate(train_data_1):
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test1_data_1):
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test2_data_1):
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for key in total_dict:
        if not count_dict.has_key(total_dict[key]):
            count_dict[total_dict[key]] = 0
        count_dict[total_dict[key]] += 1

    train_res = []
    for i, d in enumerate(train_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        train_res.append(max(t))
    test1_res = []
    for i, d in enumerate(test1_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test1_res.append(max(t))
    test2_res = []
    for i, d in enumerate(test2_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test2_res.append(max(t))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def count_feature_times_with_chusai(train_data, test1_data, test2_data, chusai_train_data):
    total_dict = {}
    count_dict = {}
    for i, d in enumerate(chusai_train_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for i, d in enumerate(train_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for i, d in enumerate(test1_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for i, d in enumerate(test2_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            total_dict[x] += 1
    for key in total_dict:
        if not count_dict.has_key(total_dict[key]):
            count_dict[total_dict[key]] = 0
        count_dict[total_dict[key]] += 1
    train_res = []
    for d in train_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        train_res.append(max(t))
    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        test1_res.append(max(t))
    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            t.append(total_dict[x])
        test2_res.append(max(t))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def count_pos_feature_with_chusai(train_data, test1_data, test2_data, chusai_train_data,
                                  chusai_labels, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last = 5100000 * 8
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    if not test_only:
        for i in range(k):
            print i,"part counting"
            print split_points[i], split_points[i+1]
            tmp = []
            total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[i],split_points[i+1])
            for d in chusai_train_data:
                xs = d.split(' ')
                for x in xs:
                    if not total_dict.has_key(x):
                        total_dict[x] = 0
                    if not pos_dict.has_key(x):
                        pos_dict[x] = 0
                    total_dict[x] += 1
                    if chusai_labels[i] == '1':
                        pos_dict[x] += 1
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data[j].split(' ')
                t = []
                for x in xs:
                    if not pos_dict.has_key(x):
                        t.append(0)
                        continue
                    t.append(pos_dict[x] + 1)
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
    for d in chusai_train_data:
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0
            if not pos_dict.has_key(x):
                pos_dict[x] = 0
            total_dict[x] += 1
            if chusai_labels[i] == '1':
                pos_dict[x] += 1
    count_dict = {-1:0}
    for key in pos_dict:
        if not count_dict.has_key(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.has_key(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x] + 1)
            train_res.append(max(t))

    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def count_pos_feature_all_set(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last = nums-4739700
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
    train_res = []
    for i ,d in enumerate(count_train_data):
        xs = train_data[i].split(' ')
        t = []
        for x in xs:
            if not total_dict.has_key(x):
                t.append(0)
            else:
                sub = 0
                if count_labels[i] == '1':
                    sub = 1
                t.append(pos_dict[x] - sub)
        train_res.append(max(t))
    count_dict = {}
    for key in pos_dict:
        if not count_dict.has_key(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.has_key(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x])
            train_res.append(max(t))

    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x])
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x])
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict


def count_pos_feature_by_history(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last = nums-4739700
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    if not test_only:
        for i in range(k):
            print i,"part counting"
            print split_points[i], split_points[i+1]
            tmp = []
            total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data[j].split(' ')
                t = []
                for x in xs:
                    if not pos_dict.has_key(x):
                        t.append(0)
                        continue
                    t.append(pos_dict[x] + 1)
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
    count_dict = {-1:0}
    for key in pos_dict:
        if not count_dict.has_key(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.has_key(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x] + 1)
            train_res.append(max(t))

    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.has_key(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict