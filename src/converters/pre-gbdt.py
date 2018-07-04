# coding=utf-8
"""
   生成gbdt的特征  需要多台服务器分块生成

"""

import numpy as np
import pandas as pd
import os
import sys
from random import random

import time

sys.path.append('../../')
from utils.donal_args import args

part = int(sys.argv[1])

"""
    辅助函数
"""

def read_data(path):
    f = open(path, 'rb')
    features = f.readline().strip().split(',')
    count_dict = {}
    for feature in features:
        count_dict[feature] = []
    num = 0
    for line in f:
        # if num > 1000:
        #     break
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            count_dict[features[i]].append(d)
        num += 1
    f.close()
    return count_dict,num

def shuffle_data(data_dict):
    rng_state = np.random.get_state()
    for key in data_dict:
        np.random.set_state(rng_state)
        np.random.shuffle(data_dict[key])

def clean(train_data, test_data):
    train_s = set()
    test_s = set()
    for i, data in enumerate(train_data):
        xs = data.split(' ')
        for x in xs:
            train_s.add(x)
    for i, data in enumerate(test_data):
        xs = data.split(' ')
        for x in xs:
            test_s.add(x)
    new_train_data = []
    new_test_data = []
    for i, data in enumerate(train_data):
        xs = data.split(' ')
        nxs = []
        for x in xs:
            if x in test_s:
                nxs.append(x)
        if len(nxs) == 0:
            nxs = ['-1']
        new_train_data.append(' '.join(nxs))
    for i, data in enumerate(test_data):
        xs = data.split(' ')
        nxs = []
        for x in xs:
            if x in train_s:
                nxs.append(x)
        if len(nxs) == 0:
            nxs = ['-1']
        new_test_data.append(' '.join(nxs))
    return new_train_data, new_test_data

def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            if not pos_dict.has_key(x):
                pos_dict[x] = 0.0
            total_dict[x] += 1
            if labels[i] == '1':
                pos_dict[x] += 1
    return total_dict, pos_dict

def gen_combine_count_dict(data1, data2, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data1):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        xs2 = data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                k = x1+'|'+x2
                if not total_dict.has_key(k):
                    total_dict[k] = 0.0
                if not pos_dict.has_key(k):
                    pos_dict[k] = 0.0
                total_dict[k] += 1
                if labels[i] == '1':
                    pos_dict[k] += 1
    return total_dict, pos_dict

def count_feature(train_data, test1_data, test2_data, labels, k, test_only= False):
    nums = len(train_data)
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    s = set()
    for d in train_data:
        xs = d.split(' ')
        for x in xs:
            s.add(x)
    b = nums // len(s)
    a = b*1.0 / 20

    train_res = []
    if not test_only:
        for i in range(k):
            tmp = []
            total_dict, pos_dict = gen_count_dict(train_data, labels, split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data[j].split(' ')
                t = []
                for x in xs:
                    if not total_dict.has_key(x):
                        t.append(0.05)
                        continue
                    t.append((a + pos_dict[x]) / (b + total_dict[x]))
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_count_dict(train_data, labels, 1, 0)
    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not total_dict.has_key(x):
                t.append(0.05)
                continue
            t.append((a + pos_dict[x]) / (b + total_dict[x]))
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not total_dict.has_key(x):
                t.append(0.05)
                continue
            t.append((a + pos_dict[x]) / (b + total_dict[x]))
        test2_res.append(max(t))

    return train_res, test1_res, test2_res

def count_combine_feature(train_data1, train_data2, test1_data1, test1_data2,
                          test2_data1, test2_data2, labels, k, test_only = False):
    nums = len(train_data1)
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    s = set()
    for i, d in enumerate(train_data1):
        xs = d.split(' ')
        xs2 = train_data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                s.add(ke)
    b = nums // len(s)
    a = b*1.0 / 20

    train_res = []
    if not test_only:
        for i in range(k):
            tmp = []
            total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels,
                                                          split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data1[j].split(' ')
                xs2 = train_data2[j].split(' ')
                t = []
                for x1 in xs:
                    for x2 in xs2:
                        ke = x1 + '|' + x2
                        if not total_dict.has_key(ke):
                            t.append(0.05)
                            continue
                        t.append((a + pos_dict[ke]) / (b + total_dict[ke]))
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, 1, 0)
    test1_res = []
    for i,d in enumerate(test1_data1):
        xs = d.split(' ')
        xs2 = test1_data2[i].split(' ')
        t = []
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    t.append(0.05)
                    continue
                t.append((a + pos_dict[ke]) / (b + total_dict[ke]))
        test1_res.append(max(t))

    test2_res = []
    for i, d in enumerate(test2_data1):
        xs = d.split(' ')
        xs2 = test2_data2[i].split(' ')
        t = []
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    t.append(0.05)
                    continue
                t.append((a + pos_dict[ke]) / (b + total_dict[ke]))
        test2_res.append(max(t))

    return train_res, test1_res, test2_res

def count_feature_times(train_data, test1_data, test2_data):
    total_dict = {}
    for i, d in enumerate(train_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            total_dict[x] += 1
    for i, d in enumerate(test1_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            total_dict[x] += 1
    for i, d in enumerate(test2_data):
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            total_dict[x] += 1
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
    return train_res, test1_res, test2_res

def count_combine_feature_times(train_data1, train_data2, test1_data1, test1_data2,
                        test2_data1, test2_data2):
    total_dict = {}
    for i, d in enumerate(train_data1):
        xs = d.split(' ')
        xs2 = train_data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0.0
                total_dict[ke] += 1
    for i, d in enumerate(test1_data1):
        xs = d.split(' ')
        xs2 = test1_data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0.0
                total_dict[ke] += 1
    for i, d in enumerate(test2_data1):
        xs = d.split(' ')
        xs2 = test2_data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    total_dict[ke] = 0.0
                total_dict[ke] += 1
    train_res = []
    for i, d in enumerate(train_data1):
        xs = d.split(' ')
        xs2 = train_data2[i].split(' ')
        t = []
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        train_res.append(max(t))
    test1_res = []
    for i, d in enumerate(test1_data1):
        xs = d.split(' ')
        xs2 = test1_data2[i].split(' ')
        t = []
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test1_res.append(max(t))
    test2_res = []
    for i, d in enumerate(test2_data1):
        xs = d.split(' ')
        xs2 = test2_data2[i].split(' ')
        t = []
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test2_res.append(max(t))
    return train_res, test1_res, test2_res

def count_pos_feature(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
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

    return train_res, test1_res, test2_res

def combine_to_one(data1, data2):
    assert  len(data1) == len(data2)
    new_res = []
    for i, d in enumerate(data1):
        x1 = data1[i]
        x2 = data2[i]
        new_x = x1 + '|' + x2
        new_res.append(new_x)
    return new_res

def add_len(data):
    res = []
    for i, d in enumerate(data):
        if d=='-1':
            res.append(0)
        else:
            xs = d.split(' ')
            res.append(len(xs))
    return res

def write_data(path, data, is_six = True):
    f = open(path, 'wb')
    for d in data:
        if is_six:
            f.write("%.6f" % (d) + '\n')
        else:
            f.write("%.1f" % (d) + '\n')
    f.close()

def write_label(path, data):
    f = open(path, 'wb')
    for d in data:
        f.write(d+'\n')
    f.close()



print "reading train data"
train_dict, train_num = read_data(args.random_combine_train_path_with_chusai)
print "reading test1 data"
test1_dict, test1_num = read_data(args.combine_test1_path)
print "reading test2 data"
test2_dict, test2_num = read_data(args.combine_test2_path)
print train_num, test1_num, test2_num
print "\n\n"


if part == 1:
    print "writing labels"
    write_label(args.gbdt_data_path + 'train/label.csv', train_dict['label'])
    write_label(args.gbdt_data_path + 'test1/label.csv', test1_dict['label'])
    write_label(args.gbdt_data_path + 'test2/label.csv', test2_dict['label'])
    print "\n\n"

print "adding len feature"
len_feature = ['interest1','interest2','interest3','interest4','interest5',
               'kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall']
if part == 1:
    for f in len_feature:
        print f + '_len adding'
        b_time = time.time()
        train_res = add_len(train_dict[f])
        test1_res = add_len(test1_dict[f])
        test2_res = add_len(test2_dict[f])
        write_data(args.gbdt_data_path + 'train/' + f + '_len.csv', train_res)
        write_data(args.gbdt_data_path + 'test1/' + f + '_len.csv', test1_res)
        write_data(args.gbdt_data_path + 'test2/' + f + '_len.csv', test2_res)
        print "costs %.1f s" % (time.time() - b_time)
    print "\n\n"

print "counting one rate feature"
user_one_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3','appIdAction','appIdInstall']
ad_one_feature = ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
if part == 1:
    for f in user_one_feature:
        print f, "rate preparing"
        b_time = time.time()
        train_res, test1_res, test2_res = count_feature(train_dict[f], test1_dict[f], test2_dict[f], train_dict['label'], 5)
        write_data(args.gbdt_data_path + 'train/' + f + '_rate_count.csv', train_res)
        write_data(args.gbdt_data_path + 'test1/' + f + '_rate_count.csv', test1_res)
        write_data(args.gbdt_data_path + 'test2/' + f + '_rate_count.csv', test2_res)
        print "costs %.1f s" % (time.time() - b_time)
    for f in ad_one_feature:
        print f, "rate preparing"
        b_time = time.time()
        train_res, test1_res, test2_res = count_feature(train_dict[f], test1_dict[f], test2_dict[f], train_dict['label'], 5)
        write_data(args.gbdt_data_path + 'train/' + f + '_rate_count.csv', train_res)
        write_data(args.gbdt_data_path + 'test1/' + f + '_rate_count.csv', test1_res)
        write_data(args.gbdt_data_path + 'test2/' + f + '_rate_count.csv', test2_res)
        print "costs %.1f s" % (time.time() - b_time)
    print "\n\n"

print "counting combine rate"
user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3']
finsh_feature_2 = ['LBS']
finsh_feature_3 = ['LBS','age','carrier','consumptionAbility','education','gender']

ad_combine_feature =  ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
if part == 2 or part == 7 or part == 8 or part == 9 or part == 10 or part == 11:
    for i, f1 in enumerate(user_combine_feature):
        if part == 2 and f1 not in ['age','carrier']:
            continue
        if part == 7 and f1 not in ['consumptionAbility','education']:
            continue
        if part == 8 and f1 not in ['gender','house','os']:
            continue
        if part == 9 and f1 not in ['ct','interest1','interest2','interest3']:
            continue
        if part == 10 and f1 not in ['interest4','interest5','marriageStatus']:
            continue
        if part == 11 and f1 not in ['topic1','topic2','topic3','kw1','kw2','kw3']:
            continue
        for f2 in user_combine_feature[i+1:]:
            print f1+'-'+f2+" counting preparing"
            b_time = time.time()
            train_res, test1_res, test2_res = count_combine_feature(train_dict[f1], train_dict[f2],
                                                                    test1_dict[f1], test1_dict[f2],
                                                                    test2_dict[f1], test2_dict[f2],
                                                                    train_dict['label'],5)
            write_data(args.gbdt_data_path + 'train/' + f1 + '-' + f2 + '_rate_count.csv', train_res)
            write_data(args.gbdt_data_path + 'test1/' + f1 + '-' + f2 + '_rate_count.csv', test1_res)
            write_data(args.gbdt_data_path + 'test2/' + f1 + '-' + f2 + '_rate_count.csv', test2_res)
            print "costs %.1f s" % (time.time() - b_time)

if part == 3 or part == 12 or part == 13:
    for i, f1 in enumerate(user_combine_feature):
        if part == 3 and f1 not in ['house','os','ct','interest1','interest2']:
            continue
        if part == 12 and f1 not in ['interest3','interest4','interest5','marriageStatus','topic1']:
            continue
        if part == 13 and f1 not in ['topic2','topic3','kw1','kw2','kw3']:
            continue
        for f2 in ad_combine_feature:
            print f1 + '-' + f2 + " counting preparing"
            b_time = time.time()
            train_res, test1_res, test2_res = count_combine_feature(train_dict[f1], train_dict[f2],
                                                                    test1_dict[f1], test1_dict[f2],
                                                                    test2_dict[f1], test2_dict[f2],
                                                                    train_dict['label'],5)
            write_data(args.gbdt_data_path + 'train/' + f1 + '-' + f2 + '_rate_count.csv', train_res)
            write_data(args.gbdt_data_path + 'test1/' + f1 + '-' + f2 + '_rate_count.csv', test1_res)
            write_data(args.gbdt_data_path + 'test2/' + f1 + '-' + f2 + '_rate_count.csv', test2_res)
            print "costs %.1f s" % (time.time() - b_time)
    print "\n\n"

print "counting one times"
user_time_feature = ['uid','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3','appIdAction','appIdInstall']
ad_time_feature = ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
if part == 4:
    for f in user_time_feature:
        print f, "times preparing"
        b_time = time.time()
        train_res, test1_res, test2_res = count_feature_times(train_dict[f], test1_dict[f], test2_dict[f])
        write_data(args.gbdt_data_path + 'train/' + f + '_times_count.csv', train_res, False)
        write_data(args.gbdt_data_path + 'test1/' + f + '_times_count.csv', test1_res, False)
        write_data(args.gbdt_data_path + 'test2/' + f + '_times_count.csv', test2_res, False)
        print "costs %.1f s" % (time.time() - b_time)
    for f in ad_time_feature:
        print f, "times preparing"
        b_time = time.time()
        train_res, test1_res, test2_res = count_feature_times(train_dict[f], test1_dict[f], test2_dict[f])
        write_data(args.gbdt_data_path + 'train/' + f + '_times_count.csv', train_res, False)
        write_data(args.gbdt_data_path + 'test1/' + f + '_times_count.csv', test1_res, False)
        write_data(args.gbdt_data_path + 'test2/' + f + '_times_count.csv', test2_res, False)
        print "costs %.1f s" % (time.time() - b_time)
    print "\n\n"

print "counting combine times"
user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                        'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                        'kw1','kw2','kw3']
finsh_feature_5 = ['LBS','age','carrier','consumptionAbility','education','gender']
finsh_feature_5_2 = ['os','ct','interest1']
finsh_feature_6 = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4']
ad_combine_feature =  ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                       'adCategoryId', 'productId', 'productType']
if part == 5:
    for i, f1 in enumerate(user_combine_feature):
        for f2 in user_combine_feature[i+1:]:
            print f1+'-'+f2+" times preparing"
            b_time = time.time()
            train_res, test1_res, test2_res = count_combine_feature_times(train_dict[f1], train_dict[f2],
                                                                          test1_dict[f1], test1_dict[f2],
                                                                          test2_dict[f1], test2_dict[f2])
            write_data(args.gbdt_data_path + 'train/' + f1 + '-' + f2 + '_times_count.csv', train_res, False)
            write_data(args.gbdt_data_path + 'test1/' + f1 + '-' + f2 + '_times_count.csv', test1_res, False)
            write_data(args.gbdt_data_path + 'test2/' + f1 + '-' + f2 + '_times_count.csv', test2_res, False)
            print "costs %.1f s" % (time.time() - b_time)
if part == 6:
    for i, f1 in enumerate(user_combine_feature):
        for f2 in ad_combine_feature:
            print f1+'-'+f2+" times preparing"
            b_time = time.time()
            train_res, test1_res, test2_res = count_combine_feature_times(train_dict[f1], train_dict[f2],
                                                                          test1_dict[f1], test1_dict[f2],
                                                                          test2_dict[f1], test2_dict[f2])
            write_data(args.gbdt_data_path + 'train/' + f1 + '-' + f2 + '_times_count.csv', train_res, False)
            write_data(args.gbdt_data_path + 'test1/' + f1 + '-' + f2 + '_times_count.csv', test1_res, False)
            write_data(args.gbdt_data_path + 'test2/' + f1 + '-' + f2 + '_times_count.csv', test2_res, False)
            print "costs %.1f s" % (time.time() - b_time)

