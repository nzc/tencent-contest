# coding=utf-8
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import sys
from time import time

sys.path.append('../')
from utils.donal_args import args

def get_top_count_feature(path, num=50):
    col_names = ['label']
    f = open(path, 'rb')
    f.readline()
    i = 0
    for line in f:
        i += 1
        if i > num:
            break
        col_names.append(line.strip().split(',')[0])
    return col_names

def read_data_from_csv(path, beg, end):
    data_arr = []
    f = open(path,'rb')
    num = 0
    for line in f:
        num += 1
        if num <= beg:
            continue
        if num > end:
            break
        data_arr.append(float(line.strip()))
    f.close()
    return np.array(data_arr).reshape([-1,1])

def read_data_from_bin(path, col_name, beg, end, part):
    data_arr = []
    for i in range(part):
        res = np.fromfile(path+col_name+'_' + str(i)+'.bin', dtype=np.int).astype(np.float).reshape([-1,1])
        data_arr.append(res)
    return np.concatenate(data_arr,axis=0)[beg:end]

def read_batch_data_from_csv(path, col_names, beg = 0, end = 7000000):
    data_arr = []
    num = 0
    for col_name in col_names:
        num += 1
        t = read_data_from_csv(path+col_name+'.csv', beg, end)
        print num, col_name, t.shape
        data_arr.append(t)
    return np.concatenate(data_arr,axis=1)

def read_batch_data_from_bin(path, col_names, beg = 0, end = 7000000, part=9):
    data_arr = []
    num = 0
    for col_name in col_names:
        num += 1
        t = read_data_from_bin(path, col_name, beg, end, part)
        print num, col_name, t.shape
        data_arr.append(t)
    return np.concatenate(data_arr,axis=1)


col_names = get_top_count_feature(args.gbdt_data_path+'analyze/importance_rank.csv',400)
uid_count_feature = ['uid_uid_pos_times_count_5_fold_all','uid_adCategoryId_pos_times_count_5_fold_all',
                 'uid_advertiserId_pos_times_count_5_fold_all','uid_campaignId_pos_times_count_5_fold_all',
                 'uid_creativeId_pos_times_count_5_fold_all','uid_creativeSize_pos_times_count_5_fold_all',
                 'uid_productId_pos_times_count_5_fold_all','uid_productType_pos_times_count_5_fold_all',
                 'uid_adCategoryId_times_count','uid_advertiserId_times_count',
                 'uid_campaignId_times_count','uid_creativeId_times_count',
                 'uid_creativeSize_times_count','uid_productType_times_count',
                 'uid_productId_times_count']
uid_count_feature_with_chusai = ['log_uid_uid_pos_times_count_5_fold_with_chusai','log_uid_adCategoryId_pos_times_count_5_fold_with_chusai',
                 'log_uid_advertiserId_pos_times_count_5_fold_with_chusai','log_uid_campaignId_pos_times_count_5_fold_with_chusai',
                 'log_uid_creativeId_pos_times_count_5_fold_with_chusai','log_uid_creativeSize_pos_times_count_5_fold_with_chusai',
                 'log_uid_productId_pos_times_count_5_fold_with_chusai','log_uid_productType_pos_times_count_5_fold_with_chusai',
                 'log_uid_adCategoryId_times_count_with_chusai','log_uid_advertiserId_times_count_with_chusai',
                 'log_uid_campaignId_times_count_with_chusai','log_uid_creativeId_times_count_with_chusai',
                 'log_uid_creativeSize_times_count_with_chusai','log_uid_productType_times_count_with_chusai',
                 'log_uid_productId_times_count_with_chusai']
beg = 0
end = 44000000
train_train_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'train/train-', uid_count_feature_with_chusai, beg, end, part=9)
beg = 44000000
end = 46000000
train_test_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'train/train-', uid_count_feature_with_chusai, beg, end, part=9)
beg = 0
end = 34000000
test2_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'test2/test-', uid_count_feature_with_chusai, beg, end, part=3)

beg = 0
end = 44000000
train_train_labels = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[0:1], beg, end)
train_train_datas = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[1:], beg, end)
beg = 44000000
end = 46000000
train_test_labels = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[0:1], beg, end)
train_test_datas = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[1:], beg, end)

# beg = 0
# end = 34000000
# test1_datas = read_batch_data_from_csv(args.gbdt_data_path+'test1/', col_names[1:], beg, end)

beg = 0
end = 34000000
test2_datas = read_batch_data_from_csv(args.gbdt_data_path+'test2/', col_names[1:], beg, end)

train_train_datas = np.concatenate([train_train_datas,train_train_bin_datas],axis=1)
train_test_datas = np.concatenate([train_test_datas,train_test_bin_datas], axis=1)
test2_datas = np.concatenate([test2_datas, test2_bin_datas], axis=1)

print train_train_datas.shape

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1200, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1, feature_fraction=0.8,
    learning_rate=0.05, min_child_weight=50
)
clf.fit(train_train_datas, train_train_labels, eval_set=[(train_train_datas, train_train_labels),(train_test_datas, train_test_labels)], eval_metric='auc',early_stopping_rounds=100)

# test1_res = clf.predict_proba(test1_datas)[:,1]
# f = open(args.gbdt_data_path+'result/test1/lgb_test1_res.csv','wb')
# for r in test1_res:
#     f.write('%.6f' % (r)+'\n')
# f.close()

test2_res = clf.predict_proba(test2_datas)[:,1]
f = open(args.gbdt_data_path+'result/test2/lgb_test2_res_2.csv','wb')
for r in test2_res:
    f.write('%.6f' % (r)+'\n')
f.close()
