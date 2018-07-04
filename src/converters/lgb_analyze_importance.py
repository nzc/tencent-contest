# coding=utf-8
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
from time import time
import sys

sys.path.append('../')
from utils.donal_args import args


col_names = ['label']
len_feature = ['interest1','interest2','interest3','interest4','interest5',
               'kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall']
for f in len_feature:
    col_names.append(f + '_len')

user_one_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3','appIdAction','appIdInstall']
ad_one_feature = ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
for f in user_one_feature:
    col_names.append(f+'_rate_count')
for f in ad_one_feature:
    col_names.append(f + '_rate_count')

user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3']
ad_combine_feature =  ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
for i, f1 in enumerate(user_combine_feature):
    if f1 == 'interest1':
        break
    for f2 in user_combine_feature[i+1:]:
        col_names.append(f1+'-'+f2+'_rate_count')
col_names.extend(['interest1-interest2_rate_count', 'interest1-interest3_rate_count',
                  'interest1-interest4_rate_count', 'interest1-interest5_rate_count'])
col_names.extend(['interest4-interest5_rate_count', 'interest4-marriageStatus_rate_count',
                  'interest4-topic1_rate_count', 'interest4-topic2_rate_count',
                  'interest4-topic3_rate_count', 'interest4-kw1_rate_count',
                  'interest4-kw2_rate_count', 'interest4-kw3_rate_count',
                  'interest5-marriageStatus_rate_count', 'interest5-topic1_rate_count',
                  'interest5-topic2_rate_count', 'interest5-topic3_rate_count',
                  'topic1-topic2_rate_count', 'topic1-topic3_rate_count'])
for i, f1 in enumerate(user_combine_feature):
    for f2 in ad_combine_feature:
        col_names.append(f1+'-'+f2+'_rate_count')

user_time_feature = ['uid','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                    'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                    'kw1','kw2','kw3','appIdAction','appIdInstall']
ad_time_feature = ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                  'adCategoryId', 'productId', 'productType']
for f in user_time_feature:
    col_names.append(f+'_times_count')
for f in ad_time_feature:
    col_names.append(f+'_times_count')

user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1',
                        'interest2','interest3','interest4','interest5','marriageStatus','topic1','topic2','topic3',
                        'kw1','kw2','kw3']
ad_combine_feature =  ['aid','advertiserId','campaignId', 'creativeId','creativeSize',
                       'adCategoryId', 'productId', 'productType']
for i, f1 in enumerate(user_combine_feature):
    for f2 in user_combine_feature[i+1:]:
        col_names.append(f1+'-'+f2+'_times_count')
for i, f1 in enumerate(user_combine_feature):
    for f2 in ad_combine_feature:
        col_names.append(f1 + '-' + f2 + '_times_count')

def read_data_from_csv(path, beg, end):
    data_arr = []
    f = open(path,'rb')
    num = 0
    for line in f:
        num += 1
        if num < beg:
            continue
        if num > end:
            break
        data_arr.append(float(line.strip()))
    f.close()
    return np.array(data_arr).reshape([-1,1])


def read_batch_data_from_csv(path, col_names, beg = 0, end = 7000000):
    data_arr = []
    num = 0
    for col_name in col_names:
        num += 1
        t = read_data_from_csv(path+col_name+'.csv', beg, end)
        print num, col_name, t.shape
        data_arr.append(t)
    return np.concatenate(data_arr,axis=1)

beg = 0
end = 20000000
train_labels = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[0:1], beg, end)
train_datas = read_batch_data_from_csv(args.gbdt_data_path+'train/', col_names[1:], beg, end)

print train_datas.shape




clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1000, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1, feature_fraction=0.8,
    learning_rate=0.05, min_child_weight=50
)
clf.fit(train_datas, train_labels, eval_set=[(train_datas, train_labels)], eval_metric='logloss',early_stopping_rounds=100)

dict = {}
for i,col_name in enumerate(col_names[1:]):
    dict[col_name] = clf.feature_importances_[i] * 100

f = open(args.gbdt_data_path+'analyze/' + 'importance_rank.csv','wb')
f.write('name,value\n')
sort_dict = sorted(dict.items(),key = lambda x:x[1],reverse = True)
for name, val in sort_dict:
    f.write(name + ',' + str(val) + '\n')
f.close()



