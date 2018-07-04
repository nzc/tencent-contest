# coding=utf-8
"""
    拼表和生成csv文件

"""

import numpy as np
import pandas as pd
import os
import sys

from time import  time

sys.path.append('../../')
from utils.donal_args import args

state = int(sys.argv[1]) # state=0：复赛数据拼表； state!=0：初赛数据拼表
if state == 0:
    ad_feature_path = args.ad_feature_path
    user_feature_path = args.user_feature_path
    raw_train_path = args.raw_train_path
    raw_test1_path = args.raw_test1_path
    raw_test2_path = args.raw_test2_path
    combine_train_path = args.combine_train_path
    combine_test1_path = args.combine_test1_path
    combine_test2_path = args.combine_test2_path
else:
    ad_feature_path = args.chusai_ad_feature_path
    user_feature_path = args.chusai_user_feature_path
    raw_train_path = args.chusai_raw_train_path
    raw_test1_path = args.chusai_raw_test1_path
    raw_test2_path = args.chusai_raw_test2_path
    combine_train_path = args.chusai_combine_train_path
    combine_test1_path = args.chusai_combine_test1_path
    combine_test2_path = args.chusai_combine_test2_path

ad_feature=pd.read_csv(ad_feature_path)

userFeature_data = []
user_feature = None
with open(user_feature_path, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        userFeature_data.append(userFeature_dict)
        if i % 100000 == 0:
            print(i)
    user_feature = pd.DataFrame(userFeature_data)
user_feature['uid'] = user_feature['uid'].apply(int)

train=pd.read_csv(raw_train_path)
predict1=pd.read_csv(raw_test1_path)
predict2 = pd.read_csv(raw_test2_path)
train.loc[train['label']==-1,'label']=0
predict1['label']=-1
predict2['label']=-2
data=pd.concat([train,predict1,predict2])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')

train = data[data.label != -1][data.label != -2]

test1 = data[data.label == -1]
test2 = data[data.label == -2]

train.to_csv(combine_train_path, index=False)
test1.to_csv(combine_test1_path, index=False)
test2.to_csv(combine_test2_path, index=False)




