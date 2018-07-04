import xgboost as xgb
import sys
from sklearn.metrics import roc_auc_score
from random import random
import numpy as np

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


col_names = get_top_count_feature(args.gbdt_data_path+'analyze/importance_rank.csv',200)
uid_count_feature = ['uid_uid_pos_times_count_5_fold_all','uid_adCategoryId_pos_times_count_5_fold_all',
                 'uid_advertiserId_pos_times_count_5_fold_all','uid_campaignId_pos_times_count_5_fold_all',
                 'uid_creativeId_pos_times_count_5_fold_all','uid_creativeSize_pos_times_count_5_fold_all',
                 'uid_productId_pos_times_count_5_fold_all','uid_productType_pos_times_count_5_fold_all',
                 'uid_adCategoryId_times_count','uid_advertiserId_times_count',
                 'uid_campaignId_times_count','uid_creativeId_times_count',
                 'uid_creativeSize_times_count','uid_productType_times_count',
                 'uid_productId_times_count']
beg = 0
end = 44000000
train_train_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'train/train-', uid_count_feature, beg, end, part=9)
beg = 44000000
end = 46000000
train_test_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'train/train-', uid_count_feature, beg, end, part=9)
beg = 0
end = 34000000
test2_bin_datas = read_batch_data_from_bin(args.dnn_data_path+'test2/test-', uid_count_feature, beg, end, part=3)

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

print train_train_datas.shape, train_test_datas.shape,test2_datas.shape

dtrain = xgb.DMatrix(train_train_datas, train_train_labels)
dtest = xgb.DMatrix(train_test_datas, train_test_labels)


param = {}
param['objective'] = 'binary:logistic'
param['booster'] = 'gbtree'
param['eta'] = 0.1
param['max_depth'] = 8
param['silent'] = 1
param['nthread'] = 16
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['eval_metric'] = 'auc'
# param['tree_method'] = 'exact'
# param['scale_pos_weight'] = 1
num_round = 800

watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=5)

test_datas = xgb.DMatrix(test2_datas)
res = bst.predict(test_datas, ntree_limit=bst.best_ntree_limit)

f = open(args.gbdt_data_path+'result/test2/xgb_test2_res.csv','wb')
for r in res:
    f.write(str(r)+'\n')
