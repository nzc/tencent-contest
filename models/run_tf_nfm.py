# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import sys
import itertools


from tf_NFM import NFM

test_out_name = sys.argv[1]
args_name = sys.argv[2]
assert args_name in ['donal_args','args','args2','args3','nzc_args']

sys.path.append('../')
from utils.tencent_data_func import *
if args_name == 'donal_args':
    from utils.donal_args import args
elif args_name == 'args':
    from utils.args import args
elif args_name == 'args2':
    from utils.args2 import args
elif args_name == 'args3':
    from utils.args3 import args
elif args_name == 'nzc_args':
    from utils.nzc_args import args



#param
field_sizes = [len(args.static_features), len(args.dynamic_features)]
dynamic_max_len_dict = args.dynamic_features_max_len_dict
learning_rate = args.lr
epochs = args.epochs
batch_size = args.batch_size

train_parts = [0,1,2,3,4,5,6,7,8]
test_parts = [7,8]
test1_parts = [0,1,2]
test2_parts = [0,1,2]

y, static_index, dynamic_index, dynamic_lengths, extern_lr_index = \
   load_concatenate_tencent_data_to_dict(args.dnn_data_path+'train/train-', train_parts,args.static_features,
                                         args.dynamic_features, is_csv=False)

# for key in args.static_features:
#     print key, static_index[key].shape
#
# for key in args.dynamic_features:
#     print key, dynamic_index[key].shape

# valid_y, valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_exctern_lr_index = \
#     load_concatenate_tencent_data_to_dict(args.dnn_data_path+'train/train-', test_parts)

dynamic_total_size_dict = load_dynamic_total_size_dict(args.dnn_data_path+'dict/',args.dynamic_features)
static_total_size_dict = load_static_total_size_dict(args.dnn_data_path+'dict/', args.static_features)


exclusive_cols = args.exclusive_cols

# test1_y, test1_static_index, test1_dynamic_index, test1_dynamic_lengths, test1_extern_lr_index = \
#     load_concatenate_tencent_data_to_dict(args.dnn_data_path+'test1/test-', test1_parts, is_csv=False)

test2_y, test2_static_index, test2_dynamic_index, test2_dynamic_lengths, test2_extern_lr_index = \
    load_concatenate_tencent_data_to_dict(args.dnn_data_path+'test2/test-', test2_parts,args.static_features,
                                          args.dynamic_features)


# y1_pred = np.array([0.0] * len(test1_y))
y2_pred = np.array([0.0] * len(test2_y))

for i in range(1):
    dfm = NFM(field_sizes=field_sizes, static_total_size_dict=static_total_size_dict, dynamic_total_size_dict=dynamic_total_size_dict,
              dynamic_max_len_dict=dynamic_max_len_dict, exclusive_cols=args.exclusive_cols,learning_rate=learning_rate,
              deep_layers=[64, 64], epoch=1, batch_size=batch_size,optimizer_type='adam')
    # dfm.fit(static_index, dynamic_index, dynamic_lengths, y,
    #         valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_y, combine=False)
    dfm.fit(static_index, dynamic_index, dynamic_lengths, y, show_eval=False, is_shuffle=True)
    # y1_pred += dfm.predict(test1_static_index, test1_dynamic_index, test1_dynamic_lengths)
    y2_pred += dfm.predict(test2_static_index, test2_dynamic_index, test2_dynamic_lengths)

# y1_pred /= 3.0
# y2_pred /= 3.0

# f = open(args.dnn_data_path+'result/test1/'+test_out_name+'.csv', 'wb')
# for y in y1_pred:
#     f.write('%.6f' % (y) + '\n')
# f.close()

f = open(args.dnn_data_path+'result/test2/'+test_out_name+'.csv', 'wb')
for y in y2_pred:
    f.write('%.6f' % (y) + '\n')
f.close()






