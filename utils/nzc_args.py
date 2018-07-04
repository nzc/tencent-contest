"""
args parameters
"""
import itertools



class args:

    root_data_path = '../data/'

    """
        converters args
    """
    ad_feature_path = root_data_path + 'adFeature.csv'
    user_feature_path = root_data_path + 'userFeature.data'
    raw_train_path = root_data_path + 'train.csv'
    raw_test1_path = root_data_path + 'test1.csv'
    raw_test2_path = root_data_path + 'test2.csv'
    combine_train_path = root_data_path + 'combine_train.csv'
    combine_test1_path = root_data_path + 'combine_test1.csv'
    combine_test2_path = root_data_path + 'combine_test2.csv'
    random_combine_train_path = root_data_path + 'random_combine_train.csv'

    gbdt_data_path = root_data_path + 'gbdt/'

    dnn_data_path = root_data_path + 'nzc_dnn/'

    """
        raw feature
    """
    dynamic_dict = {'interest1':61, 'interest2':33, 'interest3':10,
                    'interest4':10, 'interest5':86, 'kw1':5, 'kw2':5, 'kw3':5,
                    'topic1':5, 'topic2':5, 'topic3':5, 'appIdInstall':908,
                    'appIdAction':823, 'ct':4, 'os':2, 'marriageStatus':3}

    user_static_features = ['uid', 'house', 'education', 'LBS', 'consumptionAbility',
                            'gender', 'age', 'carrier']
    ad_static_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                          'adCategoryId','productId', 'productType']
    user_dynamic_features = [key for key in dynamic_dict]

    """
        dnn args
    """
    # data processing part

    static_features = ['log_uid_uid_times_count','interest1_len','interest2_len','interest5_len','aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId',
                       'productId', 'productType', 'LBS', 'age', 'carrier',
                       'consumptionAbility', 'education', 'gender', 'house']
    uid_count_feature = ['log_uid_uid_pos_times_count_5_fold_all_set', 'log_uid_adCategoryId_pos_times_count_5_fold_all_set',
                         'log_uid_advertiserId_pos_times_count_5_fold_all_set', 'log_uid_campaignId_pos_times_count_5_fold_all_set',
                         'log_uid_creativeId_pos_times_count_5_fold_all_set', 'log_uid_creativeSize_pos_times_count_5_fold_all_set',
                         'log_uid_productId_pos_times_count_5_fold_all_set', 'log_uid_productType_pos_times_count_5_fold_all_set',
                         'log_uid_adCategoryId_times_count', 'log_uid_advertiserId_times_count',
                         'log_uid_campaignId_times_count', 'log_uid_creativeId_times_count',
                         'log_uid_creativeSize_times_count', 'log_uid_productType_times_count',
                         'log_uid_productId_times_count']
    static_features.extend(uid_count_feature)

    dynamic_features = ['kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3','marriageStatus',
                        'os', 'ct', 'interest1', 'interest2',
                        'interest3', 'interest4', 'interest5']

    dynamic_features_max_len_dict = {'interest1':61, 'interest2':33, 'interest3':10,
                                     'interest4':10, 'interest5':86, 'kw1':5, 'kw2':5, 'kw3':5,
                                     'topic1':5, 'topic2':5, 'topic3':5, 'ct':4, 'os':2, 'marriageStatus':3,
                                     'os_aid':2, 'ct_aid':4, 'marriageStatus_aid':3,"interest1_aid":61,
                                     'interest2_aid':33, 'interest3_aid':10, 'interest4_aid':10, 'interest5_aid':86,
                                     'LBS_kw1':5, 'topic1_topic2':25}
    aid_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house',
                           'os', 'ct', 'marriageStatus', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5']
    uid_pos_feature = ['uid_pos_times_count_5_fold','log_uid_pos_times_count_5_fold',
                       'uid_pos_times_count_10_fold','log_uid_pos_times_count_10_fold',
                       'uid_pos_times_count_5_fold_all','log_uid_pos_times_count_5_fold_all',
                       'uid_pos_times_count_10_fold_all', 'log_uid_pos_times_count_10_fold_all']
    exclusive_cols = []
    exclusive_cols.extend(itertools.permutations(['aid', 'advertiserId', 'campaignId', 'creativeId',
                                                  'creativeSize', 'adCategoryId', 'productId', 'productType'], 2))
    for f in aid_combine_feature[9:]:
        exclusive_cols.append(('aid',f+'_aid'))
        exclusive_cols.append((f + '_aid', 'aid'))
        exclusive_cols.append((f, f + '_aid'))
        exclusive_cols.append((f + '_aid', f))
    for f1 in uid_pos_feature:
        for f2 in aid_combine_feature:
            exclusive_cols.extend([(f1,f2+'_aid'),(f2+'_aid',f1)])
        exclusive_cols.extend([(f1,'LBS_kw1'), ('LBS_kw1', f1), ('topic1_topic2',f1),(f1,'topic1_topic2')])
    exclusive_cols.extend([('kw1', 'LBS_kw1'), ('LBS_kw1', 'kw1'), ('LBS','LBS_kw1'), ('LBS_kw1','LBS')])
    exclusive_cols.extend([('topic1','topic1_topic2'),('topic1_topic2','topic1'),('topic2','topic1_topic2'),('topic1_topic2','topic2')])
    exclusive_cols.extend([('interest1_len','interest1'),('interest1','interest1_len'),('interest2','interest2_len'),('interest2_len','interest2')])
    exclusive_cols.extend([('advertiserId','log_uid_advertiserId_times_count'),('log_uid_advertiserId_times_count','advertiserId')])
    exclusive_cols.extend([('advertiserId', 'uid_advertiserId_times_count'), ('uid_advertiserId_times_count', 'advertiserId')])
    exclusive_cols.extend([('LBS_kw1','uid_pos_times_count_5_fold'),('uid_pos_times_count_5_fold','LBS_kw1')])



    extern_lr_features = []

    batch_size = 1024
    epochs = 1

    lr = 0.0002
