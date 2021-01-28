import logging

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import nni

LOG = logging.getLogger('auto-gbdt')


# specify your configurations as a dict
def get_default_parameters():
    params = {
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'binary_logloss', 'auc'},  # 评估函数
        'scale_pos_weight': 1,  # 不平衡数据
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 1,  # k 意味着每 k 次迭代执行bagging
        'verbose': 0  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    return params


def load_data(train_path='./data/train.csv'):
    # 导入数据
    print('Load data...')
    df_train = pd.read_csv(train_path)

    # 处理年龄信息
    df_train['CSNY'] = round(50 - df_train['CSNY'] / (3600 * 24 * 365))

    # 类别特征
    category_names = ['XINGBIE', 'HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY', 'GRZHZT']
    df_train[category_names] = df_train[category_names].astype('category')

    # 划分为训练集、验证集（用于lightgbm训练中的early stopping)、测试集
    df_train, df_test = train_test_split(df_train, test_size=0.3, stratify=df_train['label'], random_state=5)
    df_train, df_eval = train_test_split(df_train, test_size=0.1, stratify=df_train['label'], random_state=5)

    drop_names = ['id', 'XUELI', 'ZHIWU', 'ZHIYE', 'HYZK']
    X_train = df_train.drop(columns=drop_names+['label'], axis=1)
    y_train = df_train['label']
    X_eval = df_eval.drop(columns=drop_names+['label'], axis=1)
    y_eval = df_eval['label']
    X_test = df_test.drop(columns=drop_names+['label'], axis=1)
    y_test = df_test['label']

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

    return lgb_train, lgb_eval, X_test, y_test


def cal_metric(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]

    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


def run(lgb_train, lgb_eval, params, X_test, y_test):
    print('Start training...')

    params['num_leaves'] = int(params['num_leaves'])

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50)

    print('Start predicting...')

    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # eval
    val_metric = cal_metric(y_test, y_pred)
    print('The val_metric of prediction is:', val_metric)

    nni.report_final_result(val_metric)


if __name__ == '__main__':
    lgb_train, lgb_eval, X_test, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)

        # train
        run(lgb_train, lgb_eval, PARAMS, X_test, y_test)
    except Exception as exception:
        LOG.exception(exception)
        raise