import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


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


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

# 处理年龄信息
df_train['CSNY'] = round(50 - df_train['CSNY'] / (3600 * 24 * 365))
df_test['CSNY'] = round(50 - df_test['CSNY'] / (3600 * 24 * 365))

# 增加一些特征
df_train['1'] = df_train['DKYE'] / df_train['DKFFE']
df_test['1'] = df_test['DKYE'] / df_test['DKFFE']

df_train['2'] = (df_train['GRYJCE'] + df_train['DWYJCE']) / df_train['DKFFE']
df_test['2'] = (df_test['GRYJCE'] + df_test['DWYJCE']) / df_test['DKFFE']


df_train['3'] = (df_train['GRYJCE'] + df_train['DWYJCE']) / df_train['DKYE']
df_test['3'] = (df_test['GRYJCE'] + df_test['DWYJCE']) / df_test['DKYE']

df_train['4'] = df_train['DWYJCE'] / df_train['GRJCJS']
df_test['4'] = df_test['DWYJCE'] / df_test['GRJCJS']


print(df_train.head())
# 类别特征
category_names = ['XINGBIE', 'HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY', 'GRZHZT']
for col in category_names:
    le = LabelEncoder()
    le.fit(np.concatenate([df_train[col], df_test[col]]))
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

df_train[category_names] = df_train[category_names].astype('category')
df_test[category_names] = df_test[category_names].astype('category')


print(df_train.head())
print(df_train.info())

# 生成训练集、验证集和测试集
df_train, df_eval = train_test_split(df_train, test_size=0.2, stratify=df_train['label'], random_state=5)

drop_names = ['id', 'XUELI', 'ZHIWU', 'ZHIYE', 'HYZK', 'ZHICHEN']
X_train = df_train.drop(columns=drop_names + ['label'], axis=1)
y_train = df_train['label']
X_eval = df_eval.drop(columns=drop_names + ['label'], axis=1)
y_eval = df_eval['label']
X_test = df_test.drop(columns=drop_names, axis=1)

# 转换为Dataset数据格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

# 参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',  # 目标函数
    'metric': {'binary_logloss', 'auc'},  # 评估函数
    'scale_pos_weight': 1,  # 不平衡数据
    'num_boost_round': 400,  # 树的个数
    'num_leaves': 42,  # 叶子节点数
    'learning_rate': 0.01,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 1,  # k 意味着每 k 次迭代执行bagging
    'verbose': 0  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# 模型训练
#gbm = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=50)
gbm = lgb.train(params, lgb_train)
# 输出各特征重要度
feature_names = X_train.columns
print(pd.DataFrame({
       'column': feature_names,
       'importance': gbm.feature_importance(),
   }).sort_values(by='importance', ascending=False))



# 训练集和验证集的预测
# y_train_prob = gbm.predict(X_train, num_iteration=gbm.best_iteration)
# y_eval_prob = gbm.predict(X_eval, num_iteration=gbm.best_iteration)

y_train_prob = gbm.predict(X_train)
y_eval_prob = gbm.predict(X_eval)

# AUC指标
print('AUC')
print(roc_auc_score(y_train, y_train_prob))
print(roc_auc_score(y_eval, y_eval_prob))

# 本题采用的指标
print('本题采用的指标')
print(cal_metric(y_train, y_train_prob))
print(cal_metric(y_eval, y_eval_prob))

# 模型预测并写入csv (此处记得将前面的验证集df_eval改为0， 用全量的df_train，并设置迭代次数为best_iteration）
#y_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_prob = gbm.predict(X_test)
np.savetxt('./data/result_lightgbm.csv', y_prob, fmt='%f')

