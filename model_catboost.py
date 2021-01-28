from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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



# 类别特征
category_names = ['XINGBIE', 'HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY', 'GRZHZT']
df_train[category_names] = df_train[category_names].astype('category')
df_test[category_names] = df_test[category_names].astype('category')

print(df_train.head())
#print(df_test.info())

# 生成训练集、验证集和测试集
df_train, df_eval = train_test_split(df_train, test_size=0.2, stratify=df_train['label'], random_state=5)

drop_names = ['id', 'XUELI', 'ZHIWU', 'ZHIYE', 'HYZK', 'ZHICHEN']
X_train = df_train.drop(columns=drop_names + ['label'], axis=1)
y_train = df_train['label']
X_eval = df_eval.drop(columns=drop_names + ['label'], axis=1)
y_eval = df_eval['label']
X_test = df_test.drop(columns=drop_names, axis=1)

print(X_train.info())

# 训练 catboost 模型
categorical_features_indices = np.where(X_train.dtypes == 'category')[0].tolist()
model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, loss_function='Logloss',
                            logging_level='Verbose')
model.fit(X_train, y_train, eval_set=(X_eval, y_eval), cat_features=categorical_features_indices, plot=True)

# 输出各特征重要度
feature_names = X_train.columns
print(pd.DataFrame({
       'column': feature_names,
       'importance': model.get_feature_importance(),
   }).sort_values(by='importance', ascending=False))

print(model.get_best_iteration())

# 训练集和验证集的预测
y_train_prob = model.predict_proba(X_train, ntree_end=model.get_best_iteration())[:, 1]
y_eval_prob = model.predict_proba(X_eval, ntree_end=model.get_best_iteration())[:, 1]

# AUC指标
print('AUC')
print(roc_auc_score(y_train, y_train_prob))
print(roc_auc_score(y_eval, y_eval_prob))

# 本题采用的指标
print('本题采用的指标')
print(cal_metric(y_train, y_train_prob))
print(cal_metric(y_eval, y_eval_prob))


# 模型预测并写入csv(此处记得将前面的验证集df_eval改为0， 用全量的df_train，并设置迭代次数为best_iteration）
y_prob = model.predict_proba(X_test, ntree_end=model.get_best_iteration())[:, 1]
np.savetxt('./data/result_catboost.csv', y_prob, fmt='%f')

