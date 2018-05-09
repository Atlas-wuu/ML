# -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import scipy.io as scio

import pandas as pd
# read in data
# data = pd.read_csv('/home/wu/Documents/tianchi/double high/data_clean/train_only_num_clean_neg.csv', sep=',', encoding='utf-8')
traindata = np.genfromtxt('../data/train_final.csv', delimiter=',')
testdata = np.genfromtxt('../data/test_final.csv', delimiter=',')

dtrain5 = xgb.DMatrix(traindata[1:, 6:], traindata[1:, 5])
dtest5 = xgb.DMatrix(testdata[1:, 6:], testdata[1:, 5])
# ts_num=data[30000:, 9:].shape[0]
param = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0,#在树的叶节点上进行进一步分区所需的最小损失减少量。 算法越大，越保守。
    'max_depth': 5,#数的最大深度
    'lambda': 2,  #L2 正则的惩罚系数
    'subsample': 0.8,#用于训练模型的子样本占整个样本集合的比例
    'colsample_bytree': 0.8,#在建立树时对特征采样的比例
    'min_child_weight': 2,#孩子节点中最小的样本权重和。
    'silent': 1,
    'eta': 0.006,#为了防止过拟合，更新过程中用到的收缩步长
    'seed': 30,
}
print('1')

# param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'basparam_dist = {
#     'booster': 'gbtree',
#     'objective': 'reg:gamma',
#     'gamma': 0.3,#在树的叶节点上进行进一步分区所需的最小损失减少量。 算法越大，越保守。
#     'max_depth': 5,#数的最大深度
#     'lambda': 2,  #L2 正则的惩罚系数
#     'subsample': 0.7,#用于训练模型的子样本占整个样本集合的比例
#     'colsample_bytree': 0.7,#在建立树时对特征采样的比例
#     'min_child_weight': 2,#孩子节点中最小的样本权重和。
#     'silent': 1,
#     'eta': 0.07,#为了防止过拟合，更新过程中用到的收缩步长
#     'seed': 30,
# }e_score':3}
watchlist  = [(dtest5,'eval'), (dtrain5,'train')]
num_round = 8000
bst = xgb.train(param, dtrain5, num_round, evals = watchlist)
# make prediction
# ntree_limit must not be 0
print('2')
preds = bst.predict(dtest5, ntree_limit=num_round)
# labels = dtest1.get_label()
# a = np.sum(np.square(np.log(preds+1) - np.log(labels+1)))/ts_num
# print ('error=%f' % (np.sum(np.square(np.log(preds+1) - np.log(labels+1)))/ts_num))
np.savetxt('../data/result_our5.csv',preds,delimiter=',')

