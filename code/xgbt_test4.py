# -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import scipy.io as scio

import pandas as pd
# read in data
# data = pd.read_csv('/home/wu/Documents/tianchi/double high/data_clean/train_only_num_clean_neg.csv', sep=',', encoding='utf-8')
traindata = np.genfromtxt('../data/train_final.csv', delimiter=',')
testdata = np.genfromtxt('../data/test_final.csv', delimiter=',')

dtrain4 = xgb.DMatrix(traindata[1:, 6:], traindata[1:, 4])
dtest4 = xgb.DMatrix(testdata[1:, 6:], testdata[1:, 4])
# ts_num=data[30000:, 9:].shape[0]
param = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,#在树的叶节点上进行进一步分区所需的最小损失减少量。 算法越大，越保守。
    'max_depth': 5,#数的最大深度
    'lambda': 2,  #L2 正则的惩罚系数
    'subsample': 0.8,#用于训练模型的子样本占整个样本集合的比例
    'colsample_bytree': 0.8,#在建立树时对特征采样的比例
    'min_child_weight': 2,#孩子节点中最小的样本权重和。
    'silent': 1,
    'eta': 0.008,#为了防止过拟合，更新过程中用到的收缩步长
    'seed': 30,
}
print('1')

# param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'base_score':3}
watchlist  = [(dtest4,'eval'), (dtrain4,'train')]
num_round = 8000
bst = xgb.train(param, dtrain4, num_round, evals = watchlist)
# make prediction
# ntree_limit must not be 0
print('2')
preds = bst.predict(dtest4, ntree_limit=num_round)
# labels = dtest1.get_label()
# a = np.sum(np.square(np.log(preds+1) - np.log(labels+1)))/ts_num
# print ('error=%f' % (np.sum(np.square(np.log(preds+1) - np.log(labels+1)))/ts_num))
# scio.savemat('result_our3_4.mat', {'preds':preds})
np.savetxt('../data/result_our4.csv',preds,delimiter=',')

