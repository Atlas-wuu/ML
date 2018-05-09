# -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import scipy.io as scio
import pandas as pd

import pandas as pd
# read in data
traindata = np.genfromtxt('../data/train_final.csv', delimiter=',')
testdata = np.genfromtxt('../data/test_final.csv', delimiter=',')
a=traindata[1:, 1]
print(traindata[1:, 1])
print min(a)
b=testdata[1:, 1]
dtrain1 = xgb.DMatrix(traindata[1:, 6:], traindata[1:, 1])
dtest1 = xgb.DMatrix(testdata[1:, 6:], testdata[1:, 1])
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
watchlist  = [(dtest1,'eval'), (dtrain1,'train')]
num_round = 8000
bst = xgb.train(param, dtrain1, num_round, evals = watchlist)
# make prediction
# ntree_limit must not be 0
print('2')
preds = bst.predict(dtest1, ntree_limit=num_round)
# scio.savemat('result_our3_1.mat', {'preds':preds})
np.savetxt('../data/result_our1.csv',preds,delimiter=',')
