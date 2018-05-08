# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# import data_code_final
# import xgbt_test1
# import xgbt_test2
# import xgbt_test3
# import xgbt_test4
# import xgbt_test5
# data_code_final
# xgbt_test1
# xgbt_test2
# xgbt_test3
# xgbt_test4
# xgbt_test5

part_c = pd.read_csv('../data/vid.csv',header=None)
part_1 = pd.read_csv('../data/result_our1.csv',header=None)
part_2 = pd.read_csv('../data/result_our2.csv',header=None)
part_3 = pd.read_csv('../data/result_our3.csv',header=None)
part_4 = pd.read_csv('../data/result_our4.csv',header=None)
part_5 = pd.read_csv('../data/result_our5.csv',header=None)

part_result = pd.concat([part_c,part_1,part_2,part_3,part_4,part_5],axis=1, join='outer')
np.savetxt('../submit/submit_20180507_114537.csv',part_result,fmt="%s",delimiter=',')

