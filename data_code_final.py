# coding=utf-8

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import time
import re
import xgboost as xgb
import numpy as np
import scipy.io as scio

# part 1
start_time=time.time()
# 读取数据
train=pd.read_csv('../data/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',sep=',',encoding='gbk')
data_part1=pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
data_part2=pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep='$',encoding='utf-8')

# data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
part1_2 = pd.concat([data_part1,data_part2],axis=0)#{0/'index', 1/'columns'}, default 0
# part1_2 = data_part2#{0/'index', 1/'columns'}, default 0
part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
vid_set=pd.concat([train['vid'],test['vid']],axis=0)
vid_set=pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
part1_2=part1_2[part1_2['vid'].isin(vid_set['vid'])]

# 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
def filter_None(data):
    data=data[data['field_results']!='']
    data=data[data['field_results']!='未查']
    return data

part1_2=filter_None(part1_2)

# 过滤列表，过滤掉不重要的table_id 所在行
filter_list=['0203','0209','0702','0703','0705','0706','0709','0726','0730','0731','3601',
             '1308','1316']

part1_2=part1_2[~part1_2['table_id'].isin(filter_list)]

# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

# 数据简单处理
print(part1_2.shape)
vid_tabid_group = part1_2.groupby(['vid','table_id']).size().reset_index()
# print(vid_tabid_group.head())
# print(vid_tabid_group.shape)
#                      vid               table_id  0
# 0  000330ad1f424114719b7525f400660b     0101     1
# 1  000330ad1f424114719b7525f400660b     0102     3

# 重塑index用来去重,区分重复部分和唯一部分
print('------------------------------去重和组合-----------------------------')
vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']

# print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']

dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
dup_part = dup_part.sort_values(['vid','table_id'])
unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]

part1_2_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part1_2_dup.rename(columns={0:'field_results'},inplace=True)
part1_2_res = pd.concat([part1_2_dup,unique_part[['vid','table_id','field_results']]])

table_id_group=part1_2.groupby('table_id').size().sort_values(ascending=False)
table_id_group.to_csv('../data/part_tabid_size.csv',encoding='utf-8')

# 行列转换
print('--------------------------重新组织index和columns---------------------------')
merge_part1_2 = part1_2_res.pivot(index='vid',values='field_results',columns='table_id')
print('--------------新的part1_2组合完毕----------')
print(merge_part1_2.shape)
merge_part1_2.to_csv('../data/merge_part1_2.csv',encoding='utf-8')
print(merge_part1_2.head())
del merge_part1_2, data_part1, data_part2

time.sleep(10)

#  part 2

print('------------------------重新读取数据merge_part1_2--------------------------')
merge_part1_2=pd.read_csv('../data/merge_part1_2.csv',sep=',',encoding='utf-8')
#

# 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
def remain_feat(df,thresh=0.8):
    exclude_feats = []
    print('----------移除数据缺失多的字段-----------')
    print("移除之前总的字段数量: %s" % len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent > thresh:
            exclude_feats.append(c)
    print("移除缺失数据的字段数量: %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print("剩余的字段数量: %s" % len(feats))
    return feats

feats=remain_feat(merge_part1_2,thresh=0.92)


merge_part1_2=merge_part1_2[feats]
print(merge_part1_2.shape)
# merge_part1_2.to_csv('../data/merge_part1_2_ftr092.csv',index=False,encoding='utf-8')

# 找到train，test各自属性进行拼接
train_of_part=merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
test_of_part=merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]

train=pd.merge(train,train_of_part,on='vid')
test=pd.merge(test,test_of_part,on='vid')
#
# # 清洗训练集中的五个指标
def clean_label1(x):

    if x<=0.0:
        x=np.nan

    x=str(x)
    if '+' in x:#16.04++
        i=x.index('+')
        x=x[0:i]
    if '>' in x:#> 11.00
        i=x.index('>')
        x=x[i+1:]
    if len(x.split('.'))>2:#2.2.8  str.split("&", 8)
        i=x.rindex('.')
        x=x[0:i]+x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        x=np.nan
    if str(x).isdigit()==False and len(str(x))>4:
        x=x[0:4]
    return x

def clean_label2(x):

    if x<=0.0:
        x=np.nan
    return x


# 数据清洗
def data_clean(df):
    crx = df.columns
    crx1 = crx[1:6]
    # for c in ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']:
    for c in crx1:
        print min(df[c])

        df[c]=df[c].apply(clean_label1)
        df[c]=df[c].astype('float64')
        df[c]=df[c].apply(clean_label2)
        df[c]=df[c].astype('float64')

        print min(df[c])
    return df
train_clean=data_clean(train)
print train_clean.shape
print('---------------保存train_set和test_set---------------------')
train_clean.to_csv('../data/train_set_clean_092_0505data.csv',index=False,encoding='utf-8')
test.to_csv('../data/test_set_092_0505data.csv',index=False,encoding='utf-8')

end_time=time.time()
print('程序总共耗时:%d 秒'%int(end_time-start_time))

#  part 3

# from base_func import remain_feat
start_time=time.time()
# 读取数据
train=pd.read_csv('../data/train_set_clean_092_0505data.csv',sep=',',encoding='utf-8')
# print(train.columns)
test=pd.read_csv('../data/test_set_092_0505data.csv',sep=',',encoding='utf-8')

print(test.columns)
test_only_number=pd.read_csv('number_v1_50sub.csv',sep=',',encoding='utf-8')

# test_only_number.rename(columns={'4997':'004997', '424':'0424', '425':'0425'}, inplace = True)
number_col=test_only_number.columns
print(test_only_number.columns)


train_only_num=train[number_col]
test_only_num=test[number_col]

train_only_num.to_csv('../data/train_only_num_0_data0505.csv',index=False,encoding='utf-8')
test_only_num.to_csv('../data/test_only_num_0_data0505.csv',index=False,encoding='utf-8')
# data_part1=pd.read_csv('data/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
# data_part2=pd.read_csv('data/meinian_round1_data_part2_20180408.txt',sep='$',encoding='utf-8')
#
# test_1to2=test[1:51]
# test_1to2.to_csv('code_3_org/test_1to2.csv',index=False,encoding='utf-8')
#
# global num_len

# 修改标签
def modify_label(x):
    # 1. quanjiaobanjiao 2. get first num 3. modify
    x=str(x)
    # print(len(x))
    # num_len = num_len+len(x)

    if '０' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[０]', '0', x.decode())
        print(x)
    if '１' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[１]', '1', x.decode())
        print(x)
    if '２' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[２]', '2', x.decode())
        print(x)
    if '３' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[３]', '3', x.decode())
        print(x)
    if '４' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[４]', '4', x.decode())
        print(x)
    if '５' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[５]', '5', x.decode())
        print(x)
    if '６' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[６]', '6', x.decode())
        print(x)
    if '７' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[７]', '7', x.decode())
        print(x)
    if '８' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[８]', '8', x.decode())
        print(x)
    if '９' in x:
        print('x_org: %s' %x)
        x = re.sub(u'[９]', '9', x.decode())
        print(x)

    if x!='nan':
        a = re.findall(r"\d+\.?\d*", x)
        # print a[0]
        print('x_org: %s' %x)
        print(a)
        # if a!=[]:
        #     x=a[0]
        if a==[]:
            # if ('未见异常' in x )or ('心率正常' in x)or ('正常' in x)or ('正常 正常' in x)or ('右眼压检测正常' in x)or ('左眼压检测正常' in x):
            # if ('未见异常' in x )or ('正常' in x):
            #     x='正常'
            if ('未见异' in x )or ('正常' in x)or ('未见明显异常' in x)\
                    or ('整齐' in x)or (x=='齐')or ('无压痛' in x):
                x='正常'
            elif x == '无' or x == '代检' or x == '自述不查' or x == '未触及' or x == 'exit' or x == '未查' or x == '弃查' or x == '无 无':
                x = np.nan

            # elif 'NaN' in x:
            #     x = np.nan

            else :
                x = 1

        else:
            x=a[0]


        # x=np.mean(a)
        print(x)


    return x

# 数据清洗
def data_modify(df):
    # 1.get mean num 2. clean 3. float64
    crx = df.columns
    crx1=crx[7:]

    # crx1 = crx[1:6]
    for c in crx1:
        num_len=0
        a = df[c]
        n = len(a)
        l = []
        for i in range(n):
            b = re.findall(r"\d+\.?\d*", str(a[i]))
            if len(b) != 0:
                d = b[0]
                l.append(d)
        s = 0
        for i in range(len(l)):
            s += float(l[i])

        if len(l)==0:
            m=10
        else:
            m = s / len(l)

        print m

        # print sum(l)/len(l)
        IsDuplicated = a.drop_duplicates()
        print IsDuplicated

        # print np.mean(l)
        df[c]=df[c].apply(modify_label)
        df[c]=df[c].replace({'正常':m})

        # df[c]=df[c].replace({'心率正常':m})
        # df[c]=df[c].replace({'未见异常':m})
        # df[c]=df[c].replace({'正常 正常':m})
        # df[c]=df[c].replace({'右眼压检测正常':m})
        # df[c]=df[c].replace({'左眼压检测正常':m})
        # df[c]=df[c].replace({'右侧眼压检测正常':m})
        # df[c]=df[c].replace({'左侧眼压检测正常':m})
        # df[c]=df[c].replace({'左眼压正常，其他眼科体检未见异常':m})
        df[c]=df[c].replace({'NaN':np.nan})

        IsDuplicated = a.drop_duplicates()
        print IsDuplicated

        df[c]=df[c].astype('float64')
    return df


test_only_num_1=data_modify(test_only_num)
test_only_num_1.to_csv('../data/test_final.csv',index=False,encoding='utf-8')


train_only_num_1=data_modify(train_only_num)

# clean the nan subject of col 1:5
crx = train_only_num_1.columns
crx1 = crx[1:6]
# for c in ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']:
for c in crx1:
    print min(train_only_num_1[c])

    #  clean the subject whose label is nan 加了这里
    indexs = list(train_only_num_1[np.isnan(train_only_num_1[c])].index)
    print len(indexs)
    train_only_num_1 = train_only_num_1.drop(indexs)

    print min(train_only_num_1[c])


train_only_num_1.to_csv('../data/train_final.csv',index=False,encoding='utf-8')





#  part II

traindata = np.genfromtxt('../data/train_final.csv', delimiter=',')
testdata = np.genfromtxt('../data/test_final.csv', delimiter=',')


testdata=pd.read_csv('../data/test_final.csv',sep=',',encoding='utf-8')
col_vid=testdata.columns[0]
vid_save=testdata[col_vid]
# vid_save=vid_save[1:]

vid_save.to_csv('../data/vid.csv',index=False,encoding='utf-8')
