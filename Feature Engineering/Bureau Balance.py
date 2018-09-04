#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering
Application -train -test

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import copy
import pickle
import itertools
from itertools import chain
import time
import os

sns.set(style="ticks")

# Read Data
bureau = pd.read_csv("Data/bureau.csv").sort_values(by=['SK_ID_BUREAU']).reset_index(drop=True)
print("Bureau Shape & SK_ID_BUREAU number : ", bureau.shape) # Bureau Shape :  (1716428, 17)
bureau_fe = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].drop_duplicates(keep='first').reset_index(drop=True)}).sort_values(by='SK_ID_CURR')
bureau_balance_base = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]

bureau_balance = pd.read_csv("Data/bureau_balance.csv").sort_values(by=['SK_ID_BUREAU', 'MONTHS_BALANCE']).reset_index(drop=True)
print("bureau_balance Shape : ", bureau_balance.shape) # Bureau Shape :  (27299925, 3)
print("bureau_balance SK_ID_BUREAU number : ", len(bureau_balance.SK_ID_BUREAU.unique())) # 817395

t0 = bureau_balance.copy()
num = t0.groupby(by='SK_ID_BUREAU')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'number_month'})

def add_feas(base, feature, on, how='left'):
    base = base.merge(feature, on=on, how=how)
    del feature
    return base

def groupby_BUREAU_CURR(df_x, x, agg_list, fe, by='SK_ID_CURR', bureau_balance_base=bureau_balance_base, fillna=0):
    t0 = add_feas(bureau_balance_base, df_x[['SK_ID_BUREAU', x]], on='SK_ID_BUREAU')
    for agg in agg_list:
        nf = x + '_' + agg
        t1 = t0.groupby(by=by)[x].agg([agg]).reset_index().rename(str, columns={agg: nf})
        fe = add_feas(fe, t1, on='SK_ID_CURR')
        fe = fe.fillna(fillna)
        del t1
    return fe

'''
X:未消费 C:关闭 0:消费dpd0 1:M1 2:M2 3:M3 4:M4 5:M5+
0 拮据层级： 单笔拮据总期数               Ω = 客户层级：Sum/Max/Min/Mean/Std(拮据)
1 拮据层级： 最早那笔拮据期数（与总期数不同）               Ω = 客户层级：Sum/Max/Min/Mean/Std(拮据)
1 拮据层级: 第几期开始消费(出现数字)  + 除以最早那笔期数   客户层级: Max/Min/Sum/Mean/Std(拮据)
2 拮据层级: 第几期开始关闭(出现C)  + 除以最早那笔期数  客户层级: Max/Min/Sum/Mean/Std(拮据)
3 拮据层级: 总消费期数(数字期数个数和) 个数/除以总期数   客户层级: Max/Min/Sum/Mean/Std(拮据)
4 拮据层级: 未消费期数(X个数和) 个数/除以总期数   客户层级: Max/Min/Sum/Mean/Std(拮据)
4 拮据层级： 关闭期数(关闭期数个数和) 个数/除以总期数  客户层级： Max/Min/Sum/Mean/Std(拮据)
5 拮据层级： 最近N(1、3、6、12、24、36、全部)期未出现逾期的次数/占比(最近N期)    客户层级： Max/Min/Sum/Mean(拮据) / Ω
6 拮据层级： 最近N(1、3、6、12、24、36、全部)期出现过逾期>=1（2、3、4、5）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
7 拮据层级： 最近N(1、3、6、12、24、36、全部)期出现过逾期==1（2、3、4、5）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
8 拮据层级： 最近N(1、3、6、12、24、36、全部)期最大逾期期数  客户层级： Max/Min/Sum/Mean(拮据)
9 拮据层级： 最近N(1、3、6、12、24、36、全部)期是否出现过逾期  客户层级： Max(拮据)
10 拮据层级： 最近N(1、3、6、12、24、36、全部)期是否出现过5   客户层级： Max(拮据)
11 拮据层级： 当期（0）的逾期状况   客户层级：Max（拮据）
12 拮据层级： 当期（0）的逾期状况 是否逾期  客户层级：Max（拮据）
'''
# 0
bureau_fe = groupby_BUREAU_CURR(df_x=num, x='number_month', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe)
print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 1
t1 = t0.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').loc[:, ['SK_ID_BUREAU', 'MONTHS_BALANCE']]
t1['MONTHS_BALANCE_first'] = -t1['MONTHS_BALANCE']
del t1['MONTHS_BALANCE']
bureau_fe = groupby_BUREAU_CURR(df_x=t1, x='MONTHS_BALANCE_first', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe)
del t1

# 1
t1 = t0[(t0['STATUS'] != 'X') & (t0['STATUS'] != 'C')]
t2 = t1.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').loc[:, ['SK_ID_BUREAU', 'MONTHS_BALANCE']]
t2['MONTHS_BALANCE_first_use_'] = -t2['MONTHS_BALANCE']
del t2['MONTHS_BALANCE']
t3 = t0.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').loc[:, ['SK_ID_BUREAU', 'MONTHS_BALANCE']]
t3['MONTHS_BALANCE_first'] = -t3['MONTHS_BALANCE']
del t3['MONTHS_BALANCE']
t4 = add_feas(t3, t2, on='SK_ID_BUREAU')
t4['MONTHS_BALANCE_first_use'] = t4['MONTHS_BALANCE_first'] - t4['MONTHS_BALANCE_first_use_']
t4['MONTHS_BALANCE_first_use_ratio'] = np.where(t4['MONTHS_BALANCE_first'] == 0, 0, t4['MONTHS_BALANCE_first_use'] / t4['MONTHS_BALANCE_first'])
t4 = t4.drop(['MONTHS_BALANCE_first_use_', 'MONTHS_BALANCE_first'], axis=1)
bureau_fe = groupby_BUREAU_CURR(df_x=t4, x='MONTHS_BALANCE_first_use', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
bureau_fe = groupby_BUREAU_CURR(df_x=t4, x='MONTHS_BALANCE_first_use_ratio', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
del t1, t2, t3, t4

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 2
t1 = t0[(t0['STATUS'] == 'C')]
t2 = t1.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').loc[:, ['SK_ID_BUREAU', 'MONTHS_BALANCE']]
t2['MONTHS_BALANCE_first_close_'] = -t2['MONTHS_BALANCE']
del t2['MONTHS_BALANCE']
t3 = t0.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').loc[:, ['SK_ID_BUREAU', 'MONTHS_BALANCE']]
t3['MONTHS_BALANCE_first'] = -t3['MONTHS_BALANCE']
del t3['MONTHS_BALANCE']
t4 = add_feas(t3, t2, on='SK_ID_BUREAU')
t4['MONTHS_BALANCE_first_close'] = t4['MONTHS_BALANCE_first'] - t4['MONTHS_BALANCE_first_close_']
t4['MONTHS_BALANCE_first_close_ratio'] = np.where(t4['MONTHS_BALANCE_first'] == 0, 0, t4['MONTHS_BALANCE_first_close'] / t4['MONTHS_BALANCE_first'])
t4 = t4.drop(['MONTHS_BALANCE_first_close_', 'MONTHS_BALANCE_first'], axis=1)
bureau_fe = groupby_BUREAU_CURR(df_x=t4, x='MONTHS_BALANCE_first_close', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
bureau_fe = groupby_BUREAU_CURR(df_x=t4, x='MONTHS_BALANCE_first_close_ratio', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
del t1, t2, t3, t4

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 3 拮据层级: 总消费期数(数字期数个数和)除以总期数   客户层级: Max/Min/Sum/Mean/Std(拮据)
t1 = t0[(t0['STATUS'] != 'X') & (t0['STATUS'] != 'C')]
t2 = t1.groupby(by='SK_ID_BUREAU')['STATUS'].count().reset_index().rename(str, columns={'STATUS': 'use_month_number'})
t3 = add_feas(num, t2, on='SK_ID_BUREAU')
t3 = t3.fillna(0)
t3['use_ratio_month_number'] = t3['use_month_number'] / t3['number_month']
del t3['number_month']
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='use_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='use_ratio_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
del t1, t2, t3

# 3 拮据层级: 未消费期数(X个数和) 个数/除以总期数   客户层级: Max/Min/Sum/Mean/Std(拮据)
t1 = t0[(t0['STATUS'] == 'X')]
t2 = t1.groupby(by='SK_ID_BUREAU')['STATUS'].count().reset_index().rename(str, columns={'STATUS': 'nouse_month_number'})
t3 = add_feas(num, t2, on='SK_ID_BUREAU')
t3 = t3.fillna(0)
t3['nouse_ratio_month_number'] = t3['nouse_month_number'] / t3['number_month']
del t3['number_month']
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='nouse_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='nouse_ratio_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
del t1, t2, t3

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 4 拮据层级： 关闭期数(关闭期数个数和) 个数/除以总期数  客户层级： Max/Min/Sum/Mean/Std(拮据)
t1 = t0[(t0['STATUS'] == 'C')]
t2 = t1.groupby(by='SK_ID_BUREAU')['STATUS'].count().reset_index().rename(str, columns={'STATUS': 'close_month_number'})
t3 = add_feas(num, t2, on='SK_ID_BUREAU')
t3 = t3.fillna(0)
t3['close_ratio_month_number'] = t3['close_month_number'] / t3['number_month']
del t3['number_month']
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='close_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
bureau_fe = groupby_BUREAU_CURR(df_x=t3, x='close_ratio_month_number', agg_list=['max', 'min', 'sum', 'mean', 'std'], fe=bureau_fe, fillna=-999)
del t1, t2, t3

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 5 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期未出现逾期的次数/占比(最近N期)    客户层级： Max/Min/Sum/Mean(拮据) / Ω
# 6 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期出现过逾期>=1（2、3、4、5）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
# 7 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期出现过逾期==1（2、3、4、5）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
# 8 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期最大逾期期数  客户层级： Max/Min/Sum/Mean(拮据)
# 9 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期是否出现过逾期  客户层级： Max(拮据)
# 10 拮据层级： 最近N(全部)期是否出现过5   客户层级： Max(拮据)

def near_N_count(tdata, data, new_featuer, bureau_fe, bureau_balance_base=bureau_balance_base):
    t1 = tdata.copy()
    totol = data.copy()
    new_featuer = new_featuer
    print("New Feature: ", new_featuer)
    for N in [0, -2, -5, -11, -23, -35, -59, -107, -1e7]:
        print("最近 %d 期" % -N)
        t2 = t1[t1['MONTHS_BALANCE'] >= N].groupby(by='SK_ID_BUREAU')['STATUS'].count().reset_index().rename(str, columns={'STATUS': (new_featuer+str(N))})
        tt = totol[totol['MONTHS_BALANCE'] >= N].groupby(by='SK_ID_BUREAU')['STATUS'].count().reset_index().rename(str, columns={'STATUS': ('number_month_in_'+str(N))})
        t3 = add_feas(tt, t2, on='SK_ID_BUREAU')
        t3 = t3.fillna(0)
        t3[new_featuer+str(N)+'_ratio'] = t3[new_featuer+str(N)] / t3['number_month_in_'+str(N)]
        bureau_fe = groupby_BUREAU_CURR(df_x=t3, x=(new_featuer+str(N)), agg_list=['max', 'min', 'sum', 'mean', 'std'],
                                        fe=bureau_fe, fillna=-999)
        bureau_fe = groupby_BUREAU_CURR(df_x=t3, x=(new_featuer+str(N)+'_ratio'), agg_list=['max', 'min', 'mean'],
                                        fe=bureau_fe, fillna=-999)

        # Ω
        t4 = add_feas(bureau_balance_base, t3[['SK_ID_BUREAU', ('number_month_in_'+str(N))]], on='SK_ID_BUREAU')
        for agg in ['max', 'min', 'sum', 'mean']:
            print(agg)
            omiga = bureau_fe[['SK_ID_CURR']]
            t5 = t4.groupby(by='SK_ID_CURR')[('number_month_in_'+str(N))].agg([agg]).reset_index()
            omiga = add_feas(omiga, t5, on='SK_ID_CURR')
            omiga = omiga.fillna(-999)

            bureau_fe = add_feas(bureau_fe, omiga, on='SK_ID_CURR')
            bureau_fe[((new_featuer+str(N)) + '_' + agg + '_omiga')] = np.where(bureau_fe[agg] == -999, -999, bureau_fe[(new_featuer+str(N) + '_' + agg)] / bureau_fe[agg])
            del t5, omiga, bureau_fe[agg]
    return bureau_fe


# 5
t1 = t0[(t0['STATUS'] == 'X') | (t0['STATUS'] == 'C') | (t0['STATUS'] == '0')]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="nodpd_nearest_", bureau_fe=bureau_fe)
del t1

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 6 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期出现过逾期>=1（2、3、4、5）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
t1 = t0[((t0['STATUS'] == "1") | (t0['STATUS'] == "2") | (t0['STATUS'] == "3") | (t0['STATUS'] == "4") | (t0['STATUS'] == "5"))]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd>=1_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[((t0['STATUS'] == "2") | (t0['STATUS'] == "3") | (t0['STATUS'] == "4") | (t0['STATUS'] == "5"))]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd>=2_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[((t0['STATUS'] == "3") | (t0['STATUS'] == "4") | (t0['STATUS'] == "5"))]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd>=3_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[((t0['STATUS'] == "4") | (t0['STATUS'] == "5"))]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd>=4_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[t0['STATUS'] == "5"]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd>=5_nearest_", bureau_fe=bureau_fe)
del t1

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 7 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期出现过逾期==1（2、3、4）的次数/占比  客户层级： Max/Min/Sum/Mean(拮据) / Ω
t1 = t0[t0['STATUS'] == "1"]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd==1_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[t0['STATUS'] == "2"]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd==2_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[t0['STATUS'] == "3"]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd==3_nearest_", bureau_fe=bureau_fe)
del t1
t1 = t0[t0['STATUS'] == "4"]
bureau_fe = near_N_count(tdata=t1, data=t0, new_featuer="dpd==4_nearest_", bureau_fe=bureau_fe)
del t1

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 8 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期最大逾期期数  客户层级： Max/Min/Sum/Mean(拮据)
# 9 拮据层级： 最近N(0、-2、-5、-11、-23、-35、全部)期是否出现过逾期  客户层级： Max(拮据)
# 10 拮据层级： 最近N(全部)期是否出现过5   客户层级： Max(拮据)
t1 = t0[((t0['STATUS'] == "1") | (t0['STATUS'] == "2") | (t0['STATUS'] == "3") | (t0['STATUS'] == "4") | (t0['STATUS'] == "5"))]
for N in [0, -2, -5, -11, -23, -35, -59, -107, -1e7]:
    print("最近 %d 期" % N)
    t2 = t1[t1['MONTHS_BALANCE'] >= N].groupby(by='SK_ID_BUREAU')['STATUS'].max().reset_index().rename(str, columns={'STATUS': ("max_dpd_nearest_"+str(N))})
    t2[("max_dpd_nearest_"+str(N))] = t2[("max_dpd_nearest_"+str(N))].astype('int32')
    t3 = add_feas(bureau_balance, t2, on='SK_ID_BUREAU')
    t3 = t3.drop_duplicates(subset=['SK_ID_BUREAU'], keep='first').reset_index(drop=True).sort_values(by='SK_ID_BUREAU')[['SK_ID_BUREAU', ("max_dpd_nearest_"+str(N))]]
    t3 = t3.fillna(0).reset_index(drop=True)
    t3 = add_feas(bureau_balance_base, t3, on='SK_ID_BUREAU')
    for agg in ['max', 'min', 'sum', 'mean']:
        x = ("max_dpd_nearest_"+str(N))
        nf = x + '_' + agg
        print(nf)
        t4 = t3.groupby(by='SK_ID_CURR')[x].agg([agg]).reset_index().rename(str, columns={agg: nf})
        bureau_fe = add_feas(bureau_fe, t4, on='SK_ID_CURR')
        bureau_fe = bureau_fe.fillna(-999)
        del x, nf, t4
    bureau_fe["flag_dpd_nearest_" + str(N)] = np.where(bureau_fe[(("max_dpd_nearest_"+str(N)) + '_max')] >= 1, 1, 0)
    if N == -1e7:
        bureau_fe["flag_dpd5_nearest_" + str(N)] = np.where(bureau_fe[(("max_dpd_nearest_" + str(N)) + '_max')] >= 5, 1, 0)
    del t2, t3
del t1

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# 11 拮据层级： 当期（0）的逾期状况   客户层级：众数（拮据）
# 12 拮据层级： 当期（0）的逾期状况 是否逾期  客户层级：Max（拮据）
t1 = t0[t0['MONTHS_BALANCE'] == 0].copy()
t1.loc[t1['STATUS'] == 'X', 'STATUS'] = -1
t1.loc[t1['STATUS'] == 'C', 'STATUS'] = -2
t1['STATUS'] = t1['STATUS'].astype('int32')
t2 = add_feas(bureau_balance_base, t1[['SK_ID_BUREAU', 'STATUS']], on='SK_ID_BUREAU')
t2 = t2.fillna(-999)
t3 = t2.groupby(by='SK_ID_CURR')['STATUS'].agg(lambda x: np.mean(pd.Series.mode(x))).reset_index().rename(str, columns={'STATUS': 'now_status_mode'})
t4 = t2.groupby(by='SK_ID_CURR')['STATUS'].max().reset_index().rename(str, columns={'STATUS': 'now_status_max'})
t4['now_status_dpd_flag'] = np.where(t4['now_status_max'] >= 1, 1, 0)

bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

print("bureau_fe : ", bureau_fe.shape[1] - 1)

# to_hdf
bureau_fe = bureau_fe.fillna(0)
print("bureau_fe : ", bureau_fe.shape[1] - 1)
bureau_fe.to_hdf('Data_/Bureau/bureau_balance.hdf', 'bureau_balance', mode='w', format='table')
