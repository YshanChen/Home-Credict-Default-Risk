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
import gc

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns
def add_feas(base, feature, on='SK_ID_CURR', how='left'):
    base = base.merge(feature, on=on, how=how)
    del feature
    return base
def stat_simple(data, x, groupby, nf_base, agg_list=['max', 'min', 'mean', 'sum', 'std']):
    t1 = data[['SK_ID_PREV', 'SK_ID_CURR', x]].copy()
    t2 = t1.groupby(by=groupby)[x].agg(agg_list).reset_index().rename(str, columns={'max': nf_base+'_max', 'min': nf_base+'_min', 'mean': nf_base+'_mean', 'sum': nf_base+'_sum', 'std': nf_base+'_std'})
    return t2
def NAME_CONTRACT_STATUS_flag(data, x):
    t1 = data[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']].copy()
    nf = x + '_flag'
    t1[nf] = np.where(t1['NAME_CONTRACT_STATUS'] == x, 1, 0)

    t2 = t1.groupby(by='SK_ID_CURR')[nf].max().reset_index()
    return t2

# Read Data
ccd = pd.read_csv("Data/credit_card_balance.csv").sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
print("NAME_CONTRACT_STATUS Types: ", ccd['NAME_CONTRACT_STATUS'].value_counts())
ccd_base = ccd[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(subset=['SK_ID_PREV'], keep='first')
ccd_fe = ccd_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')
'''
Signed             11058
Demand              1365 
Sent proposal        513 -> Active
Refused               17 -> Active
Approved               5 -> Active
'''

# NAME_CONTRACT_STATUS Flag
t1 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Signed')
t2 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Demand')
t3 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Sent proposal')
t4 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Refused')
t5 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Approved')
t6 = NAME_CONTRACT_STATUS_flag(data=ccd, x='Completed')

ccd_fe = add_feas(ccd_fe, t1, on="SK_ID_CURR")
ccd_fe = add_feas(ccd_fe, t2, on="SK_ID_CURR")
ccd_fe = add_feas(ccd_fe, t3, on="SK_ID_CURR")
ccd_fe = add_feas(ccd_fe, t4, on="SK_ID_CURR")
ccd_fe = add_feas(ccd_fe, t5, on="SK_ID_CURR")
ccd_fe = add_feas(ccd_fe, t6, on="SK_ID_CURR")
ccd.loc[(ccd['NAME_CONTRACT_STATUS'] == 'Signed') | (ccd['NAME_CONTRACT_STATUS'] == 'Sent proposal') | (ccd['NAME_CONTRACT_STATUS'] == 'Refused') | (ccd['NAME_CONTRACT_STATUS'] == 'Approved'), 'NAME_CONTRACT_STATUS'] = 'Active'

t0 = ccd.copy().fillna(0)
# 标记 Completed1  Demand2  Amortized3  (Demand | Amortized)
t0['ccd_status_t'] = np.where(t0['NAME_CONTRACT_STATUS'] == 'Active', 0,
                              np.where(t0['NAME_CONTRACT_STATUS'] == 'Completed', 1,
                                       np.where(t0['NAME_CONTRACT_STATUS'] == 'Demand', 2, 3)))
t1 = t0.groupby(by='SK_ID_PREV')['ccd_status_t'].max().reset_index().rename(str, columns={'ccd_status_t': 'ccd_status'})
t0 = add_feas(t0, t1, on='SK_ID_PREV')
del t0['ccd_status_t']
del t1, t2, t3, t4, t5, t6

def first_mon_f(data, x, active=False):
    if active:
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_PREV', 'MONTHS_BALANCE', x]].copy()
        nf = x + '_active_firstMon'
    else:
        t1 = data[['SK_ID_PREV', 'MONTHS_BALANCE', x]].copy()
        nf = x + '_firstMon'

    t2 = t1[t1[x] != 0]
    t3 = t2.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV']).rename(str, columns={'MONTHS_BALANCE': nf})
    t3[nf] = -t3[nf]

    feature = add_feas(ccd_base, t3[['SK_ID_PREV', nf]], on='SK_ID_PREV')
    del feature['SK_ID_CURR']
    feature = feature.fillna(0)
    return feature
def count_f(data, x, active=False, base=ccd_base, stat_to_CURR=False, agg_list=['max', 'min', 'mean', 'sum', 'std']):
    if active:
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', x]].copy()
        nf = x + '_count_active'
    else:
        t1 = data[['SK_ID_PREV', 'SK_ID_CURR', x]].copy()
        nf = x + '_count'
    t2 = t1.groupby(by='SK_ID_PREV')[x].count().reset_index().rename(str, columns={x: nf})
    t3 = add_feas(base, t2, on='SK_ID_PREV')
    t3 = t3[['SK_ID_CURR', 'SK_ID_PREV', nf]].copy()
    feature = t3

    if stat_to_CURR:
        t4 = t3.groupby(by='SK_ID_CURR')[nf].agg(agg_list).reset_index().rename(str, columns={'max':nf+'_max', 'min':nf+'_min', 'mean':nf+'_mean', 'sum':nf+'_sum', 'std':nf+'_std'})
        feature = t4

    return feature
# 最早一笔/最近一笔 最早一笔-最近一笔 最早一笔-最近一笔/最早一笔 最近那笔/最早那笔
def first_last_f(data, x, sortValues='MONTHS_BALANCE', active=False, extra=False, base=ccd_base, stat_to_CURR=False, agg_list=['max', 'min', 'mean', 'sum', 'std']):
    if x == 'MONTHS_BALANCE':
        if active:
            t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_PREV', x]].copy()
            nf = x + '_active'
        else:
            t1 = data[['SK_ID_PREV', x]].copy()
            nf = x
    else:
        if active:
            t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_PREV', sortValues, x]].copy()
            nf = x + '_active'
        else:
            t1 = data[['SK_ID_PREV', sortValues, x]].copy()
            nf = x

    nf_first = nf +'_ft'
    nf_last = nf +'_lt'

    t2_1 = t1.sort_values(by=['SK_ID_PREV', sortValues]).drop_duplicates(subset=['SK_ID_PREV'], keep='first').rename(str, columns={x: nf_first})
    t2_2 = t1.sort_values(by=['SK_ID_PREV', sortValues]).drop_duplicates(subset=['SK_ID_PREV'], keep='last').rename(str, columns={x: nf_last})
    t3 = add_feas(base, t2_1[['SK_ID_PREV', nf_first]], on='SK_ID_PREV')
    t3 = add_feas(t3, t2_2[['SK_ID_PREV', nf_last]], on='SK_ID_PREV')

    if extra:
        t3 = t3[['SK_ID_CURR', 'SK_ID_PREV', nf_first, nf_last]]
        t3[nf_first + '-' + nf_last] = t3[nf_first] - t3[nf_last]
        t3['rto-' + nf_first + '-' + nf_last] = t3[nf_first + '-' + nf_last] / t3[nf_first]
        t3[nf_last + '/' + nf_first] = t3[nf_last] / t3[nf_first]
        t3[nf_first + '-' + nf_last + '_b0_flag'] = np.where(t3[nf_first + '-' + nf_last] > 0, 1, 0)
        t3[nf_first + '-' + nf_last + '_l0_flag'] = np.where(t3[nf_first + '-' + nf_last] < 0, 1, 0)
    feature = t3

    if stat_to_CURR & extra:
        t4 = t3.groupby(by='SK_ID_CURR')[nf_first].agg(agg_list).reset_index().rename(str, columns={'max':nf_first+'_max', 'min':nf_first+'_min', 'mean':nf_first+'_mean', 'sum':nf_first+'_sum', 'std':nf_first+'_std'})
        t5 = t3.groupby(by='SK_ID_CURR')[nf_last].agg(agg_list).reset_index().rename(str, columns={'max': nf_last + '_max', 'min': nf_last + '_min', 'mean': nf_last + '_mean', 'sum': nf_last + '_sum', 'std': nf_last + '_std'})
        t6 = t3.groupby(by='SK_ID_CURR')[nf_first + '-' + nf_last].agg(agg_list).reset_index().rename(str, columns={'max': nf_first + '-' + nf_last + '_max', 'min': nf_first + '-' + nf_last + '_min', 'mean': nf_first + '-' + nf_last + '_mean', 'sum': nf_first + '-' + nf_last + '_sum', 'std': nf_first + '-' + nf_last + '_std'})
        t7 = t3.groupby(by='SK_ID_CURR')['rto-' + nf_first + '-' + nf_last].agg(agg_list).reset_index().rename(str, columns={'max': 'rto-' + nf_first + '-' + nf_last + '_max', 'min': 'rto-' + nf_first + '-' + nf_last + '_min', 'mean': 'rto-' + nf_first + '-' + nf_last + '_mean', 'sum': 'rto-' + nf_first + '-' + nf_last + '_sum', 'std': 'rto-' + nf_first + '-' + nf_last + '_std'})
        t8 = t3.groupby(by='SK_ID_CURR')[nf_last + '/' + nf_first].agg(agg_list).reset_index().rename(str, columns={'max':nf_last + '/' + nf_first+'_max', 'min':nf_last + '/' + nf_first+'_min', 'mean':nf_last + '/' + nf_first+'_mean', 'sum':nf_last + '/' + nf_first+'_sum', 'std':nf_last + '/' + nf_first+'_std'})
        t9 = t3.groupby(by='SK_ID_CURR')[nf_first + '-' + nf_last + '_b0_flag'].agg('max').reset_index().rename(str, columns={'max': nf_first + '-' + nf_last + '_b0_flag'})
        t10 = t3.groupby(by='SK_ID_CURR')[nf_first + '-' + nf_last + '_l0_flag'].agg('max').reset_index().rename(str, columns={'max': nf_first + '-' + nf_last + '_l0_flag'})

        t11 = add_feas(base, t4, on='SK_ID_CURR')
        t11 = add_feas(t11, t5, on='SK_ID_CURR')
        t11 = add_feas(t11, t6, on='SK_ID_CURR')
        t11 = add_feas(t11, t7, on='SK_ID_CURR')
        t11 = add_feas(t11, t8, on='SK_ID_CURR')
        t11 = add_feas(t11, t9, on='SK_ID_CURR')
        t11 = add_feas(t11, t10, on='SK_ID_CURR')

        del t11['SK_ID_PREV']
        feature = t11

    return feature
def stat_f(data, x, stat_to_CURR=True, active=False, agg_list=['max', 'min', 'mean', 'sum', 'std'], base=ccd_base):
    if active:
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_PREV', x]].copy()
        nf = x + '_active'
    else:
        t1 = data[['SK_ID_PREV', x]].copy()
        nf = x + ''
    t2 = t1.groupby('SK_ID_PREV')[x].agg(agg_list).reset_index().rename(str, columns={'max': nf+'_max', 'min': nf+'_min', 'mean': nf+'_mean', 'sum': nf+'_sum', 'std': nf+'_std'})
    t3 = add_feas(base, t2, on='SK_ID_PREV')
    feature = t3

    if stat_to_CURR:
        for fe in [feas for feas in t3.columns if feas not in ['SK_ID_CURR', 'SK_ID_PREV']]:
            t4 = t3.groupby('SK_ID_CURR')[fe].agg(agg_list).reset_index().rename(str, columns={'max': fe+'_max', 'min': fe+'_min', 'mean': fe+'_mean', 'sum': fe+'_sum', 'std': fe+'_std'})
            base = add_feas(base, t4, on='SK_ID_CURR')
            del t4
        del base['SK_ID_PREV']
        feature = base
    return feature
def div_f(data, x, y, add_1=False):
    t1 = data.copy()
    nf = x + '/' + y
    if add_1:
        t1[nf] = t1[x] / (t1[y]+1)
    else:
        t1[nf] = t1[x] / t1[y]
    return t1

# 变量衍生 ------------------------------------------------------
# 1. 笔数 各类型笔数 占比 => SK_ID_CURR  (0.007145754070182892 笔数 > 1)
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt'})
ccd_fe = add_feas(ccd_fe, t2, on="SK_ID_CURR")
del t1, t2

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt_active'})
ccd_fe = add_feas(ccd_fe, t2, on="SK_ID_CURR")
del t1, t2

t1 = t0.loc[t0['ccd_status'] == 1, ['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt_completed'})
ccd_fe = add_feas(ccd_fe, t2, on="SK_ID_CURR")
del t1, t2

t1 = t0.loc[t0['ccd_status'] == 2, ['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt_demand'})
ccd_fe = add_feas(ccd_fe, t2, on="SK_ID_CURR")
del t1, t2

ccd_fe = ccd_fe.fillna(0)
ccd_fe['SK_ID_PREV_rto_active'] = ccd_fe['SK_ID_PREV_cnt_active'] / ccd_fe['SK_ID_PREV_cnt']
ccd_fe['SK_ID_PREV_rto_completed'] = ccd_fe['SK_ID_PREV_cnt_completed'] / ccd_fe['SK_ID_PREV_cnt']
ccd_fe['SK_ID_PREV_rto_demand'] = ccd_fe['SK_ID_PREV_cnt_demand'] / ccd_fe['SK_ID_PREV_cnt']


# ---------------------------------- 存在两笔及以上的删除 ------------------------------------------
# 保留存在逾期记录 活跃 余额最大
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt'})
t2 = t2[t2['SK_ID_PREV_cnt'] > 1]
t3 = add_feas(t2, t0)
t4 = t3.groupby('SK_ID_PREV')['SK_DPD'].sum().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_sum'})
t3 = add_feas(t3, t4, on='SK_ID_PREV')
t5 = t3.groupby('SK_ID_PREV')['AMT_BALANCE'].max().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_max'})
t6 = add_feas(t3, t5, on='SK_ID_PREV')
t6 = t6.sort_values(by=['SK_ID_CURR', 'SK_DPD_sum', 'ccd_status', 'AMT_BALANCE_max'], ascending=[True, False, True, False])[['SK_ID_CURR', 'SK_ID_PREV','SK_DPD_sum', 'ccd_status', 'AMT_BALANCE_max']]
remain_prev_list = t6[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')
remain_prev_list['remain_flag'] = 1
t7 = add_feas(t6, remain_prev_list, on='SK_ID_PREV')
t7 = t7.fillna(0)
del_prev_list = t7.loc[t7['remain_flag'] == 0, 'SK_ID_PREV'].drop_duplicates(keep='first')
del t1, t2, t3, t4, t5, t6, t7

# 删除多余PREV
del_prev_list = del_prev_list.tolist()
t0 = t0[~t0['SK_ID_PREV'].isin(del_prev_list)]  # 3840312-3815880 = 24432

# check
# t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
# t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt'})
# (t2['SK_ID_PREV_cnt']>1).any() # False

# 不需要SK_ID_PREV 直接SK_ID_CURR
t0 = t0.drop(['SK_ID_PREV'], axis=1)



# 2. SK_ID_PREV: months_balance count 最早 最近 stat => SK_ID_CURR: stat (active)
t1 = count_f(data=t0, x='MONTHS_BALANCE', stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1, on="SK_ID_CURR")
ccd_fe = ccd_fe.fillna(0)

t1 = first_last_f(data=t0, x='MONTHS_BALANCE', stat_to_CURR=True)
t2 = first_last_f(data=t0, x='MONTHS_BALANCE', stat_to_CURR=True, active=True)
ccd_fe = add_feas(ccd_fe, t1, on='SK_ID_CURR')
ccd_fe = add_feas(ccd_fe, t2, on='SK_ID_CURR')
ccd_fe = ccd_fe.fillna(-999)

del t1, t2

# 3. SK_ID_PREV: Months_balance 最早期数-最晚时间/最早期数（as 5/57）/count stat  =》 stat  (未结清)
t1 = first_last_f(data=t0, x='MONTHS_BALANCE')
t1['mon_diff_rto'] = (t1['MONTHS_BALANCE_lt'] - t1['MONTHS_BALANCE_ft']) / (-t1['MONTHS_BALANCE_ft'])
t2 = count_f(data=t0, x='MONTHS_BALANCE')
t3 = add_feas(t1, t2[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t3['mon_diff_rto_cnt'] = (t3['MONTHS_BALANCE_lt'] - t3['MONTHS_BALANCE_ft']) / (t3['MONTHS_BALANCE_count'])
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'mon_diff_rto', 'mon_diff_rto_cnt']], on='SK_ID_CURR')
del t1, t2, t3

t1 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t1['mon_diff_rto_active'] = (t1['MONTHS_BALANCE_active_lt'] - t1['MONTHS_BALANCE_active_ft']) / (-t1['MONTHS_BALANCE_active_ft'])
t2 = count_f(data=t0, x='MONTHS_BALANCE')
t3 = add_feas(t1, t2[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t3['mon_diff_rto_cnt_active'] = (t3['MONTHS_BALANCE_active_lt'] - t3['MONTHS_BALANCE_active_ft']) / (t3['MONTHS_BALANCE_count'])
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'mon_diff_rto_active', 'mon_diff_rto_cnt_active']], on='SK_ID_CURR')
del t1, t2, t3

# 4. AMT_CREDIT_LIMIT_ACTUAL stat 最早那笔 最近那笔 最早那笔-最近那笔  最早那笔-最近那笔/最早那笔 最近那笔/最早那笔 stat  最早那笔-最近那笔>0flag 最早那笔-最近那笔<0flag => SK_ID_CURR: stat flag(active)
t1 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=True, stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', active=True, extra=True, stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = stat_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', stat_to_CURR=True, active=False, agg_list=['max', 'min', 'mean', 'sum', 'std'], base=ccd_base)
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = stat_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', stat_to_CURR=True, active=True, agg_list=['max', 'min', 'mean', 'sum', 'std'], base=ccd_base)
ccd_fe = add_feas(ccd_fe, t1)
del t1


# 5. AMT_BALANCE = 0 的期数/总期数 /最早那笔期数 stat
t1 = t0.loc[t0['AMT_BALANCE'] == 0, ['SK_ID_PREV', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_PREV')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_eq0_cnt'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE')
t4 = count_f(data=t0, x='MONTHS_BALANCE')
t5 = add_feas(ccd_base, t2[['SK_ID_PREV', 'AMT_BALANCE_eq0_cnt']], on='SK_ID_PREV')
t5 = add_feas(t5, t3[['SK_ID_PREV', 'MONTHS_BALANCE_ft']], on='SK_ID_PREV')
t5 = add_feas(t5, t4[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t5['AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_ft'] = t5['AMT_BALANCE_eq0_cnt'] / -t5['MONTHS_BALANCE_ft']
t5['AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_count'] = t5['AMT_BALANCE_eq0_cnt'] / t5['MONTHS_BALANCE_count']
del t5['SK_ID_PREV']
for feas in ['AMT_BALANCE_eq0_cnt', 'AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_ft', 'AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_count']:
    t6 = t5.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5

t1 = t0.loc[(t0['AMT_BALANCE'] == 0) & (t0['ccd_status'] == 0), ['SK_ID_PREV', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_PREV')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_eq0_cnt_active'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t4 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t5 = add_feas(ccd_base, t2[['SK_ID_PREV', 'AMT_BALANCE_eq0_cnt_active']], on='SK_ID_PREV')
t5 = add_feas(t5, t3[['SK_ID_PREV', 'MONTHS_BALANCE_active_ft']], on='SK_ID_PREV')
t5 = add_feas(t5, t4[['SK_ID_PREV', 'MONTHS_BALANCE_count_active']], on='SK_ID_PREV')
t5['AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_active_ft'] = t5['AMT_BALANCE_eq0_cnt_active'] / -t5['MONTHS_BALANCE_active_ft']
t5['AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_count_active'] = t5['AMT_BALANCE_eq0_cnt_active'] / t5['MONTHS_BALANCE_count_active']
del t5['SK_ID_PREV']
for feas in ['AMT_BALANCE_eq0_cnt_active', 'AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_active_ft', 'AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_count_active']:
    t6 = t5.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5

# 5. AMT_BALANCE 是否支用 => SK_ID_CURR: flag(active)
t1 = t0[['SK_ID_PREV', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_PREV')['AMT_BALANCE'].max().reset_index().rename(str, columns={'AMT_BALANCE':'encash_flag_t'})
t2['encash_flag'] = np.where(t2['encash_flag_t'] > 0, 1, 0)
t3 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = t3.groupby('SK_ID_CURR')['encash_flag'].max().reset_index()
ccd_fe = add_feas(ccd_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_PREV', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_PREV')['AMT_BALANCE'].max().reset_index().rename(str, columns={'AMT_BALANCE':'encash_flag_t'})
t2['active_encash_flag'] = np.where(t2['encash_flag_t'] > 0, 1, 0)
t3 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = t3.groupby('SK_ID_CURR')['active_encash_flag'].max().reset_index()
ccd_fe = add_feas(ccd_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# 6. AMT_BALANCE 最早那笔余额(+非0) 最早支用那笔余额/最早那笔授信额度/最近那笔授信额度    最大那笔余额 最大那笔余额/最早那笔授信额度/最近那笔授信额度  stat   => SK_ID_CURR: stat(active)
t1 = first_last_f(data=t0, x='AMT_BALANCE', extra=False, stat_to_CURR=False)
t1 = t1[['SK_ID_PREV', 'AMT_BALANCE_ft']]
t2 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE']].copy()
t3 = first_last_f(data=t2, x='AMT_BALANCE', extra=False, stat_to_CURR=False)
t3 = t3[['SK_ID_PREV', 'AMT_BALANCE_ft']].rename(str, columns={'AMT_BALANCE_ft': 'AMT_BALANCE_ft_encash'})
t4 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, stat_to_CURR=False)
t4 = t4[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')

t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft'] = t5['AMT_BALANCE_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_ft']+1)
t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt'] = t5['AMT_BALANCE_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_lt']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft'] = t5['AMT_BALANCE_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_ft']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt'] = t5['AMT_BALANCE_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_lt']+1)
t5 = t5.fillna(0)

for feas in ['AMT_BALANCE_ft', 'AMT_BALANCE_ft_encash', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt']:
    t6 = t5.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6

del t1, t2, t3, t4, t5

# 7. AMT_BALANCE 第一次支用期数 第一次支用期数/总期数 第一次支用期数/最早那期 stat   => SK_ID_CURR: stat(active)

t1 =first_mon_f(data=t0, x='AMT_BALANCE')
t2 = first_last_f(data=t0, x='MONTHS_BALANCE')
t3 = count_f(data=t0, x='MONTHS_BALANCE')
t4 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t4 = add_feas(t4, t2[['SK_ID_PREV', 'MONTHS_BALANCE_ft']], on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t4['AMT_BALANCE_firstMon/MONTHS_BALANCE_ft'] = t4['AMT_BALANCE_firstMon'] / -t4['MONTHS_BALANCE_ft']
t4['AMT_BALANCE_firstMon/MONTHS_BALANCE_count'] = t4['AMT_BALANCE_firstMon'] / t4['MONTHS_BALANCE_count']

for feas in ['AMT_BALANCE_firstMon', 'AMT_BALANCE_firstMon/MONTHS_BALANCE_ft', 'AMT_BALANCE_firstMon/MONTHS_BALANCE_count']:
    t5 = t4.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4

t1 =first_mon_f(data=t0, x='AMT_BALANCE', active=True)
t2 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t3 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t4 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t4 = add_feas(t4, t2[['SK_ID_PREV', 'MONTHS_BALANCE_active_ft']], on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'MONTHS_BALANCE_count_active']], on='SK_ID_PREV')
t4['AMT_BALANCE_active_firstMon/MONTHS_BALANCE_active_ft'] = t4['AMT_BALANCE_active_firstMon'] / -t4['MONTHS_BALANCE_active_ft']
t4['AMT_BALANCE_active_firstMon/MONTHS_BALANCE_count_active'] = t4['AMT_BALANCE_active_firstMon'] / t4['MONTHS_BALANCE_count_active']

for feas in ['AMT_BALANCE_active_firstMon', 'AMT_BALANCE_active_firstMon/MONTHS_BALANCE_active_ft', 'AMT_BALANCE_active_firstMon/MONTHS_BALANCE_count_active']:
    t5 = t4.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4


# 8. AMT_BALANCE stat stat/最早那笔授信额度/最近那笔授信额度   排除掉0的stat/最早那笔授信额度/最近那笔授信额度   => SK_ID_CURR: stat(active)
t1 = stat_f(data=t0, x='AMT_BALANCE', stat_to_CURR=False)
del t1['SK_ID_CURR']
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, stat_to_CURR=False)
t2 = t2[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t3 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t3 = add_feas(t3, t2, on='SK_ID_PREV')
for feas in ['AMT_BALANCE_max', 'AMT_BALANCE_min','AMT_BALANCE_mean', 'AMT_BALANCE_sum', 'AMT_BALANCE_std']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]:
    print(feas)
    t4 = t3.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t4)
    del t4
del t1, t2, t3

t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE']].copy().rename(columns={'AMT_BALANCE':'AMT_BALANCE_encash'})
t1 = stat_f(data=t1, x='AMT_BALANCE_encash', stat_to_CURR=False)
del t1['SK_ID_CURR']
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, stat_to_CURR=False)
t2 = t2[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t3 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t3 = add_feas(t3, t2, on='SK_ID_PREV')
for feas in ['AMT_BALANCE_encash_max', 'AMT_BALANCE_encash_min','AMT_BALANCE_encash_mean', 'AMT_BALANCE_encash_sum', 'AMT_BALANCE_encash_std']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]:
    print(feas)
    t4 = t3.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t4)
    del t4
del t1, t2, t3

t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE', 'ccd_status']].copy().rename(columns={'AMT_BALANCE':'AMT_BALANCE_encash'})
t1 = stat_f(data=t1, x='AMT_BALANCE_encash', stat_to_CURR=False, active=True)
del t1['SK_ID_CURR']
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, stat_to_CURR=False, active=True)
t2 = t2[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']]
t3 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t3 = add_feas(t3, t2, on='SK_ID_PREV')
for feas in ['AMT_BALANCE_encash_active_max', 'AMT_BALANCE_encash_active_min','AMT_BALANCE_encash_active_mean', 'AMT_BALANCE_encash_active_sum']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_active_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_active_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_active_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_active_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']]:
    print(feas)
    t4 = t3.groupby('SK_ID_CURR')[feas].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': feas+'_max', 'min': feas+'_min', 'mean': feas+'_mean', 'sum': feas+'_sum', 'std': feas+'_std'})
    ccd_fe = add_feas(ccd_fe, t4)
    del t4
del t1, t2, t3

# 9. AMT_BALANCE diff(-1,-2) stat stat/最早那笔授信额度/最近那笔授信额度 排除掉0的diff(-1,-2) stat/最早那笔授信额度/最近那笔授信额度   => SK_ID_CURR: stat(active)
# 10. AMT_BALANCE 余额/授信额度 diff(-1,-2) stat 排除掉0的 => SK_ID_CURR: stat(active)
# 11. AMT_BALANCE 余额/授信额度 是否连续2\3期下降（diff 均小等于0）排除掉0的 => SK_ID_CURR: flag(active)

gc.collect()

# 12. AMT_DRAWINGS 4个AMT_DRAWINGS stat 3个sum占总AMT_DRAWINGS sum比重 3个mean占总AMT_DRAWINGS mean比重 stat => SK_ID_CURR: stat(active)
t1 = stat_f(data=t0, x='AMT_DRAWINGS_ATM_CURRENT', stat_to_CURR=True)
t2 = stat_f(data=t0, x='AMT_DRAWINGS_POS_CURRENT', stat_to_CURR=True)
t3 = stat_f(data=t0, x='AMT_DRAWINGS_OTHER_CURRENT', stat_to_CURR=True)
t4 = stat_f(data=t0, x='AMT_DRAWINGS_CURRENT', stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = stat_f(data=t0[t0['AMT_DRAWINGS_ATM_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_ATM_CURRENT', stat_to_CURR=True)
t2 = stat_f(data=t0[t0['AMT_DRAWINGS_POS_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_POS_CURRENT', stat_to_CURR=True)
t3 = stat_f(data=t0[t0['AMT_DRAWINGS_OTHER_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_OTHER_CURRENT', stat_to_CURR=True)
t4 = stat_f(data=t0[t0['AMT_DRAWINGS_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_CURRENT', stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = stat_f(data=t0, x='AMT_DRAWINGS_ATM_CURRENT', stat_to_CURR=True, active=True)
t2 = stat_f(data=t0, x='AMT_DRAWINGS_POS_CURRENT', stat_to_CURR=True, active=True)
t3 = stat_f(data=t0, x='AMT_DRAWINGS_OTHER_CURRENT', stat_to_CURR=True, active=True)
t4 = stat_f(data=t0, x='AMT_DRAWINGS_CURRENT', stat_to_CURR=True, active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()


t1 = stat_f(data=t0[t0['AMT_DRAWINGS_ATM_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_ATM_CURRENT', stat_to_CURR=True, active=True)
t2 = stat_f(data=t0[t0['AMT_DRAWINGS_POS_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_POS_CURRENT', stat_to_CURR=True, active=True)
t3 = stat_f(data=t0[t0['AMT_DRAWINGS_OTHER_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_OTHER_CURRENT', stat_to_CURR=True, active=True)
t4 = stat_f(data=t0[t0['AMT_DRAWINGS_CURRENT'] != 0].copy(), x='AMT_DRAWINGS_CURRENT', stat_to_CURR=True, active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t5['AMT_DRAWINGS_ATM_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum'] = t5['AMT_DRAWINGS_ATM_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_POS_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum'] = t5['AMT_DRAWINGS_POS_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum'] = t5['AMT_DRAWINGS_OTHER_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
for x in ['AMT_DRAWINGS_ATM_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum', 'AMT_DRAWINGS_POS_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum']:
    t6 = t5.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t5['AMT_DRAWINGS_ATM_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act'] = t5['AMT_DRAWINGS_ATM_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_POS_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act'] = t5['AMT_DRAWINGS_POS_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act'] = t5['AMT_DRAWINGS_OTHER_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
for x in ['AMT_DRAWINGS_ATM_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act', 'AMT_DRAWINGS_POS_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act', 'AMT_DRAWINGS_OTHER_CURRENT_sum/AMT_DRAWINGS_CURRENT_sum_act']:
    t6 = t5.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t5['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_ATM_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_POS_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_OTHER_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
for x in ['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean', 'AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean', 'AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean']:
    t6 = t5.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t5['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_ATM_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_POS_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_OTHER_CURRENT_sum'] / (t5['AMT_DRAWINGS_CURRENT_sum']+1)
for x in ['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act', 'AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act', 'AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act']:
    t6 = t5.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t6)
    del t6
del t1, t2, t3, t4, t5
gc.collect()

# 13. AMT_DRAWINGS 不为0次数/总期数/最早期数/余额非0期数
t1 = t0[t0['AMT_DRAWINGS_CURRENT'] != 0]
t2 = t1.groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_no0'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE')
t4 = count_f(data=t0, x='MONTHS_BALANCE')
t5 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_PREV', 'AMT_BALANCE']].copy()
t6 = t5.groupby('SK_ID_PREV')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt'})
t7 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t7 = add_feas(t7, t3[['SK_ID_PREV', 'MONTHS_BALANCE_ft']], on='SK_ID_PREV')
t7 = add_feas(t7, t4[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t7 = add_feas(t7, t6[['SK_ID_PREV', 'AMT_BALANCE_neq0_cnt']], on='SK_ID_PREV')
t7 = t7.fillna(0)
t7['AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_ft'] = t7['AMT_DRAWINGS_CURRENT_no0'] / -t7['MONTHS_BALANCE_ft']
t7['AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_count'] = t7['AMT_DRAWINGS_CURRENT_no0'] / t7['MONTHS_BALANCE_count']
t7['AMT_DRAWINGS_CURRENT_no0/AMT_BALANCE_neq0_cnt'] = t7['AMT_DRAWINGS_CURRENT_no0'] / t7['AMT_BALANCE_neq0_cnt']
for x in ['AMT_DRAWINGS_CURRENT_no0', 'AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_ft', 'AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_count', 'AMT_DRAWINGS_CURRENT_no0/AMT_BALANCE_neq0_cnt']:
    t8 = t7.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    t8 = t8.fillna(0)
    ccd_fe = add_feas(ccd_fe, t8)
    del t8
del t1, t2, t3, t4, t5, t6, t7

# 13. AMT_DRAWINGS 平均每月支用金额（4class_AMT_DRAWINGS/总期数/最早期数/不为0次数/余额非0期数） stat => SK_ID_CURR: stat(active)
t1 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0.groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t6 = first_last_f(data=t0, x='MONTHS_BALANCE')
t6['MONTHS_BALANCE_ft'] = -t6['MONTHS_BALANCE_ft']
t7 = count_f(data=t0, x='MONTHS_BALANCE')
t8 = t0[t0['AMT_DRAWINGS_CURRENT'] != 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_neq0_cnt'})
t9 = t0[t0['AMT_BALANCE'] != 0].groupby('SK_ID_PREV')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt'})
t10 = add_feas(ccd_base, t5, on='SK_ID_PREV')
t10 = add_feas(t5, t6[['SK_ID_PREV', 'MONTHS_BALANCE_ft']], on='SK_ID_PREV')
t10 = add_feas(t10, t7[['SK_ID_PREV', 'MONTHS_BALANCE_count']], on='SK_ID_PREV')
t10 = add_feas(t10, t8, on='SK_ID_PREV')
t10 = add_feas(t10, t9, on='SK_ID_PREV')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = t10.fillna(0)
for x in [x for x in t10.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_DRAWINGS_ATM_CURRENT_sum','AMT_DRAWINGS_POS_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_sum','AMT_DRAWINGS_CURRENT_sum', 'MONTHS_BALANCE_ft', 'MONTHS_BALANCE_count','AMT_DRAWINGS_CURRENT_neq0_cnt', 'AMT_BALANCE_neq0_cnt']]:
    print(x)
    t11 = t10.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t11)
    del t11
del t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t5 = add_feas(t5, t2, on='SK_ID_PREV')
t5 = add_feas(t5, t3, on='SK_ID_PREV')
t5 = add_feas(t5, t4, on='SK_ID_PREV')
t6 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t6['MONTHS_BALANCE_active_ft'] = -t6['MONTHS_BALANCE_active_ft']
t7 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t8 = t0[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0)].groupby('SK_ID_PREV')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_neq0_cnt_act'})
t9 = t0[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0)].groupby('SK_ID_PREV')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt_act'})
t10 = add_feas(ccd_base, t5, on='SK_ID_PREV')
t10 = add_feas(t5, t6[['SK_ID_PREV', 'MONTHS_BALANCE_active_ft']], on='SK_ID_PREV')
t10 = add_feas(t10, t7[['SK_ID_PREV', 'MONTHS_BALANCE_count_active']], on='SK_ID_PREV')
t10 = add_feas(t10, t8, on='SK_ID_PREV')
t10 = add_feas(t10, t9, on='SK_ID_PREV')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_active_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_count_active')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_BALANCE_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_active_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_count_active')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_active_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_count_active')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_BALANCE_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_active_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_count_active')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt_act')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt_act')
t10 = t10.fillna(0)
for x in [x for x in t10.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_DRAWINGS_ATM_CURRENT_sum','AMT_DRAWINGS_POS_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_sum','AMT_DRAWINGS_CURRENT_sum', 'MONTHS_BALANCE_active_ft', 'MONTHS_BALANCE_count_active','AMT_DRAWINGS_CURRENT_neq0_cnt_act', 'AMT_BALANCE_neq0_cnt_act']]:
    print(x)
    t11 = t10.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t11)
    del t11
del t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

# 14. AMT_DRAWINGS AMT_DRAWINGS/当月授信额度/当月余额（排除余额为0月份） stat => SK_ID_CURR: stat(active)
t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_PREV', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE']].copy()
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_BALANCE', add_1=True)
t2 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t2 = t2.fillna(0)
for x in ['AMT_DRAWINGS_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT/AMT_BALANCE']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean'})
    ccd_fe = add_feas(ccd_fe, t3)
    del t3
del t1, t2

t1 = t0.loc[(t0['AMT_BALANCE'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_PREV', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE']].copy()
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_BALANCE', add_1=True)
t2 = add_feas(ccd_base, t1, on='SK_ID_PREV')
t2 = t2.fillna(0)
for x in ['AMT_DRAWINGS_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT/AMT_BALANCE']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean']).reset_index().rename(str, columns={'max': x+'_act_max', 'min': x+'_act_min', 'mean': x+'_act_mean'})
    ccd_fe = add_feas(ccd_fe, t3)
    del t3
del t1, t2

# 15. AMT_DRAWINGS 主要取款方式(金额最多) => SK_ID_CURR （oh encoding） 客户级
t1 = t0[['SK_ID_CURR', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR').sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'ATM', 'AMT_DRAWINGS_POS_CURRENT':'POS', 'AMT_DRAWINGS_OTHER_CURRENT':'OTHER'})
t2['AMT_DRAWINGS_main'] = t2[['ATM', 'POS', 'OTHER']].idxmax(axis=1)
t3 = pd.get_dummies(data=t2, columns=['AMT_DRAWINGS_main'], dummy_na=False)
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'AMT_DRAWINGS_main_ATM', 'AMT_DRAWINGS_main_OTHER', 'AMT_DRAWINGS_main_POS']])
del t1, t2, t3


# 16. AMT_DRAWINGS 取款方式是否存在ATM POS OTHER Flag => SK_ID_CURR Flag
t1 = t0[['SK_ID_CURR', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR').max().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'ATM', 'AMT_DRAWINGS_POS_CURRENT':'POS', 'AMT_DRAWINGS_OTHER_CURRENT':'OTHER'})
t2['ATM_flag'] = np.where(t2['ATM'] > 0, 1, 0)
t2['POS_flag'] = np.where(t2['POS'] > 0, 1, 0)
t2['OTHER_flag'] = np.where(t2['OTHER'] > 0, 1, 0)
ccd_fe = add_feas(ccd_fe, t2[['SK_ID_CURR', 'ATM_flag', 'POS_flag', 'OTHER_flag']])
del t1, t2

# 17. AMT_DRAWINGS 首次支用金额（4个） / 最早那笔授信额度 stat=> SK_ID_CURR: stat(active)
t1 = t0.loc[t0['AMT_DRAWINGS_CURRENT'] != 0, ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_PREV'], keep='first').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_first_neq0'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft']], on='SK_ID_PREV')
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_first_neq0', y='AMT_CREDIT_LIMIT_ACTUAL_ft')
for x in ['AMT_DRAWINGS_CURRENT_first_neq0', 'AMT_DRAWINGS_CURRENT_first_neq0/AMT_CREDIT_LIMIT_ACTUAL_ft']:
    t5 = t4.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4

t1 = t0.loc[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_PREV'], keep='first').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_first_neq0_act'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft']], on='SK_ID_PREV')
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_first_neq0_act', y='AMT_CREDIT_LIMIT_ACTUAL_ft')
for x in ['AMT_DRAWINGS_CURRENT_first_neq0_act', 'AMT_DRAWINGS_CURRENT_first_neq0_act/AMT_CREDIT_LIMIT_ACTUAL_ft']:
    t5 = t4.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4

# 18. AMT_DRAWINGS 最后支用金额（4个）stat / 最近那笔授信额度=> SK_ID_CURR: stat(active)
t1 = t0.loc[t0['AMT_DRAWINGS_CURRENT'] != 0, ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_PREV'], keep='last').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_last_neq0'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft']], on='SK_ID_PREV')
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_last_neq0', y='AMT_CREDIT_LIMIT_ACTUAL_ft')
for x in ['AMT_DRAWINGS_CURRENT_last_neq0', 'AMT_DRAWINGS_CURRENT_last_neq0/AMT_CREDIT_LIMIT_ACTUAL_ft']:
    t5 = t4.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4

t1 = t0.loc[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_PREV'], keep='last').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_last_neq0_act'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL_ft']], on='SK_ID_PREV')
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_last_neq0_act', y='AMT_CREDIT_LIMIT_ACTUAL_ft')
for x in ['AMT_DRAWINGS_CURRENT_last_neq0_act', 'AMT_DRAWINGS_CURRENT_last_neq0_act/AMT_CREDIT_LIMIT_ACTUAL_ft']:
    t5 = t4.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t5)
    del t5
del t1, t2, t3, t4


# 19. 最低分期付款 stat => SK_ID_CURR: stat(active)（支付金额）（总支付金额）
t1 = stat_f(data=t0, x='AMT_INST_MIN_REGULARITY', stat_to_CURR=True)
t2 = stat_f(data=t0, x='AMT_PAYMENT_CURRENT', stat_to_CURR=True)
t3 = stat_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT', stat_to_CURR=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

t1 = stat_f(data=t0, x='AMT_INST_MIN_REGULARITY', stat_to_CURR=True, active=True)
t2 = stat_f(data=t0, x='AMT_PAYMENT_CURRENT', stat_to_CURR=True, active=True)
t3 = stat_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT', stat_to_CURR=True, active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 20. 最低分期付款/额度 /总支用 stat  => SK_ID_CURR: stat(active)（支付金额）（总支付金额）
t1 = t0[['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_DRAWINGS_CURRENT', add_1=True)
for x in ['AMT_INST_MIN_REGULARITY/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_INST_MIN_REGULARITY/AMT_DRAWINGS_CURRENT',
          'AMT_PAYMENT_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_PAYMENT_CURRENT/AMT_DRAWINGS_CURRENT',
       'AMT_PAYMENT_TOTAL_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_PAYMENT_TOTAL_CURRENT/AMT_DRAWINGS_CURRENT']:
    t2 = stat_f(data=t1, x=x, stat_to_CURR=True)
    ccd_fe = add_feas(ccd_fe, t2)
    del t2
del t1

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t1 = t1.rename(str, columns={'AMT_CREDIT_LIMIT_ACTUAL':'AMT_CREDIT_LIMIT_ACTUAL_act', 'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_act'})
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_CREDIT_LIMIT_ACTUAL_act', add_1=True)
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_DRAWINGS_CURRENT_act', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL_act', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_DRAWINGS_CURRENT_act', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL_act', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_DRAWINGS_CURRENT_act', add_1=True)
for x in ['AMT_INST_MIN_REGULARITY/AMT_CREDIT_LIMIT_ACTUAL_act', 'AMT_INST_MIN_REGULARITY/AMT_DRAWINGS_CURRENT_act',
          'AMT_PAYMENT_CURRENT/AMT_CREDIT_LIMIT_ACTUAL_act', 'AMT_PAYMENT_CURRENT/AMT_DRAWINGS_CURRENT_act',
          'AMT_PAYMENT_TOTAL_CURRENT/AMT_CREDIT_LIMIT_ACTUAL_act', 'AMT_PAYMENT_TOTAL_CURRENT/AMT_DRAWINGS_CURRENT_act']:
    t2 = stat_f(data=t1, x=x, stat_to_CURR=True)
    ccd_fe = add_feas(ccd_fe, t2)
    del t2
del t1


# 21. 最低分期付款 平均每月金额（平均每月金额/总期数/最早期数/余额非0期数） stat => SK_ID_CURR: stat(active)（支付金额）（总支付金额）
# 22. 最低分期付款 第一次期数 第一次期数/总期数 第一次期数/最早那期 第一次期数-第一次支用期数/总期数 第一次期数-第一次支用期数/最早那期 stat   => SK_ID_CURR: stat(active)（支付金额）（总支付金额）
# 23. 最低分期付款金额=0的次数/余额！=0的期数 （支付金额）（总支付金额）
# 23. 支付金额-最低分期付款金额 stat 支付金额-最低分期付款金额/最低分期付款金额 支付金额-最低分期付款金额/支付总金额 stat
# 24. 支付金额-最低分期付款金额=0次数/支付金额!=0次数
# 25. 总支付金额-最低分期付款金额 stat 总支付金额-最低分期付款金额/最低分期付款金额 总支付金额-最低分期付款金额/总支付金额 stat
# 26. 总支付金额-最低分期付款金额=0次数/总支付金额!=0次数
# 27. 总支付金额-支付金额/总支付金额
# 28. 总支付金额-支付金额>0次数/总支付金额！=0次数 <0
# 29. 应收本金 应收金额 应收总金额 stat
# 30. 第一笔应收总金额-第一笔应收本金/第一笔应收总金额  应收总金额-应收本金/应收总金额 stat
# 31. 应收本金（应收金额）（应收总金额）-最低分期（支付金额）（支付总金额）/应收本金（应收金额）（应收总金额）-余额
# 32. 应收本金（应收金额）（应收总金额）-总支用/应收本金（应收金额）（应收总金额）-余额
# 33. 趋势 是否递减
# 34. 应收本金（应收金额）（应收总金额）-余额/应收本金（应收金额）（应收总金额）
# 35. 授信额度-应收本金（应收金额）（应收总金额）/授信额度
# 36. 应收总金额-余额>0次数/应收总金额！=0次数 应收总金额-余额<0次数/应收总金额！=0次数
# 37. 支用：stat
# 38. 支用金额sum/支用次数sum
# 39. 单笔支用最高金额 每条记录支用金额/次数 max  /授信额度  /单笔平均支用金额  /
# 40. 单笔支用最低金额 每条记录支用金额/次数 min  /授信额度  /单笔平均支用金额  /
# 41. 每笔支用占比 stat   sum->支用占比
# 42. 支用最多（次数）方式 oh encoding   占比
# 43. 是否三种支用方式都用
# 44. 是否只有一种支用方式
# 45. 支用次数最大-支用次数最小（1）/支用次数最大 /平均只有次数
# 46. 总支用最长连续支用月份数 /最期数 /最早期数
# 47. 是否中间停止过支用 次数 /（第一次支用期数到最后一次支用期数）
# 48. NAME_CONTRACT_STATUS SK_DPD SK_DPD_DEF 参考bureau balance
# 49. 错位 +-*/



# to_hdf
ccd_fe = ccd_fe.fillna(-999)
print("pos_fe : ", ccd_fe.shape[1] - 1)
ccd_fe.to_hdf('Data_/Pos/credict_card_balance.hdf', 'credict_card_balance', mode='w', format='table')



