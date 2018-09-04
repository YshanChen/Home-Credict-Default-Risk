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
def add_feas_cont(base, features, on='SK_ID_CURR', how='left'):
    t = base
    for feature in features:
        feature_cp = feature.copy()
        t = t.merge(feature_cp, on=on, how=how)
    return t
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
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_CURR', 'MONTHS_BALANCE', x]].copy()
        nf = x + '_active_firstMon'
    else:
        t1 = data[['SK_ID_CURR', 'MONTHS_BALANCE', x]].copy()
        nf = x + '_firstMon'

    t2 = t1[t1[x] != 0]
    t3 = t2.sort_values(by=['SK_ID_CURR', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_CURR']).rename(str, columns={'MONTHS_BALANCE': nf})
    t3[nf] = -t3[nf]
    feature = add_feas(ccd_base, t3[['SK_ID_CURR', nf]])
    return feature


def div_f(data, x, y, add_1=False, x_neg=False, y_neg=False):
    t1 = data.copy()
    nf = x + '/' + y
    print(nf)

    if x_neg:
        t1[x] = -t1[x]
    if y_neg:
        t1[y] = -t1[y]

    if add_1:
        t1[nf] = t1[x] / (t1[y]+1)
    else:
        t1[nf] = t1[x] / t1[y]
    return t1

def substr(data, x, y, x_neg=False, y_neg=False):
    t1 = data.copy()
    nf = x + '_sub_' + y
    print(nf)

    if x_neg:
        t1[x] = -t1[x]
    if y_neg:
        t1[y] = -t1[y]

    t1[nf] = t1[x] - t1[y]

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
del t1, t2, t3, t4, t5, t6, t7, remain_prev_list

# 删除多余PREV
del_prev_list = del_prev_list.tolist()
t0 = t0[~t0['SK_ID_PREV'].isin(del_prev_list)]  # 3840312-3815880 = 24432

# check
# t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'ccd_status']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'])
# t2 = t1.groupby(by="SK_ID_CURR")['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt'})
# (t2['SK_ID_PREV_cnt']>1).any() # False

# 不需要SK_ID_PREV 直接SK_ID_CURR
t0 = t0.drop(['SK_ID_PREV'], axis=1)

# Base
ccd_base = t0[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')

def first_last_f(data, x, sortValues='MONTHS_BALANCE', active=False, extra=False, base=ccd_base):
    if x == 'MONTHS_BALANCE':
        if active:
            t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_CURR', x]].copy()
            nf = x + '_active'
        else:
            t1 = data[['SK_ID_CURR', x]].copy()
            nf = x
    else:
        if active:
            t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_CURR', sortValues, x]].copy()
            nf = x + '_active'
        else:
            t1 = data[['SK_ID_CURR', sortValues, x]].copy()
            nf = x

    nf_first = nf +'_ft'
    nf_last = nf +'_lt'

    t2_1 = t1.sort_values(by=['SK_ID_CURR', sortValues]).drop_duplicates(subset=['SK_ID_CURR'], keep='first').rename(str, columns={x: nf_first})
    t2_2 = t1.sort_values(by=['SK_ID_CURR', sortValues]).drop_duplicates(subset=['SK_ID_CURR'], keep='last').rename(str, columns={x: nf_last})
    t3 = add_feas(base, t2_1[['SK_ID_CURR', nf_first]])
    t3 = add_feas(t3, t2_2[['SK_ID_CURR', nf_last]])

    if extra:
        t3 = t3[['SK_ID_CURR', nf_first, nf_last]]
        t3[nf_first + '-' + nf_last] = t3[nf_first] - t3[nf_last]
        t3['rto-' + nf_first + '-' + nf_last] = t3[nf_first + '-' + nf_last] / t3[nf_first]
        t3[nf_last + '/' + nf_first] = t3[nf_last] / t3[nf_first]
        t3[nf_first + '-' + nf_last + '_b0_flag'] = np.where(t3[nf_first + '-' + nf_last] > 0, 1, 0)
        t3[nf_first + '-' + nf_last + '_l0_flag'] = np.where(t3[nf_first + '-' + nf_last] < 0, 1, 0)
    feature = t3.copy()
    return feature
def count_f(data, x, active=False, base=ccd_base):
    if active:
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_CURR', x]].copy()
        nf = x + '_count_active'
    else:
        t1 = data[['SK_ID_CURR', x]].copy()
        nf = x + '_count'
    t2 = t1.groupby(by='SK_ID_CURR')[x].count().reset_index().rename(str, columns={x: nf})
    t3 = add_feas(base, t2)
    t3 = t3[['SK_ID_CURR', nf]].copy()
    feature = t3.copy()
    return feature
def stat_f(data, x, active=False, agg_list=['max', 'min', 'mean', 'sum', 'std'], base=ccd_base):
    if active:
        t1 = data.loc[data['ccd_status'] == 0, ['SK_ID_CURR', x]].copy()
        nf = x + '_active'
    else:
        t1 = data[['SK_ID_CURR', x]].copy()
        nf = x + ''
    t2 = t1.groupby('SK_ID_CURR')[x].agg(agg_list).reset_index().rename(str, columns={'max': nf+'_max', 'min': nf+'_min', 'mean': nf+'_mean', 'sum': nf+'_sum', 'std': nf+'_std'})
    t3 = add_feas(base, t2)
    feature = t3
    return feature

# 2. SK_ID_PREV: months_balance count 最早 最近 stat => SK_ID_CURR: stat (active)
# 最早一笔-最近一笔 最早一笔-最近一笔/最早一笔 最近那笔/最早那笔
t1 = count_f(data=t0, x='MONTHS_BALANCE')
ccd_fe = add_feas(ccd_fe, t1, on="SK_ID_CURR")
ccd_fe = ccd_fe.fillna(0)
del t1

t1 = first_last_f(data=t0, x='MONTHS_BALANCE')
t2 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
ccd_fe = add_feas(ccd_fe, t1, on='SK_ID_CURR')
ccd_fe = add_feas(ccd_fe, t2, on='SK_ID_CURR')
del t1, t2

# 3. SK_ID_PREV: Months_balance 最早期数-最晚时间/最早期数（as 5/57）/count stat  =》 stat  (未结清)
t1 = first_last_f(data=t0, x='MONTHS_BALANCE')
t1['mon_diff_rto'] = (t1['MONTHS_BALANCE_lt'] - t1['MONTHS_BALANCE_ft']) / (-t1['MONTHS_BALANCE_ft'])
t2 = count_f(data=t0, x='MONTHS_BALANCE')
t3 = add_feas(ccd_base, t1)
t3 = add_feas(t1, t2[['SK_ID_CURR', 'MONTHS_BALANCE_count']])
t3['mon_diff_rto_cnt'] = (t3['MONTHS_BALANCE_lt'] - t3['MONTHS_BALANCE_ft']) / (t3['MONTHS_BALANCE_count'])
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'mon_diff_rto', 'mon_diff_rto_cnt']])
del t1, t2, t3

t1 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t1['mon_diff_rto_active'] = (t1['MONTHS_BALANCE_active_lt'] - t1['MONTHS_BALANCE_active_ft']) / (-t1['MONTHS_BALANCE_active_ft'])
t2 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t3 = add_feas(ccd_base, t1)
t3 = add_feas(t1, t2[['SK_ID_CURR', 'MONTHS_BALANCE_count_active']])
t3['mon_diff_rto_cnt_active'] = (t3['MONTHS_BALANCE_active_lt'] - t3['MONTHS_BALANCE_active_ft']) / (t3['MONTHS_BALANCE_count_active'])
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'mon_diff_rto_active', 'mon_diff_rto_cnt_active']])
del t1, t2, t3

# 4. AMT_CREDIT_LIMIT_ACTUAL stat 最早那笔 最近那笔 最早那笔-最近那笔  最早那笔-最近那笔/最早那笔 最近那笔/最早那笔  最早那笔-最近那笔>0flag 最早那笔-最近那笔<0flag (active)
t1 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=True)
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', active=True, extra=True)
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = stat_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
ccd_fe = add_feas(ccd_fe, t1)
del t1

t1 = stat_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', active=True)
ccd_fe = add_feas(ccd_fe, t1)
del t1

# 5. AMT_BALANCE = 0 的期数/总期数 /最早那笔期数 (active)
t1 = t0.loc[t0['AMT_BALANCE'] == 0, ['SK_ID_CURR', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_eq0_cnt'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE')
t4 = count_f(data=t0, x='MONTHS_BALANCE')
t5 = add_feas(ccd_base, t2[['SK_ID_CURR', 'AMT_BALANCE_eq0_cnt']])
t5 = add_feas(t5, t3[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t5 = add_feas(t5, t4[['SK_ID_CURR', 'MONTHS_BALANCE_count']])
t5['AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_ft'] = t5['AMT_BALANCE_eq0_cnt'] / -t5['MONTHS_BALANCE_ft']
t5['AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_count'] = t5['AMT_BALANCE_eq0_cnt'] / t5['MONTHS_BALANCE_count']
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR','AMT_BALANCE_eq0_cnt', 'AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_ft', 'AMT_BALANCE_eq0_cnt/MONTHS_BALANCE_count']])
del t1, t2, t3, t4, t5

t1 = t0.loc[(t0['AMT_BALANCE'] == 0) & (t0['ccd_status'] == 0), ['SK_ID_CURR', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_eq0_cnt_active'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t4 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t5 = add_feas(ccd_base, t2[['SK_ID_CURR', 'AMT_BALANCE_eq0_cnt_active']])
t5 = add_feas(t5, t3[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t5 = add_feas(t5, t4[['SK_ID_CURR', 'MONTHS_BALANCE_count_active']])
t5['AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_active_ft'] = t5['AMT_BALANCE_eq0_cnt_active'] / -t5['MONTHS_BALANCE_active_ft']
t5['AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_count_active'] = t5['AMT_BALANCE_eq0_cnt_active'] / t5['MONTHS_BALANCE_count_active']
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', 'AMT_BALANCE_eq0_cnt_active', 'AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_active_ft', 'AMT_BALANCE_eq0_cnt_active/MONTHS_BALANCE_count_active']])
del t1, t2, t3, t4, t5

# 5. AMT_BALANCE 是否支用 flag(active)
t1 = t0[['SK_ID_CURR', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_BALANCE'].max().reset_index().rename(str, columns={'AMT_BALANCE':'encash_flag_t'})
t2['encash_flag'] = np.where(t2['encash_flag_t'] > 0, 1, 0)
t3 = add_feas(ccd_base, t2[['SK_ID_CURR', 'encash_flag']])
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'AMT_BALANCE']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_BALANCE'].max().reset_index().rename(str, columns={'AMT_BALANCE':'encash_flag_t'})
t2['active_encash_flag'] = np.where(t2['encash_flag_t'] > 0, 1, 0)
t3 = add_feas(ccd_base, t2[['SK_ID_CURR', 'active_encash_flag']])
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 6. AMT_BALANCE 最早那笔余额(+非0) 最早支用那笔余额/最早那笔授信额度/最近那笔授信额度    最大那笔余额 最大那笔余额/最早那笔授信额度/最近那笔授信额度  (active)
t1 = first_last_f(data=t0, x='AMT_BALANCE', extra=False)
t1 = t1[['SK_ID_CURR', 'AMT_BALANCE_ft']]
t2 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE']].copy()
t3 = first_last_f(data=t2, x='AMT_BALANCE', extra=False)
t3 = t3[['SK_ID_CURR', 'AMT_BALANCE_ft']].rename(str, columns={'AMT_BALANCE_ft': 'AMT_BALANCE_ft_encash'})
t4 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False)
t4 = t4[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft'] = t5['AMT_BALANCE_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_ft']+1)
t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt'] = t5['AMT_BALANCE_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_lt']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft'] = t5['AMT_BALANCE_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_ft']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt'] = t5['AMT_BALANCE_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_lt']+1)
t5 = t5.fillna(0)
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', 'AMT_BALANCE_ft', 'AMT_BALANCE_ft_encash', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt']])
del t1, t2, t3, t4, t5

t1 = first_last_f(data=t0, x='AMT_BALANCE', extra=False, active=True)
t1 = t1[['SK_ID_CURR', 'AMT_BALANCE_active_ft']]
t2 = t0.loc[(t0['AMT_BALANCE'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE', 'ccd_status']].copy()
t3 = first_last_f(data=t2, x='AMT_BALANCE', extra=False, active=True)
t3 = t3[['SK_ID_CURR', 'AMT_BALANCE_active_ft']].rename(str, columns={'AMT_BALANCE_active_ft': 'AMT_BALANCE_active_ft_encash'})
t4 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, active=True)
t4 = t4[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']]
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft_act'] = t5['AMT_BALANCE_active_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_active_ft']+1)
t5['AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt_act'] = t5['AMT_BALANCE_active_ft'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_active_lt']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft_act'] = t5['AMT_BALANCE_active_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_active_ft']+1)
t5['AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt_act'] = t5['AMT_BALANCE_active_ft_encash'] / (t5['AMT_CREDIT_LIMIT_ACTUAL_active_lt']+1)
t5 = t5.fillna(0)
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', 'AMT_BALANCE_active_ft', 'AMT_BALANCE_active_ft_encash', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_ft_act', 'AMT_BALANCE_ft/AMT_CREDIT_LIMIT_ACTUAL_lt_act', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_ft_act', 'AMT_BALANCE_ft_encash/AMT_CREDIT_LIMIT_ACTUAL_lt_act']])
del t1, t2, t3, t4, t5

# 7. AMT_BALANCE 第一次支用期数 第一次支用期数/总期数 第一次支用期数/最早那期 stat   => SK_ID_CURR: stat(active)
t1 = first_mon_f(data=t0, x='AMT_BALANCE')
t2 = first_last_f(data=t0, x='MONTHS_BALANCE')
t3 = count_f(data=t0, x='MONTHS_BALANCE')
t4 = add_feas(ccd_base, t1)
t4 = add_feas(t4, t2[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t4 = add_feas(t4, t3[['SK_ID_CURR', 'MONTHS_BALANCE_count']])
t4['AMT_BALANCE_firstMon/MONTHS_BALANCE_ft'] = t4['AMT_BALANCE_firstMon'] / -t4['MONTHS_BALANCE_ft']
t4['AMT_BALANCE_firstMon/MONTHS_BALANCE_count'] = t4['AMT_BALANCE_firstMon'] / t4['MONTHS_BALANCE_count']
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_BALANCE_firstMon', 'AMT_BALANCE_firstMon/MONTHS_BALANCE_ft', 'AMT_BALANCE_firstMon/MONTHS_BALANCE_count']])
del t1, t2, t3, t4

t1 = first_mon_f(data=t0, x='AMT_BALANCE', active=True)
t2 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t3 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t4 = add_feas(ccd_base, t1)
t4 = add_feas(t4, t2[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t4 = add_feas(t4, t3[['SK_ID_CURR', 'MONTHS_BALANCE_count_active']])
t4['AMT_BALANCE_active_firstMon/MONTHS_BALANCE_active_ft'] = t4['AMT_BALANCE_active_firstMon'] / -t4['MONTHS_BALANCE_active_ft']
t4['AMT_BALANCE_active_firstMon/MONTHS_BALANCE_count_active'] = t4['AMT_BALANCE_active_firstMon'] / t4['MONTHS_BALANCE_count_active']
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_BALANCE_active_firstMon', 'AMT_BALANCE_active_firstMon/MONTHS_BALANCE_active_ft', 'AMT_BALANCE_active_firstMon/MONTHS_BALANCE_count_active']])
del t1, t2, t3, t4

# 8. AMT_BALANCE stat stat/最早那笔授信额度/最近那笔授信额度   排除掉0的stat/最早那笔授信额度/最近那笔授信额度  (active)
t1 = stat_f(data=t0, x='AMT_BALANCE')
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False)
t2 = t2[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t3 = add_feas(ccd_base, t1)
t3 = add_feas(t3, t2)
for feas in ['AMT_BALANCE_max', 'AMT_BALANCE_min', 'AMT_BALANCE_mean', 'AMT_BALANCE_sum', 'AMT_BALANCE_std']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]:
    print(feas)
    ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', feas]])
del t1, t2, t3

t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE']].copy().rename(columns={'AMT_BALANCE': 'AMT_BALANCE_encash'})
t1 = stat_f(data=t1, x='AMT_BALANCE_encash')
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False)
t2 = t2[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]
t3 = add_feas(ccd_base, t1)
t3 = add_feas(t3, t2)
for feas in ['AMT_BALANCE_encash_max', 'AMT_BALANCE_encash_min', 'AMT_BALANCE_encash_mean', 'AMT_BALANCE_encash_sum', 'AMT_BALANCE_encash_std']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft', 'AMT_CREDIT_LIMIT_ACTUAL_lt']]:
    print(feas)
    ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', feas]])
del t1, t2, t3

t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE', 'ccd_status']].copy().rename(columns={'AMT_BALANCE': 'AMT_BALANCE_encash'})
t1 = stat_f(data=t1, x='AMT_BALANCE_encash', active=True)
t2 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', extra=False, active=True)
t2 = t2[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']]
t3 = add_feas(ccd_base, t1)
t3 = add_feas(t3, t2)
for feas in ['AMT_BALANCE_encash_active_max', 'AMT_BALANCE_encash_active_min','AMT_BALANCE_encash_active_mean', 'AMT_BALANCE_encash_active_sum', 'AMT_BALANCE_encash_active_std']:
    nf1 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_active_ft'
    nf2 = feas + '_/AMT_CREDIT_LIMIT_ACTUAL_active_lt'
    t3[nf1] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_active_ft'] + 0)
    t3[nf2] = t3[feas] / (t3['AMT_CREDIT_LIMIT_ACTUAL_active_lt'] + 0)
for feas in [x for x in t3.columns if x not in ['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']]:
    print(feas)
    ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', feas]])
del t1, t2, t3

gc.collect()

# 9. AMT_BALANCE diff(-1,-2) stat stat/最早那笔授信额度/最近那笔授信额度 排除掉0的diff(-1,-2) stat/最早那笔授信额度/最近那笔授信额度   => SK_ID_CURR: stat(active)
# 10. AMT_BALANCE 余额/授信额度 diff(-1,-2) stat 排除掉0的 => SK_ID_CURR: stat(active)
# 11. AMT_BALANCE 余额/授信额度 是否连续2\3期下降（diff 均小等于0）排除掉0的 => SK_ID_CURR: flag(active)



# 12. AMT_DRAWINGS 4个AMT_DRAWINGS stat 3个sum占总AMT_DRAWINGS sum比重 3个mean占总AMT_DRAWINGS mean比重 stat (active)
t1 = stat_f(data=t0, x='AMT_DRAWINGS_ATM_CURRENT')
t2 = stat_f(data=t0, x='AMT_DRAWINGS_POS_CURRENT')
t3 = stat_f(data=t0, x='AMT_DRAWINGS_OTHER_CURRENT')
t4 = stat_f(data=t0, x='AMT_DRAWINGS_CURRENT')
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = stat_f(data=t0[t0['AMT_DRAWINGS_ATM_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_ATM_CURRENT': 'AMT_DRAWINGS_ATM_CURRENT_encash'}), x='AMT_DRAWINGS_ATM_CURRENT_encash')
t2 = stat_f(data=t0[t0['AMT_DRAWINGS_POS_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_POS_CURRENT': 'AMT_DRAWINGS_POS_CURRENT_encash'}), x='AMT_DRAWINGS_POS_CURRENT_encash')
t3 = stat_f(data=t0[t0['AMT_DRAWINGS_OTHER_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_OTHER_CURRENT': 'AMT_DRAWINGS_OTHER_CURRENT_encash'}), x='AMT_DRAWINGS_OTHER_CURRENT_encash')
t4 = stat_f(data=t0[t0['AMT_DRAWINGS_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_encash'}), x='AMT_DRAWINGS_CURRENT_encash')
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = stat_f(data=t0, x='AMT_DRAWINGS_ATM_CURRENT', active=True)
t2 = stat_f(data=t0, x='AMT_DRAWINGS_POS_CURRENT', active=True)
t3 = stat_f(data=t0, x='AMT_DRAWINGS_OTHER_CURRENT', active=True)
t4 = stat_f(data=t0, x='AMT_DRAWINGS_CURRENT', active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = stat_f(data=t0[t0['AMT_DRAWINGS_ATM_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_ATM_CURRENT': 'AMT_DRAWINGS_ATM_CURRENT_encash'}), x='AMT_DRAWINGS_ATM_CURRENT_encash', active=True)
t2 = stat_f(data=t0[t0['AMT_DRAWINGS_POS_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_POS_CURRENT': 'AMT_DRAWINGS_POS_CURRENT_encash'}), x='AMT_DRAWINGS_POS_CURRENT_encash', active=True)
t3 = stat_f(data=t0[t0['AMT_DRAWINGS_OTHER_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_OTHER_CURRENT': 'AMT_DRAWINGS_OTHER_CURRENT_encash'}), x='AMT_DRAWINGS_OTHER_CURRENT_encash', active=True)
t4 = stat_f(data=t0[t0['AMT_DRAWINGS_CURRENT'] != 0].copy().rename(str,columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_encash'}), x='AMT_DRAWINGS_CURRENT_encash', active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4
gc.collect()

t1 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_tot'})
t2 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_tot'})
t3 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_tot'})
t4 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_tot'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5 = t5.fillna(0)
t5['AMT_DRAWINGS_ATM_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot'] = t5['AMT_DRAWINGS_ATM_CURRENT_tot'] / (t5['AMT_DRAWINGS_CURRENT_tot']+1)
t5['AMT_DRAWINGS_POS_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot'] = t5['AMT_DRAWINGS_POS_CURRENT_tot'] / (t5['AMT_DRAWINGS_CURRENT_tot']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot'] = t5['AMT_DRAWINGS_OTHER_CURRENT_tot'] / (t5['AMT_DRAWINGS_CURRENT_tot']+1)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_tot_act'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_tot_act'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_tot_act'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_tot_act'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5 = t5.fillna(0)
t5['AMT_DRAWINGS_ATM_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot_act'] = t5['AMT_DRAWINGS_ATM_CURRENT_tot_act'] / (t5['AMT_DRAWINGS_CURRENT_tot_act']+1)
t5['AMT_DRAWINGS_POS_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot_act'] = t5['AMT_DRAWINGS_POS_CURRENT_tot_act'] / (t5['AMT_DRAWINGS_CURRENT_tot_act']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_tot/AMT_DRAWINGS_CURRENT_tot_act'] = t5['AMT_DRAWINGS_OTHER_CURRENT_tot_act'] / (t5['AMT_DRAWINGS_CURRENT_tot_act']+1)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_totmean'})
t2 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_totmean'})
t3 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_totmean'})
t4 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_totmean'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5 = t5.fillna(0)
t5['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_ATM_CURRENT_totmean'] / (t5['AMT_DRAWINGS_CURRENT_totmean']+1)
t5['AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_POS_CURRENT_totmean'] / (t5['AMT_DRAWINGS_CURRENT_totmean']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean'] = t5['AMT_DRAWINGS_OTHER_CURRENT_totmean'] / (t5['AMT_DRAWINGS_CURRENT_totmean']+1)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5
gc.collect()

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_totmean_act'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_totmean_act'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_totmean_act'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].mean().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_totmean_act'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t5 = t5.fillna(0)
t5['AMT_DRAWINGS_ATM_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_ATM_CURRENT_totmean_act'] / (t5['AMT_DRAWINGS_CURRENT_totmean_act']+1)
t5['AMT_DRAWINGS_POS_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_POS_CURRENT_totmean_act'] / (t5['AMT_DRAWINGS_CURRENT_totmean_act']+1)
t5['AMT_DRAWINGS_OTHER_CURRENT_mean/AMT_DRAWINGS_CURRENT_mean_act'] = t5['AMT_DRAWINGS_OTHER_CURRENT_totmean_act'] / (t5['AMT_DRAWINGS_CURRENT_totmean_act']+1)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5
gc.collect()

# 13. AMT_DRAWINGS 不为0次数/总期数/最早期数/余额非0期数
t1 = t0[t0['AMT_DRAWINGS_CURRENT'] != 0]
t2 = t1.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_no0'})
t3 = first_last_f(data=t0, x='MONTHS_BALANCE')
t4 = count_f(data=t0, x='MONTHS_BALANCE')
t5 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_CURR', 'AMT_BALANCE']].copy()
t6 = t5.groupby('SK_ID_CURR')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt'})
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_count']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_neq0_cnt']])
t7 = t7.fillna(0)
t7['AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_ft'] = t7['AMT_DRAWINGS_CURRENT_no0'] / -t7['MONTHS_BALANCE_ft']
t7['AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_count'] = t7['AMT_DRAWINGS_CURRENT_no0'] / t7['MONTHS_BALANCE_count']
t7['AMT_DRAWINGS_CURRENT_no0/AMT_BALANCE_neq0_cnt'] = t7['AMT_DRAWINGS_CURRENT_no0'] / t7['AMT_BALANCE_neq0_cnt']
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_no0', 'AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_ft', 'AMT_DRAWINGS_CURRENT_no0/MONTHS_BALANCE_count', 'AMT_DRAWINGS_CURRENT_no0/AMT_BALANCE_neq0_cnt']])
del t1, t2, t3, t4, t5, t6, t7

# 13. AMT_DRAWINGS 平均每月支用金额（4class_AMT_DRAWINGS/总期数/最早期数/不为0次数/余额非0期数） stat(active)
t1 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t6 = first_last_f(data=t0, x='MONTHS_BALANCE')
t6['MONTHS_BALANCE_ft'] = -t6['MONTHS_BALANCE_ft']
t7 = count_f(data=t0, x='MONTHS_BALANCE')
t8 = t0[t0['AMT_DRAWINGS_CURRENT'] != 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_neq0_cnt'})
t9 = t0[t0['AMT_BALANCE'] != 0].groupby('SK_ID_CURR')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt'})
t10 = add_feas(ccd_base, t5)
t10 = add_feas(t10, t6[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t10 = add_feas(t10, t7[['SK_ID_CURR', 'MONTHS_BALANCE_count']])
t10 = add_feas(t10, t8)
t10 = add_feas(t10, t9)
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_ATM_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt') # 1
t10 = div_f(data=t10, x='AMT_DRAWINGS_POS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt') # 2
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_OTHER_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_ft')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='MONTHS_BALANCE_count')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_DRAWINGS_CURRENT_neq0_cnt')
t10 = div_f(data=t10, x='AMT_DRAWINGS_CURRENT_sum', y='AMT_BALANCE_neq0_cnt')
t10 = t10.fillna(0)
for feas in [x for x in t10.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_DRAWINGS_ATM_CURRENT_sum', 'AMT_DRAWINGS_POS_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_sum', 'AMT_DRAWINGS_CURRENT_sum', 'MONTHS_BALANCE_ft', 'MONTHS_BALANCE_count','AMT_DRAWINGS_CURRENT_neq0_cnt', 'AMT_BALANCE_neq0_cnt']]:
    print(feas)
    ccd_fe = add_feas(ccd_fe, t10[['SK_ID_CURR', feas]])
    gc.collect()
del t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

t1 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_ATM_CURRENT':'AMT_DRAWINGS_ATM_CURRENT_sum'})
t2 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_POS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_POS_CURRENT':'AMT_DRAWINGS_POS_CURRENT_sum'})
t3 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_OTHER_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_OTHER_CURRENT':'AMT_DRAWINGS_OTHER_CURRENT_sum'})
t4 = t0[t0['ccd_status'] == 0].groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_sum'})
t5 = add_feas(ccd_base, t1)
t5 = add_feas(t5, t2)
t5 = add_feas(t5, t3)
t5 = add_feas(t5, t4)
t6 = first_last_f(data=t0, x='MONTHS_BALANCE', active=True)
t6['MONTHS_BALANCE_active_ft'] = -t6['MONTHS_BALANCE_active_ft']
t7 = count_f(data=t0, x='MONTHS_BALANCE', active=True)
t8 = t0[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0)].groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].count().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT':'AMT_DRAWINGS_CURRENT_neq0_cnt_act'})
t9 = t0[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0)].groupby('SK_ID_CURR')['AMT_BALANCE'].count().reset_index().rename(str, columns={'AMT_BALANCE': 'AMT_BALANCE_neq0_cnt_act'})
t10 = add_feas(ccd_base, t5)
t10 = add_feas(t5, t6[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t10 = add_feas(t10, t7[['SK_ID_CURR', 'MONTHS_BALANCE_count_active']])
t10 = add_feas(t10, t8)
t10 = add_feas(t10, t9)
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
for feas in [x for x in t10.columns if x not in ['SK_ID_CURR', 'SK_ID_CURR', 'AMT_DRAWINGS_ATM_CURRENT_sum','AMT_DRAWINGS_POS_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_sum','AMT_DRAWINGS_CURRENT_sum', 'MONTHS_BALANCE_active_ft', 'MONTHS_BALANCE_count_active','AMT_DRAWINGS_CURRENT_neq0_cnt_act', 'AMT_BALANCE_neq0_cnt_act']]:
    print(feas)
    ccd_fe = add_feas(ccd_fe, t10[['SK_ID_CURR', feas]])
del t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

# 14. AMT_DRAWINGS AMT_DRAWINGS/当月授信额度/当月余额（排除余额为0月份） stat (active)
t1 = t0.loc[t0['AMT_BALANCE'] != 0, ['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE']].copy()
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_BALANCE', add_1=True)
t2 = add_feas(ccd_base, t1)
t2 = t2.fillna(0)
for x in ['AMT_DRAWINGS_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT/AMT_BALANCE']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean'})
    ccd_fe = add_feas(ccd_fe, t3)
    del t3
del t1, t2

t1 = t0.loc[(t0['AMT_BALANCE'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE']].copy()
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_DRAWINGS_CURRENT', y='AMT_BALANCE', add_1=True)
t2 = add_feas(ccd_base, t1)
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

# 17. AMT_DRAWINGS 首次支用金额（4个） / 最早那笔授信额度  (active)
t1 = t0.loc[t0['AMT_DRAWINGS_CURRENT'] != 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_CURR'], keep='first').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_first_neq0'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2)
t4 = add_feas(t4, t3[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_ft']])
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_first_neq0', y='AMT_CREDIT_LIMIT_ACTUAL_ft', add_1=True)
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_first_neq0', 'AMT_DRAWINGS_CURRENT_first_neq0/AMT_CREDIT_LIMIT_ACTUAL_ft']])
del t1, t2, t3, t4

t1 = t0.loc[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_CURR'], keep='first').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_first_neq0_act'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', active=True)
t4 = add_feas(ccd_base, t2)
t4 = add_feas(t4, t3[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_active_ft']])
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_first_neq0_act', y='AMT_CREDIT_LIMIT_ACTUAL_active_ft', add_1=True)
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_first_neq0_act', 'AMT_DRAWINGS_CURRENT_first_neq0_act/AMT_CREDIT_LIMIT_ACTUAL_active_ft']])
del t1, t2, t3, t4

# 18. AMT_DRAWINGS 最后支用金额（4个）stat / 最近那笔授信额度=> SK_ID_CURR: stat(active)
t1 = t0.loc[t0['AMT_DRAWINGS_CURRENT'] != 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_CURR'], keep='last').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_last_neq0'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL')
t4 = add_feas(ccd_base, t2)
t4 = add_feas(t4, t3[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_lt']])
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_last_neq0', y='AMT_CREDIT_LIMIT_ACTUAL_lt', add_1=True)
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_last_neq0', 'AMT_DRAWINGS_CURRENT_last_neq0/AMT_CREDIT_LIMIT_ACTUAL_lt']])
del t1, t2, t3, t4

t1 = t0.loc[(t0['AMT_DRAWINGS_CURRENT'] != 0) & (t0['ccd_status'] == 0), ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT']].copy()
t2 = t1.drop_duplicates(subset=['SK_ID_CURR'], keep='last').rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_last_neq0_act'})
t3 = first_last_f(data=t0, x='AMT_CREDIT_LIMIT_ACTUAL', active=True)
t4 = add_feas(ccd_base, t2)
t4 = add_feas(t4, t3[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL_active_lt']])
t4 = t4.fillna(0)
t4 = div_f(data=t4, x='AMT_DRAWINGS_CURRENT_last_neq0_act', y='AMT_CREDIT_LIMIT_ACTUAL_active_lt')
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_last_neq0_act', 'AMT_DRAWINGS_CURRENT_last_neq0_act/AMT_CREDIT_LIMIT_ACTUAL_active_lt']])
del t1, t2, t3, t4

# 19. 最低分期付款 stat => SK_ID_CURR: stat(active)（支付金额）（总支付金额）
t1 = stat_f(data=t0, x='AMT_INST_MIN_REGULARITY')
t2 = stat_f(data=t0, x='AMT_PAYMENT_CURRENT')
t3 = stat_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

t1 = stat_f(data=t0, x='AMT_INST_MIN_REGULARITY', active=True)
t2 = stat_f(data=t0, x='AMT_PAYMENT_CURRENT', active=True)
t3 = stat_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT', active=True)
ccd_fe = add_feas(ccd_fe, t1)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 20. 最低分期付款/额度 /总支用 stat  (active)（支付金额）（总支付金额）
t1 = t0[['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_INST_MIN_REGULARITY', y='AMT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_CURRENT', y='AMT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t1 = div_f(data=t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_DRAWINGS_CURRENT', add_1=True)
for x in ['AMT_INST_MIN_REGULARITY/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_INST_MIN_REGULARITY/AMT_DRAWINGS_CURRENT',
          'AMT_PAYMENT_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_PAYMENT_CURRENT/AMT_DRAWINGS_CURRENT',
       'AMT_PAYMENT_TOTAL_CURRENT/AMT_CREDIT_LIMIT_ACTUAL', 'AMT_PAYMENT_TOTAL_CURRENT/AMT_DRAWINGS_CURRENT']:
    print(x)
    t2 = t1.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ccd_fe = add_feas(ccd_fe, t2)
    del t2
del t1

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
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
    print(x)
    t2 = t1.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={
        'max': x + '_max', 'min': x + '_min', 'mean': x + '_mean', 'sum': x + '_sum', 'std': x + '_std'})
    ccd_fe = add_feas(ccd_fe, t2)
    del t2
del t1


# 21. 最低分期付款 平均每月金额（总金额(非0)/总期数/最早期数/余额非0期数） (active)（支付金额）（总支付金额）
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum().reset_index().rename(str, columns={'AMT_INST_MIN_REGULARITY':'AMT_INST_MIN_REGULARITY_sum'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_sum/MONTHS_BALANCE_count',
            'AMT_INST_MIN_REGULARITY_sum/MONTHS_BALANCE_ft',
            'AMT_INST_MIN_REGULARITY_sum/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['AMT_INST_MIN_REGULARITY'] != 0, ['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum().reset_index().rename(str, columns={'AMT_INST_MIN_REGULARITY':'AMT_INST_MIN_REGULARITY_sum_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_nq0', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_nq0', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_nq0', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_sum_nq0/MONTHS_BALANCE_count',
            'AMT_INST_MIN_REGULARITY_sum_nq0/MONTHS_BALANCE_ft',
            'AMT_INST_MIN_REGULARITY_sum_nq0/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum().reset_index().rename(str, columns={'AMT_INST_MIN_REGULARITY':'AMT_INST_MIN_REGULARITY_sum_act'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_sum_act/MONTHS_BALANCE_count_active',
            'AMT_INST_MIN_REGULARITY_sum_act/MONTHS_BALANCE_active_ft',
            'AMT_INST_MIN_REGULARITY_sum_act/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[(t0['ccd_status'] == 0) & (t0['AMT_INST_MIN_REGULARITY'] != 0), ['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum().reset_index().rename(str, columns={'AMT_INST_MIN_REGULARITY':'AMT_INST_MIN_REGULARITY_sum_act_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act_nq0', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act_nq0', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_INST_MIN_REGULARITY_sum_act_nq0', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_sum_act_nq0/MONTHS_BALANCE_count_active',
            'AMT_INST_MIN_REGULARITY_sum_act_nq0/MONTHS_BALANCE_active_ft',
            'AMT_INST_MIN_REGULARITY_sum_act_nq0/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

# 支付金额
t1 = t0[['SK_ID_CURR', 'AMT_PAYMENT_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_CURRENT':'AMT_PAYMENT_CURRENT_sum'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_sum/MONTHS_BALANCE_count',
            'AMT_PAYMENT_CURRENT_sum/MONTHS_BALANCE_ft',
            'AMT_PAYMENT_CURRENT_sum/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['AMT_PAYMENT_CURRENT'] != 0, ['SK_ID_CURR', 'AMT_PAYMENT_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_CURRENT':'AMT_PAYMENT_CURRENT_sum_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_nq0', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_nq0', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_nq0', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_sum_nq0/MONTHS_BALANCE_count',
            'AMT_PAYMENT_CURRENT_sum_nq0/MONTHS_BALANCE_ft',
            'AMT_PAYMENT_CURRENT_sum_nq0/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'AMT_PAYMENT_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_CURRENT':'AMT_PAYMENT_CURRENT_sum_act'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_sum_act/MONTHS_BALANCE_count_active',
            'AMT_PAYMENT_CURRENT_sum_act/MONTHS_BALANCE_active_ft',
            'AMT_PAYMENT_CURRENT_sum_act/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[(t0['ccd_status'] == 0) & (t0['AMT_PAYMENT_CURRENT'] != 0), ['SK_ID_CURR', 'AMT_PAYMENT_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_CURRENT':'AMT_PAYMENT_CURRENT_sum_act_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act_nq0', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act_nq0', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_CURRENT_sum_act_nq0', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_sum_act_nq0/MONTHS_BALANCE_count_active',
            'AMT_PAYMENT_CURRENT_sum_act_nq0/MONTHS_BALANCE_active_ft',
            'AMT_PAYMENT_CURRENT_sum_act_nq0/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

# 总支付金额
t1 = t0[['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_TOTAL_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_TOTAL_CURRENT':'AMT_PAYMENT_TOTAL_CURRENT_sum'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_sum/MONTHS_BALANCE_count',
            'AMT_PAYMENT_TOTAL_CURRENT_sum/MONTHS_BALANCE_ft',
            'AMT_PAYMENT_TOTAL_CURRENT_sum/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['AMT_PAYMENT_TOTAL_CURRENT'] != 0, ['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_TOTAL_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_TOTAL_CURRENT':'AMT_PAYMENT_TOTAL_CURRENT_sum_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE')
t4 = first_last_f(t0, x='MONTHS_BALANCE')
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE')
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count']])
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_nq0', y='MONTHS_BALANCE_count')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_nq0', y='MONTHS_BALANCE_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_nq0', y='AMT_BALANCE_count')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_sum_nq0/MONTHS_BALANCE_count',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_nq0/MONTHS_BALANCE_ft',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_nq0/AMT_BALANCE_count']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_TOTAL_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_TOTAL_CURRENT':'AMT_PAYMENT_TOTAL_CURRENT_sum_act'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_sum_act/MONTHS_BALANCE_count_active',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_act/MONTHS_BALANCE_active_ft',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_act/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0.loc[(t0['ccd_status'] == 0) & (t0['AMT_PAYMENT_TOTAL_CURRENT'] != 0), ['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['AMT_PAYMENT_TOTAL_CURRENT'].sum().reset_index().rename(str, columns={'AMT_PAYMENT_TOTAL_CURRENT':'AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0'})
t3 = count_f(t0, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t6 = count_f(t5, x='AMT_BALANCE', active=True)
t7 = add_feas(ccd_base, t2)
t7 = add_feas(t7, t3)
t7 = add_feas(t7, t4[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']])
t7 = add_feas(t7, t6[['SK_ID_CURR', 'AMT_BALANCE_count_active']])
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0', y='MONTHS_BALANCE_count_active')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0', y='MONTHS_BALANCE_active_ft')
t7 = div_f(data=t7, x='AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0', y='AMT_BALANCE_count_active')
t7 = t7.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0/MONTHS_BALANCE_count_active',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0/MONTHS_BALANCE_active_ft',
            'AMT_PAYMENT_TOTAL_CURRENT_sum_act_nq0/AMT_BALANCE_count_active']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6, t7

# 22. 最低分期付款 第一次期数 第一次期数/总期数 第一次期数/最早那期 第一次支用期数-第一次期数/总期数 第一次支用期数-第一次期数/最早那期  (active)（支付金额）（总支付金额）
t1 = first_mon_f(data=t0, x='AMT_INST_MIN_REGULARITY')
t2 = count_f(t0, x='AMT_INST_MIN_REGULARITY')
t3 = first_last_f(t0, x='MONTHS_BALANCE')
t3['MONTHS_BALANCE_ft'] = -t3['MONTHS_BALANCE_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT')
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_firstMon', y='AMT_INST_MIN_REGULARITY_firstMon')
t5 = div_f(t5, x='AMT_INST_MIN_REGULARITY_firstMon', y='AMT_INST_MIN_REGULARITY_count')
t5 = div_f(t5, x='AMT_INST_MIN_REGULARITY_firstMon', y='MONTHS_BALANCE_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_INST_MIN_REGULARITY_firstMon', y='AMT_INST_MIN_REGULARITY_count')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_INST_MIN_REGULARITY_firstMon', y='MONTHS_BALANCE_ft')
t5 = t5.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_firstMon',
            'AMT_INST_MIN_REGULARITY_firstMon/AMT_INST_MIN_REGULARITY_count',
            'AMT_INST_MIN_REGULARITY_firstMon/MONTHS_BALANCE_ft',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_INST_MIN_REGULARITY_firstMon/AMT_INST_MIN_REGULARITY_count',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_INST_MIN_REGULARITY_firstMon/MONTHS_BALANCE_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5

t1 = first_mon_f(data=t0, x='AMT_INST_MIN_REGULARITY', active=True)
t2 = count_f(t0, x='AMT_INST_MIN_REGULARITY', active=True)
t3 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t3['MONTHS_BALANCE_active_ft'] = -t3['MONTHS_BALANCE_active_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT', active=True)
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon', y='AMT_INST_MIN_REGULARITY_active_firstMon')
t5 = div_f(t5, x='AMT_INST_MIN_REGULARITY_active_firstMon', y='AMT_INST_MIN_REGULARITY_count_active')
t5 = div_f(t5, x='AMT_INST_MIN_REGULARITY_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_INST_MIN_REGULARITY_active_firstMon', y='AMT_INST_MIN_REGULARITY_count_active')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_INST_MIN_REGULARITY_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = t5.fillna(0)
for fea in ['AMT_INST_MIN_REGULARITY_active_firstMon',
            'AMT_INST_MIN_REGULARITY_active_firstMon/AMT_INST_MIN_REGULARITY_count_active',
            'AMT_INST_MIN_REGULARITY_active_firstMon/MONTHS_BALANCE_active_ft',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_INST_MIN_REGULARITY_active_firstMon/AMT_INST_MIN_REGULARITY_count_active',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_INST_MIN_REGULARITY_active_firstMon/MONTHS_BALANCE_active_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5

t1 = first_mon_f(data=t0, x='AMT_PAYMENT_CURRENT')
t2 = count_f(t0, x='AMT_PAYMENT_CURRENT')
t3 = first_last_f(t0, x='MONTHS_BALANCE')
t3['MONTHS_BALANCE_ft'] = -t3['MONTHS_BALANCE_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT')
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_firstMon', y='AMT_PAYMENT_CURRENT_firstMon')
t5 = div_f(t5, x='AMT_PAYMENT_CURRENT_firstMon', y='AMT_PAYMENT_CURRENT_count')
t5 = div_f(t5, x='AMT_PAYMENT_CURRENT_firstMon', y='MONTHS_BALANCE_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_CURRENT_firstMon', y='AMT_PAYMENT_CURRENT_count')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_CURRENT_firstMon', y='MONTHS_BALANCE_ft')
t5 = t5.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_firstMon',
            'AMT_PAYMENT_CURRENT_firstMon/AMT_PAYMENT_CURRENT_count',
            'AMT_PAYMENT_CURRENT_firstMon/MONTHS_BALANCE_ft',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_CURRENT_firstMon/AMT_PAYMENT_CURRENT_count',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_CURRENT_firstMon/MONTHS_BALANCE_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5

t1 = first_mon_f(data=t0, x='AMT_PAYMENT_CURRENT', active=True)
t2 = count_f(t0, x='AMT_PAYMENT_CURRENT', active=True)
t3 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t3['MONTHS_BALANCE_active_ft'] = -t3['MONTHS_BALANCE_active_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT', active=True)
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon', y='AMT_PAYMENT_CURRENT_active_firstMon')
t5 = div_f(t5, x='AMT_PAYMENT_CURRENT_active_firstMon', y='AMT_PAYMENT_CURRENT_count_active')
t5 = div_f(t5, x='AMT_PAYMENT_CURRENT_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_CURRENT_active_firstMon', y='AMT_PAYMENT_CURRENT_count_active')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_CURRENT_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = t5.fillna(0)
for fea in ['AMT_PAYMENT_CURRENT_active_firstMon',
            'AMT_PAYMENT_CURRENT_active_firstMon/AMT_PAYMENT_CURRENT_count_active',
            'AMT_PAYMENT_CURRENT_active_firstMon/MONTHS_BALANCE_active_ft',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_CURRENT_active_firstMon/AMT_PAYMENT_CURRENT_count_active',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_CURRENT_active_firstMon/MONTHS_BALANCE_active_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5


t1 = first_mon_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT')
t2 = count_f(t0, x='AMT_PAYMENT_TOTAL_CURRENT')
t3 = first_last_f(t0, x='MONTHS_BALANCE')
t3['MONTHS_BALANCE_ft'] = -t3['MONTHS_BALANCE_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT')
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_firstMon')
t5 = div_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_count')
t5 = div_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT_firstMon', y='MONTHS_BALANCE_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_count')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_firstMon', y='MONTHS_BALANCE_ft')
t5 = t5.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_firstMon',
            'AMT_PAYMENT_TOTAL_CURRENT_firstMon/AMT_PAYMENT_TOTAL_CURRENT_count',
            'AMT_PAYMENT_TOTAL_CURRENT_firstMon/MONTHS_BALANCE_ft',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_firstMon/AMT_PAYMENT_TOTAL_CURRENT_count',
            'AMT_DRAWINGS_CURRENT_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_firstMon/MONTHS_BALANCE_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5

t1 = first_mon_f(data=t0, x='AMT_PAYMENT_TOTAL_CURRENT', active=True)
t2 = count_f(t0, x='AMT_PAYMENT_TOTAL_CURRENT', active=True)
t3 = first_last_f(t0, x='MONTHS_BALANCE', active=True)
t3['MONTHS_BALANCE_active_ft'] = -t3['MONTHS_BALANCE_active_ft']
t4 = first_mon_f(data=t0, x='AMT_DRAWINGS_CURRENT', active=True)
t5 = add_feas_cont(ccd_base, features=[t1, t2, t3[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']], t4])
t5 = substr(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_active_firstMon')
t5 = div_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT_active_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_count_active')
t5 = div_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_active_firstMon', y='AMT_PAYMENT_TOTAL_CURRENT_count_active')
t5 = div_f(t5, x='AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_active_firstMon', y='MONTHS_BALANCE_active_ft')
t5 = t5.fillna(0)
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_active_firstMon',
            'AMT_PAYMENT_TOTAL_CURRENT_active_firstMon/AMT_PAYMENT_TOTAL_CURRENT_count_active',
            'AMT_PAYMENT_TOTAL_CURRENT_active_firstMon/MONTHS_BALANCE_active_ft',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_active_firstMon/AMT_PAYMENT_TOTAL_CURRENT_count_active',
            'AMT_DRAWINGS_CURRENT_active_firstMon_sub_AMT_PAYMENT_TOTAL_CURRENT_active_firstMon/MONTHS_BALANCE_active_ft']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5


# 23. 最低分期付款金额!=0的次数/余额！=0的期数 （支付金额）（总支付金额）
t1 = t0[t0['AMT_INST_MIN_REGULARITY'] != 0].copy()
t2 = count_f(t1, x='AMT_INST_MIN_REGULARITY').rename(str, columns={'AMT_INST_MIN_REGULARITY_count':'AMT_INST_MIN_REGULARITY_count_neq0'})
t3 = t0[t0['AMT_BALANCE'] != 0].copy()
t4 = count_f(t3, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count':'AMT_BALANCE_count_neq0'})
t5 = add_feas_cont(ccd_base, features=[t2, t4])
t6 = div_f(t5, x='AMT_INST_MIN_REGULARITY_count_neq0', y='AMT_BALANCE_count_neq0')
for fea in ['AMT_INST_MIN_REGULARITY_count_neq0/AMT_BALANCE_count_neq0']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6

t1 = t0[t0['AMT_PAYMENT_CURRENT'] != 0].copy()
t2 = count_f(t1, x='AMT_PAYMENT_CURRENT').rename(str, columns={'AMT_PAYMENT_CURRENT_count':'AMT_PAYMENT_CURRENT_count_neq0'})
t3 = t0[t0['AMT_BALANCE'] != 0].copy()
t4 = count_f(t3, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count':'AMT_BALANCE_count_neq0'})
t5 = add_feas_cont(ccd_base, features=[t2, t4])
t6 = div_f(t5, x='AMT_PAYMENT_CURRENT_count_neq0', y='AMT_BALANCE_count_neq0')
for fea in ['AMT_PAYMENT_CURRENT_count_neq0/AMT_BALANCE_count_neq0']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6

t1 = t0[t0['AMT_PAYMENT_TOTAL_CURRENT'] != 0].copy()
t2 = count_f(t1, x='AMT_PAYMENT_TOTAL_CURRENT').rename(str, columns={'AMT_PAYMENT_TOTAL_CURRENT_count':'AMT_PAYMENT_TOTAL_CURRENT_count_neq0'})
t3 = t0[t0['AMT_BALANCE'] != 0].copy()
t4 = count_f(t3, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count':'AMT_BALANCE_count_neq0'})
t5 = add_feas_cont(ccd_base, features=[t2, t4])
t6 = div_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT_count_neq0', y='AMT_BALANCE_count_neq0')
for fea in ['AMT_PAYMENT_TOTAL_CURRENT_count_neq0/AMT_BALANCE_count_neq0']:
    print(fea)
    ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR', fea]])
del t1, t2, t3, t4, t5, t6

# 23. 支付金额-最低分期付款金额 stat 支付金额-最低分期付款金额/最低分期付款金额 支付金额-最低分期付款金额/支付总金额 stat
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_CURRENT', y='AMT_INST_MIN_REGULARITY')
t2 = div_f(t2, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY', y='AMT_INST_MIN_REGULARITY', add_1=True)
t2 = div_f(t2, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY', y='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t2 = t2.fillna(0)
t3 = stat_f(t2, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY')
t4 = stat_f(t2, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY/AMT_INST_MIN_REGULARITY')
t5 = stat_f(t2, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY/AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5

# 24. 支付金额-最低分期付款金额>0次数/支付金额!=0次数
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_CURRENT', y='AMT_INST_MIN_REGULARITY')
t3 = t2[t2['AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY'] > 0].copy()
t4 = count_f(t3, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY').rename(columns={'AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY_count':'AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY_bg0'})
t5 = t1[t1['AMT_PAYMENT_CURRENT'] != 0].copy()
t6 = count_f(t5, x='AMT_PAYMENT_CURRENT').rename(columns={'AMT_PAYMENT_CURRENT_count':'AMT_PAYMENT_CURRENT_neq0'})
t7 = add_feas_cont(base=ccd_base, features=[t4, t6])
t7 = div_f(t7, x='AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY_bg0', y='AMT_PAYMENT_CURRENT_neq0')
t7 = t7.fillna(1)
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_PAYMENT_CURRENT_sub_AMT_INST_MIN_REGULARITY_bg0/AMT_PAYMENT_CURRENT_neq0']])
del t1, t2, t3, t4, t5, t6, t7

# 25. 总支付金额-最低分期付款金额 stat 总支付金额-最低分期付款金额/最低分期付款金额 总支付金额-最低分期付款金额/总支付金额 stat
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_INST_MIN_REGULARITY')
t2 = div_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY', y='AMT_INST_MIN_REGULARITY', add_1=True)
t2 = div_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY', y='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t2 = t2.fillna(0)
t3 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY')
t4 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY/AMT_INST_MIN_REGULARITY')
t5 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY/AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
ccd_fe = add_feas(ccd_fe, t5)
del t1, t2, t3, t4, t5

# 26. 总支付金额-最低分期付款金额<0次数/总支付金额!=0次数
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_INST_MIN_REGULARITY')
t3 = t2[t2['AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY'] < 0].copy()
t4 = count_f(t3, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY').rename(columns={'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY_count':'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY_sm0'})
t5 = t1[t1['AMT_PAYMENT_TOTAL_CURRENT'] != 0].copy()
t6 = count_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT').rename(columns={'AMT_PAYMENT_TOTAL_CURRENT_count':'AMT_PAYMENT_TOTAL_CURRENT_neq0'})
t7 = add_feas_cont(base=ccd_base, features=[t4, t6])
t7 = div_f(t7, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY_sm0', y='AMT_PAYMENT_TOTAL_CURRENT_neq0')
t7 = t7.fillna(0)
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY_sm0/AMT_PAYMENT_TOTAL_CURRENT_neq0']])
del t1, t2, t3, t4, t5, t6, t7

# 27. 总支付金额-支付金额/总支付金额
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_PAYMENT_CURRENT')
t2 = div_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT', y='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t3 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT/AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 28. 总支付金额-支付金额<0次数/总支付金额！=0次数
t1 = t0[['SK_ID_CURR', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']].copy()
t2 = substr(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_PAYMENT_CURRENT')
t3 = t2[t2['AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT'] < 0].copy()
t4 = count_f(t3, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT').rename(columns={'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT_count':'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT_sm0'})
t5 = t1[t1['AMT_PAYMENT_TOTAL_CURRENT'] != 0].copy()
t6 = count_f(t5, x='AMT_PAYMENT_TOTAL_CURRENT').rename(columns={'AMT_PAYMENT_TOTAL_CURRENT_count':'AMT_PAYMENT_TOTAL_CURRENT_neq0'})
t7 = add_feas_cont(base=ccd_base, features=[t4, t6])
t7 = div_f(t7, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT_sm0', y='AMT_PAYMENT_TOTAL_CURRENT_neq0')
t7 = t7.fillna(1)
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT_sm0/AMT_PAYMENT_TOTAL_CURRENT_neq0']])
del t1, t2, t3, t4, t5, t6, t7

# 29. 应收本金 应收金额 应收总金额 stat
t1 = t0[['SK_ID_CURR', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']].copy()
t2 = stat_f(t1, x='AMT_RECEIVABLE_PRINCIPAL')
t3 = stat_f(t1, x='AMT_RECIVABLE')
t4 = stat_f(t1, x='AMT_TOTAL_RECEIVABLE')
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4

# 30. 第一笔应收总金额-第一笔应收本金/第一笔应收总金额  应收总金额-应收本金/应收总金额 stat
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']].copy()
t2 = first_last_f(t1, x='AMT_TOTAL_RECEIVABLE')[['SK_ID_CURR', 'AMT_TOTAL_RECEIVABLE_ft']]
t3 = first_last_f(t1, x='AMT_RECEIVABLE_PRINCIPAL')[['SK_ID_CURR', 'AMT_RECEIVABLE_PRINCIPAL_ft']]
t4 = add_feas_cont(base=ccd_base, features=[t2, t3])
t4 = substr(data=t4, x='AMT_TOTAL_RECEIVABLE_ft', y='AMT_RECEIVABLE_PRINCIPAL_ft')
t4 = div_f(data=t4, x='AMT_TOTAL_RECEIVABLE_ft_sub_AMT_RECEIVABLE_PRINCIPAL_ft', y='AMT_TOTAL_RECEIVABLE_ft', add_1=True)
ccd_fe = add_feas(ccd_fe, t4[['SK_ID_CURR', 'AMT_TOTAL_RECEIVABLE_ft_sub_AMT_RECEIVABLE_PRINCIPAL_ft/AMT_TOTAL_RECEIVABLE_ft']])
del t1, t2, t3, t4

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']].copy()
t2 = substr(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_RECEIVABLE_PRINCIPAL')
t2 = div_f(t2, x='AMT_TOTAL_RECEIVABLE_sub_AMT_RECEIVABLE_PRINCIPAL', y='AMT_TOTAL_RECEIVABLE')
t2 = t2.fillna(0)
t3 = stat_f(t2, x='AMT_TOTAL_RECEIVABLE_sub_AMT_RECEIVABLE_PRINCIPAL/AMT_TOTAL_RECEIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 31. 应收本金（应收金额）（应收总金额）-最低分期（支付金额）/应收本金（应收金额）（应收总金额）-余额
def sub_div_f(data, x, y, z, add_1=False, fillna_=-999):
    t1 = data.copy()
    nf = x + '_sub_' + y + '_div_' + z
    print(nf)

    if add_1:
        t1[nf] = (t1[x] - t1[y]) / (t1[z] + 1)
    else:
        t1[nf] = (t1[x] - t1[y]) / (t1[z])

    if fillna_ > -999:
        t1 = t1.fillna(fillna_)

    return t1

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
         'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
         'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
         'AMT_RECIVABLE',
         'AMT_TOTAL_RECEIVABLE']].copy()
t2 = sub_div_f(t1, x='AMT_RECEIVABLE_PRINCIPAL', y='AMT_INST_MIN_REGULARITY', z='AMT_RECEIVABLE_PRINCIPAL', add_1=True)
t3 = stat_f(t2, x='AMT_RECEIVABLE_PRINCIPAL_sub_AMT_INST_MIN_REGULARITY_div_AMT_RECEIVABLE_PRINCIPAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_RECEIVABLE_PRINCIPAL', y='AMT_PAYMENT_CURRENT', z='AMT_RECEIVABLE_PRINCIPAL', add_1=True)
t3 = stat_f(t2, x='AMT_RECEIVABLE_PRINCIPAL_sub_AMT_PAYMENT_CURRENT_div_AMT_RECEIVABLE_PRINCIPAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_RECIVABLE', y='AMT_INST_MIN_REGULARITY', z='AMT_RECIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_RECIVABLE_sub_AMT_INST_MIN_REGULARITY_div_AMT_RECIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_RECIVABLE', y='AMT_PAYMENT_CURRENT', z='AMT_RECIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_RECIVABLE_sub_AMT_PAYMENT_CURRENT_div_AMT_RECIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_INST_MIN_REGULARITY', z='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t3 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_INST_MIN_REGULARITY_div_AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_PAYMENT_CURRENT', z='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t3 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_PAYMENT_CURRENT_div_AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_PAYMENT_TOTAL_CURRENT', y='AMT_BALANCE', z='AMT_PAYMENT_TOTAL_CURRENT', add_1=True)
t3 = stat_f(t2, x='AMT_PAYMENT_TOTAL_CURRENT_sub_AMT_BALANCE_div_AMT_PAYMENT_TOTAL_CURRENT')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3


# 32. 应收本金（应收金额）（应收总金额）-总支用/应收本金（应收金额）（应收总金额）-余额
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
         'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
         'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
         'AMT_RECIVABLE',
         'AMT_TOTAL_RECEIVABLE']].copy()
t2 = sub_div_f(t1, x='AMT_RECEIVABLE_PRINCIPAL', y='AMT_PAYMENT_TOTAL_CURRENT', z='AMT_RECEIVABLE_PRINCIPAL', add_1=True)
t3 = stat_f(t2, x='AMT_RECEIVABLE_PRINCIPAL_sub_AMT_PAYMENT_TOTAL_CURRENT_div_AMT_RECEIVABLE_PRINCIPAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_RECIVABLE', y='AMT_PAYMENT_TOTAL_CURRENT', z='AMT_RECIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_RECIVABLE_sub_AMT_PAYMENT_TOTAL_CURRENT_div_AMT_RECIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_PAYMENT_TOTAL_CURRENT', z='AMT_TOTAL_RECEIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_TOTAL_RECEIVABLE_sub_AMT_PAYMENT_TOTAL_CURRENT_div_AMT_TOTAL_RECEIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_PAYMENT_TOTAL_CURRENT', z='AMT_BALANCE', add_1=True)
t3 = stat_f(t2, x='AMT_TOTAL_RECEIVABLE_sub_AMT_PAYMENT_TOTAL_CURRENT_div_AMT_BALANCE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

# 33. 趋势 是否递减 todo!!
# 34. 应收本金（应收金额）（应收总金额）-余额/应收本金（应收金额）（应收总金额）
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
         'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
         'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
         'AMT_RECIVABLE',
         'AMT_TOTAL_RECEIVABLE']].copy()
t2 = sub_div_f(t1, x='AMT_RECEIVABLE_PRINCIPAL', y='AMT_BALANCE', z='AMT_RECEIVABLE_PRINCIPAL', add_1=True)
t3 = stat_f(t2, x='AMT_RECEIVABLE_PRINCIPAL_sub_AMT_BALANCE_div_AMT_RECEIVABLE_PRINCIPAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_RECIVABLE', y='AMT_BALANCE', z='AMT_RECIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_RECIVABLE_sub_AMT_BALANCE_div_AMT_RECIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_BALANCE', z='AMT_TOTAL_RECEIVABLE', add_1=True)
t3 = stat_f(t2, x='AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_div_AMT_TOTAL_RECEIVABLE')
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 35. 授信额度-应收本金（应收金额）（应收总金额）/授信额度
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',
         'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
         'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
         'AMT_RECIVABLE',
         'AMT_TOTAL_RECEIVABLE']].copy()
t2 = sub_div_f(t1, x='AMT_CREDIT_LIMIT_ACTUAL', y='AMT_INST_MIN_REGULARITY', z='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t3 = stat_f(t2, x='AMT_CREDIT_LIMIT_ACTUAL_sub_AMT_INST_MIN_REGULARITY_div_AMT_CREDIT_LIMIT_ACTUAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_CREDIT_LIMIT_ACTUAL', y='AMT_RECIVABLE', z='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t3 = stat_f(t2, x='AMT_CREDIT_LIMIT_ACTUAL_sub_AMT_RECIVABLE_div_AMT_CREDIT_LIMIT_ACTUAL')
ccd_fe = add_feas(ccd_fe, t3)
del t2, t3

t2 = sub_div_f(t1, x='AMT_CREDIT_LIMIT_ACTUAL', y='AMT_TOTAL_RECEIVABLE', z='AMT_CREDIT_LIMIT_ACTUAL', add_1=True)
t3 = stat_f(t2, x='AMT_CREDIT_LIMIT_ACTUAL_sub_AMT_TOTAL_RECEIVABLE_div_AMT_CREDIT_LIMIT_ACTUAL')
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 36. 应收总金额-余额>0次数/应收总金额！=0次数    应收总金额-余额<0次数/应收总金额！=0次数
t1 = t0[['SK_ID_CURR', 'AMT_BALANCE', 'AMT_TOTAL_RECEIVABLE']].copy()
t2 = substr(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_BALANCE')
t3 = t2[t2['AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE'] > 0].copy()
t4 = count_f(t3, x='AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE').rename(columns={'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_count':'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_bg0'})
t4 = t4.fillna(0)
t5 = t1[t1['AMT_TOTAL_RECEIVABLE'] != 0].copy()
t6 = count_f(t5, x='AMT_TOTAL_RECEIVABLE').rename(columns={'AMT_TOTAL_RECEIVABLE_count':'AMT_TOTAL_RECEIVABLE_neq0'})
t7 = add_feas_cont(base=ccd_base, features=[t4, t6])
t7 = div_f(t7, x='AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_bg0', y='AMT_TOTAL_RECEIVABLE_neq0')
t7 = t7.fillna(0)
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_bg0/AMT_TOTAL_RECEIVABLE_neq0']])
del t1, t2, t3, t4, t5, t6, t7

t1 = t0[['SK_ID_CURR', 'AMT_BALANCE', 'AMT_TOTAL_RECEIVABLE']].copy()
t2 = substr(t1, x='AMT_TOTAL_RECEIVABLE', y='AMT_BALANCE')
t3 = t2[t2['AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE'] < 0].copy()
t4 = count_f(t3, x='AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE').rename(columns={'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_count':'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_sm0'})
t4 = t4.fillna(0)
t5 = t1[t1['AMT_TOTAL_RECEIVABLE'] != 0].copy()
t6 = count_f(t5, x='AMT_TOTAL_RECEIVABLE').rename(columns={'AMT_TOTAL_RECEIVABLE_count':'AMT_TOTAL_RECEIVABLE_neq0'})
t7 = add_feas_cont(base=ccd_base, features=[t4, t6])
t7 = div_f(t7, x='AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_sm0', y='AMT_TOTAL_RECEIVABLE_neq0')
t7 = t7.fillna(1)
ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR', 'AMT_TOTAL_RECEIVABLE_sub_AMT_BALANCE_sm0/AMT_TOTAL_RECEIVABLE_neq0']])
del t1, t2, t3, t4, t5, t6, t7

# 37. 支用：stat
t1 = t0[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t2 = stat_f(t1, x='CNT_DRAWINGS_ATM_CURRENT')
t3 = stat_f(t1, x='CNT_DRAWINGS_POS_CURRENT')
t4 = stat_f(t1, x='CNT_DRAWINGS_OTHER_CURRENT')
t5 = stat_f(t1, x='CNT_DRAWINGS_CURRENT')
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
ccd_fe = add_feas(ccd_fe, t5)
del t2, t3, t4, t5

# 38. 支用金额sum/支用次数sum
t2 = t1.groupby('SK_ID_CURR')[['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT']].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_sum', 'CNT_DRAWINGS_CURRENT':'CNT_DRAWINGS_CURRENT_sum'})
t3 = div_f(t2, x='AMT_DRAWINGS_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']])
del t2, t3

# 39. 单笔支用最高金额 每条记录支用金额/次数 max  /首次/末次授信额度  /单笔平均支用金额  /
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t1['CNT_DRAWINGS_CURRENT'] = t1['CNT_DRAWINGS_CURRENT'] + 0.1
t2 = div_f(t1, x='AMT_DRAWINGS_CURRENT', y='CNT_DRAWINGS_CURRENT')
t2 = t2.fillna(0)
t3 = t2.groupby('SK_ID_CURR')[['AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT']].max().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max'})

t4 = first_last_f(t1, x='AMT_CREDIT_LIMIT_ACTUAL')

t5 = t1.groupby('SK_ID_CURR')[['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT']].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_sum', 'CNT_DRAWINGS_CURRENT':'CNT_DRAWINGS_CURRENT_sum'})
t6 = div_f(t5, x='AMT_DRAWINGS_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t6 = t6.fillna(0)

t7 = add_feas_cont(ccd_base, features=[t3, t4, t6[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']]])

t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max', y='AMT_CREDIT_LIMIT_ACTUAL_ft', add_1=True)
t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max', y='AMT_CREDIT_LIMIT_ACTUAL_lt', add_1=True)
t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max', y='AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum', add_1=True)

ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max/AMT_CREDIT_LIMIT_ACTUAL_ft',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max/AMT_CREDIT_LIMIT_ACTUAL_lt',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_max/AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']])
del t1, t2, t3, t4, t5, t6, t7

# 40. 单笔支用最低金额 每条记录支用金额/次数 min  /授信额度  /单笔平均支用金额  /
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t1['CNT_DRAWINGS_CURRENT'] = t1['CNT_DRAWINGS_CURRENT'] + 0.1
t2 = div_f(t1, x='AMT_DRAWINGS_CURRENT', y='CNT_DRAWINGS_CURRENT')
t2 = t2.fillna(0)
t3 = t2.groupby('SK_ID_CURR')[['AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT']].min().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min'})

t4 = first_last_f(t1, x='AMT_CREDIT_LIMIT_ACTUAL')

t5 = t1.groupby('SK_ID_CURR')[['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT']].sum().reset_index().rename(str, columns={'AMT_DRAWINGS_CURRENT': 'AMT_DRAWINGS_CURRENT_sum', 'CNT_DRAWINGS_CURRENT':'CNT_DRAWINGS_CURRENT_sum'})
t6 = div_f(t5, x='AMT_DRAWINGS_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t6 = t6.fillna(0)

t7 = add_feas_cont(ccd_base, features=[t3, t4, t6[['SK_ID_CURR', 'AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']]])

t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min', y='AMT_CREDIT_LIMIT_ACTUAL_ft', add_1=True)
t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min', y='AMT_CREDIT_LIMIT_ACTUAL_lt', add_1=True)
t7 = div_f(t7, x='AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min', y='AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum', add_1=True)

ccd_fe = add_feas(ccd_fe, t7[['SK_ID_CURR',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min/AMT_CREDIT_LIMIT_ACTUAL_ft',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min/AMT_CREDIT_LIMIT_ACTUAL_lt',
                              'AMT_DRAWINGS_CURRENT/CNT_DRAWINGS_CURRENT_min/AMT_DRAWINGS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']])
del t1, t2, t3, t4, t5, t6, t7

# 41. 每笔支用占比 stat   sum->支用占比
t1 = t0[['SK_ID_CURR', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t1 = div_f(t1, x='CNT_DRAWINGS_ATM_CURRENT', y='CNT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(t1, x='CNT_DRAWINGS_POS_CURRENT', y='CNT_DRAWINGS_CURRENT', add_1=True)
t1 = div_f(t1, x='CNT_DRAWINGS_OTHER_CURRENT', y='CNT_DRAWINGS_CURRENT', add_1=True)
t2 = stat_f(t1, x='CNT_DRAWINGS_ATM_CURRENT/CNT_DRAWINGS_CURRENT')
t3 = stat_f(t1, x='CNT_DRAWINGS_POS_CURRENT/CNT_DRAWINGS_CURRENT')
t4 = stat_f(t1, x='CNT_DRAWINGS_OTHER_CURRENT/CNT_DRAWINGS_CURRENT')
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
ccd_fe = add_feas(ccd_fe, t4)
del t1, t2, t3, t4

t1 = t0[['SK_ID_CURR', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR').sum().reset_index().rename(str, columns={'CNT_DRAWINGS_ATM_CURRENT':'CNT_DRAWINGS_ATM_CURRENT_sum', 'CNT_DRAWINGS_POS_CURRENT':'CNT_DRAWINGS_POS_CURRENT_sum', 'CNT_DRAWINGS_OTHER_CURRENT':'CNT_DRAWINGS_OTHER_CURRENT_sum', 'CNT_DRAWINGS_CURRENT':'CNT_DRAWINGS_CURRENT_sum'})
t2 = div_f(t2, x='CNT_DRAWINGS_ATM_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t2 = div_f(t2, x='CNT_DRAWINGS_POS_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t2 = div_f(t2, x='CNT_DRAWINGS_OTHER_CURRENT_sum', y='CNT_DRAWINGS_CURRENT_sum')
t2 = t2.fillna(0)
ccd_fe = add_feas(ccd_fe, t2[['SK_ID_CURR',
                              'CNT_DRAWINGS_ATM_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum',
                              'CNT_DRAWINGS_POS_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum',
                              'CNT_DRAWINGS_OTHER_CURRENT_sum/CNT_DRAWINGS_CURRENT_sum']])
del t1, t2

# 42. 支用最多（次数）方式 oh encoding   占比
t1 = t0[['SK_ID_CURR', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR').sum().reset_index().rename(str, columns={'CNT_DRAWINGS_ATM_CURRENT':'ATM_cnt', 'CNT_DRAWINGS_POS_CURRENT':'POS_cnt', 'CNT_DRAWINGS_OTHER_CURRENT':'OTHER_cnt'})
t2['CNT_DRAWINGS_main'] = t2[['ATM_cnt', 'POS_cnt', 'OTHER_cnt']].idxmax(axis=1)
t3 = pd.get_dummies(data=t2, columns=['CNT_DRAWINGS_main'], dummy_na=False)
t3 = div_f(t3, x='ATM_cnt', y='CNT_DRAWINGS_CURRENT')
t3 = div_f(t3, x='POS_cnt', y='CNT_DRAWINGS_CURRENT')
t3 = div_f(t3, x='OTHER_cnt', y='CNT_DRAWINGS_CURRENT')
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t3[['SK_ID_CURR', 'CNT_DRAWINGS_main_ATM_cnt', 'CNT_DRAWINGS_main_OTHER_cnt', 'CNT_DRAWINGS_main_POS_cnt',
                              'ATM_cnt/CNT_DRAWINGS_CURRENT', 'POS_cnt/CNT_DRAWINGS_CURRENT', 'OTHER_cnt/CNT_DRAWINGS_CURRENT']])
del t1, t2, t3

# 43. 支用方式总数， 是否三种都用，是否只用一种
t1 = t0[['SK_ID_CURR', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR').sum().reset_index().rename(str, columns={'CNT_DRAWINGS_ATM_CURRENT':'CNT_DRAWINGS_ATM_CURRENT_sum', 'CNT_DRAWINGS_POS_CURRENT':'CNT_DRAWINGS_POS_CURRENT_sum', 'CNT_DRAWINGS_OTHER_CURRENT':'CNT_DRAWINGS_OTHER_CURRENT_sum', 'CNT_DRAWINGS_CURRENT':'CNT_DRAWINGS_CURRENT_sum'})
t2['AMT_flag'] = np.where((t2['CNT_DRAWINGS_ATM_CURRENT_sum'] > 0), 1, 0)
t2['POS_flag'] = np.where((t2['CNT_DRAWINGS_POS_CURRENT_sum'] > 0), 1, 0)
t2['OTHER_flag'] = np.where((t2['CNT_DRAWINGS_OTHER_CURRENT_sum'] > 0), 1, 0)
t2['use_number'] = t2['AMT_flag'] + t2['POS_flag'] + t2['OTHER_flag']
t2['use_all_flag'] = np.where(t2['use_number'] == 3, 1, 0)
t2['use_one_flag'] = np.where(t2['use_number'] == 1, 1, 0)
ccd_fe = add_feas(ccd_fe, t2[['SK_ID_CURR', 'use_number', 'use_all_flag', 'use_one_flag']])
del t1, t2

# 45. 支用次数最大/平均只有次数
t1 = t0[['SK_ID_CURR', 'CNT_DRAWINGS_CURRENT']].copy()
t2 = t1.groupby('SK_ID_CURR')['CNT_DRAWINGS_CURRENT'].agg(['max', 'mean']).reset_index().rename(str, columns={'max':'CNT_DRAWINGS_CURRENT_max', 'mean':'CNT_DRAWINGS_CURRENT_mean'})
t2 = div_f(t2, x='CNT_DRAWINGS_CURRENT_max', y='CNT_DRAWINGS_CURRENT_mean', add_1=True)
ccd_fe = add_feas(ccd_fe, t2[['SK_ID_CURR', 'CNT_DRAWINGS_CURRENT_max/CNT_DRAWINGS_CURRENT_mean']])
del t1, t2

# 46. 总支用最长连续支用月份数 /总期数 /最早期数  active
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_DRAWINGS_CURRENT']].copy()
t1['dra_flag'] = np.where((t1['CNT_DRAWINGS_CURRENT'] > 0), 1, 0)
t1['dra_flag_diff'] = t1.groupby(by='SK_ID_CURR')['dra_flag'].diff()
t2 = t1.groupby(by='SK_ID_CURR')['dra_flag_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'CNT_DRAWINGS_CURRENT_diff_t'})
t2['continue_months_DRAWINGS'] = np.ceil(t2['CNT_DRAWINGS_CURRENT_diff_t'] / 2)
del t2['CNT_DRAWINGS_CURRENT_diff_t']
t3 = count_f(t1, x='MONTHS_BALANCE')
t4 = first_last_f(t1, x='MONTHS_BALANCE')[['SK_ID_CURR', 'MONTHS_BALANCE_ft']]
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = add_feas_cont(base=ccd_base, features=[t2, t3, t4])
t5 = div_f(t5, x='continue_months_DRAWINGS', y='MONTHS_BALANCE_count')
t5 = div_f(t5, x='continue_months_DRAWINGS', y='MONTHS_BALANCE_ft')
t5 = t5.fillna(0)
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', 'continue_months_DRAWINGS',
                              'continue_months_DRAWINGS/MONTHS_BALANCE_count',
                              'continue_months_DRAWINGS/MONTHS_BALANCE_ft']])
del t1, t2, t3, t4, t5

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_DRAWINGS_CURRENT', 'ccd_status']].copy()
t1['dra_flag'] = np.where((t1['CNT_DRAWINGS_CURRENT'] > 0), 1, 0)
t1['dra_flag_diff'] = t1.groupby(by='SK_ID_CURR')['dra_flag'].diff()
t2 = t1.groupby(by='SK_ID_CURR')['dra_flag_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'CNT_DRAWINGS_CURRENT_diff_t'})
t2['continue_months_DRAWINGS_act'] = np.ceil(t2['CNT_DRAWINGS_CURRENT_diff_t'] / 2)
del t2['CNT_DRAWINGS_CURRENT_diff_t']

t3 = count_f(t1, x='MONTHS_BALANCE', active=True)
t4 = first_last_f(t1, x='MONTHS_BALANCE', active=True)[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']]
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = add_feas_cont(base=ccd_base, features=[t2, t3, t4])
t5 = div_f(t5, x='continue_months_DRAWINGS_act', y='MONTHS_BALANCE_count_active')
t5 = div_f(t5, x='continue_months_DRAWINGS_act', y='MONTHS_BALANCE_active_ft')
t5 = t5.fillna(0)
ccd_fe = add_feas(ccd_fe, t5[['SK_ID_CURR', 'continue_months_DRAWINGS_act',
                              'continue_months_DRAWINGS_act/MONTHS_BALANCE_count_active',
                              'continue_months_DRAWINGS_act/MONTHS_BALANCE_active_ft']])
del t1, t2, t3, t4, t5

# 47. 是否中间停止过支用 次数 /（第一次支用期数到最后一次支用期数）todo
# 48. NAME_CONTRACT_STATUS SK_DPD SK_DPD_DEF 参考bureau balance

# 1) SK_DPD SK_DPD_DEF stat   active
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = stat_f(t1, x='SK_DPD')
t3 = stat_f(t1, x='SK_DPD_DEF')
t2 = t2.fillna(0)
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = stat_f(t1, x='SK_DPD', active=True)
t3 = stat_f(t1, x='SK_DPD_DEF', active=True)
t2 = t2.fillna(0)
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t2)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 2) SK_DPD - SK_DPD_DEF stat    active
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t1 = substr(t1, x='SK_DPD', y='SK_DPD_DEF')
t2 = stat_f(t1, x='SK_DPD_sub_SK_DPD_DEF')
t2 = t2.fillna(0)
ccd_fe = add_feas(ccd_fe, t2)
del t1, t2

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t1 = substr(t1, x='SK_DPD', y='SK_DPD_DEF')
t2 = stat_f(t1, x='SK_DPD_sub_SK_DPD_DEF', active=True)
t2 = t2.fillna(0)
ccd_fe = add_feas(ccd_fe, t2)
del t1, t2

# 3) SK_ID_CURR  SK_DPD SK_DPD_DEF 逾期 flag active
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD'].max().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_max'})
t3 = add_feas(ccd_base, t2)
t3['SK_DPD_flag'] = np.where(t3['SK_DPD_max'] > 0, 1, 0)
del t3['SK_DPD_max']
ccd_fe = add_feas(ccd_fe, t3, on='SK_ID_CURR')
del t1, t2, t3

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD'].max().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_max_active'})
t3 = add_feas(ccd_base, t2)
t3['SK_DPD_active_flag'] = np.where(t3['SK_DPD_max_active'] > 0, 1, 0)
del t3['SK_DPD_max_active']
ccd_fe = add_feas(ccd_fe, t3, on='SK_ID_CURR')
del t1, t2, t3

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().reset_index().rename(str, columns={'SK_DPD_DEF': 'SK_DPD_DEF_max'})
t3 = add_feas(ccd_base, t2)
t3['SK_DPD_DEF_flag'] = np.where(t3['SK_DPD_DEF_max'] > 0, 1, 0)
del t3['SK_DPD_DEF_max']
ccd_fe = add_feas(ccd_fe, t3, on='SK_ID_CURR')
del t1, t2, t3

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().reset_index().rename(str, columns={'SK_DPD_DEF': 'SK_DPD_DEF_max_active'})
t3 = add_feas(ccd_base, t2)
t3['SK_DPD_DEF_active_flag'] = np.where(t3['SK_DPD_DEF_max_active'] > 0, 1, 0)
del t3['SK_DPD_DEF_max_active']
ccd_fe = add_feas(ccd_fe, t3, on='SK_ID_CURR')
del t1, t2, t3


# 4) SK_ID_CURR  SK_DPD SK_DPD_DEF 逾期次数（SK_DPD>0 count） /Months_balance个数 /最早期数/ 余额不为0期数 active
t1 = t0.loc[t0['SK_DPD'] > 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
t2 = count_f(t1, x='SK_DPD').rename(str, columns={'SK_DPD_count': 'SK_DPD_count_bg0'}).fillna(0)
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE')
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE')[['SK_ID_CURR', 'MONTHS_BALANCE_ft']]
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count': 'AMT_BALANCE_count_neq0'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='SK_DPD_count_bg0', y='MONTHS_BALANCE_count')
t6 = div_f(t6, x='SK_DPD_count_bg0', y='MONTHS_BALANCE_ft')
t6 = div_f(t6, x='SK_DPD_count_bg0', y='AMT_BALANCE_count_neq0')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR',
                              'SK_DPD_count_bg0/MONTHS_BALANCE_count',
                              'SK_DPD_count_bg0/MONTHS_BALANCE_ft',
                              'SK_DPD_count_bg0/AMT_BALANCE_count_neq0']])
del t1, t2, t3, t4, t5, t6

t1 = t0.loc[t0['SK_DPD_DEF'] > 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD_DEF']].copy()
t2 = count_f(t1, x='SK_DPD_DEF').rename(str, columns={'SK_DPD_DEF_count': 'SK_DPD_DEF_count_bg0'}).fillna(0)
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE')
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE')[['SK_ID_CURR', 'MONTHS_BALANCE_ft']]
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count': 'AMT_BALANCE_count_neq0'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0', y='MONTHS_BALANCE_count')
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0', y='MONTHS_BALANCE_ft')
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0', y='AMT_BALANCE_count_neq0')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR',
                              'SK_DPD_DEF_count_bg0/MONTHS_BALANCE_count',
                              'SK_DPD_DEF_count_bg0/MONTHS_BALANCE_ft',
                              'SK_DPD_DEF_count_bg0/AMT_BALANCE_count_neq0']])
del t1, t2, t3, t4, t5, t6

t1 = t0.loc[t0['SK_DPD'] > 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'ccd_status']].copy()
t2 = count_f(t1, x='SK_DPD', active=True).rename(str, columns={'SK_DPD_count_active': 'SK_DPD_count_bg0_act'}).fillna(0)
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE', active=True)
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE', active=True)[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']]
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE', active=True).rename(str, columns={'AMT_BALANCE_count_active': 'AMT_BALANCE_count_neq0_act'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='SK_DPD_count_bg0_act', y='MONTHS_BALANCE_count_active')
t6 = div_f(t6, x='SK_DPD_count_bg0_act', y='MONTHS_BALANCE_active_ft')
t6 = div_f(t6, x='SK_DPD_count_bg0_act', y='AMT_BALANCE_count_neq0_act')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR',
                              'SK_DPD_count_bg0_act/MONTHS_BALANCE_count_active',
                              'SK_DPD_count_bg0_act/MONTHS_BALANCE_active_ft',
                              'SK_DPD_count_bg0_act/AMT_BALANCE_count_neq0_act']])
del t1, t2, t3, t4, t5, t6

t1 = t0.loc[t0['SK_DPD_DEF'] > 0, ['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD_DEF', 'ccd_status']].copy()
t2 = count_f(t1, x='SK_DPD_DEF', active=True).rename(str, columns={'SK_DPD_DEF_count_active': 'SK_DPD_DEF_count_bg0_act'}).fillna(0)
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE', active=True)
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE', active=True)[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']]
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE', active=True).rename(str, columns={'AMT_BALANCE_count_active': 'AMT_BALANCE_count_neq0_act'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0_act', y='MONTHS_BALANCE_count_active')
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0_act', y='MONTHS_BALANCE_active_ft')
t6 = div_f(t6, x='SK_DPD_DEF_count_bg0_act', y='AMT_BALANCE_count_neq0_act')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR',
                              'SK_DPD_DEF_count_bg0_act/MONTHS_BALANCE_count_active',
                              'SK_DPD_DEF_count_bg0_act/MONTHS_BALANCE_active_ft',
                              'SK_DPD_DEF_count_bg0_act/AMT_BALANCE_count_neq0_act']])
del t1, t2, t3, t4, t5, t6

# 5) SK_DPD 逾期次数（连续逾期算为1次） /Months_balance个数 /最早期数/ 余额不为0期数 active
t1 = t0[['SK_ID_CURR', 'SK_DPD']].copy()
t1['SK_DPD_t'] = np.where(t1['SK_DPD'] > 0, 1, 0)
t1['SK_DPD_diff'] = t1.groupby(by='SK_ID_CURR')['SK_DPD_t'].diff()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'SK_DPD_diff_t'})
t2['continue_DPD_count'] = np.ceil(t2['SK_DPD_diff_t'] / 2)
del t2['SK_DPD_diff_t']
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE')
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE')[['SK_ID_CURR', 'MONTHS_BALANCE_ft']]
t4['MONTHS_BALANCE_ft'] = -t4['MONTHS_BALANCE_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE').rename(str, columns={'AMT_BALANCE_count': 'AMT_BALANCE_count_neq0'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='continue_DPD_count', y='MONTHS_BALANCE_count')
t6 = div_f(t6, x='continue_DPD_count', y='MONTHS_BALANCE_ft')
t6 = div_f(t6, x='continue_DPD_count', y='AMT_BALANCE_count_neq0')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR', 'continue_DPD_count',
                              'continue_DPD_count/MONTHS_BALANCE_count',
                              'continue_DPD_count/MONTHS_BALANCE_ft',
                              'continue_DPD_count/AMT_BALANCE_count_neq0']])
del t1, t2, t3, t4, t5, t6

t1 = t0.loc[t0['ccd_status'] == 0, ['SK_ID_CURR', 'SK_DPD']].copy()
t1['SK_DPD_t'] = np.where(t1['SK_DPD'] > 0, 1, 0)
t1['SK_DPD_diff'] = t1.groupby(by='SK_ID_CURR')['SK_DPD_t'].diff()
t2 = t1.groupby(by='SK_ID_CURR')['SK_DPD_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'SK_DPD_diff_t'})
t2['continue_DPD_count_act'] = np.ceil(t2['SK_DPD_diff_t'] / 2)
del t2['SK_DPD_diff_t']
t3 = t0.copy()
t3 = count_f(t3, x='MONTHS_BALANCE', active=True)
t4 = t0.copy()
t4 = first_last_f(t4, x='MONTHS_BALANCE', active=True)[['SK_ID_CURR', 'MONTHS_BALANCE_active_ft']]
t4['MONTHS_BALANCE_active_ft'] = -t4['MONTHS_BALANCE_active_ft']
t5 = t0[t0['AMT_BALANCE'] != 0].copy()
t5 = count_f(t5, x='AMT_BALANCE', active=True).rename(str, columns={'AMT_BALANCE_count_active': 'AMT_BALANCE_count_active_neq0'}).fillna(0)
t6 = add_feas_cont(ccd_base, features=[t2, t3, t4, t5])
t6 = div_f(t6, x='continue_DPD_count_act', y='MONTHS_BALANCE_count_active')
t6 = div_f(t6, x='continue_DPD_count_act', y='MONTHS_BALANCE_active_ft')
t6 = div_f(t6, x='continue_DPD_count_act', y='AMT_BALANCE_count_active_neq0')
t6 = t6.fillna(0)
ccd_fe = add_feas(ccd_fe, t6[['SK_ID_CURR', 'continue_DPD_count_act',
                              'continue_DPD_count_act/MONTHS_BALANCE_count_active',
                              'continue_DPD_count_act/MONTHS_BALANCE_active_ft',
                              'continue_DPD_count_act/AMT_BALANCE_count_active_neq0']])
del t1, t2, t3, t4, t5, t6

# 最近N(-1、-3、-6、-12、-24、-36、-60、-120)期未出现逾期的次数/占比(最近N期)
# 最近N(-1、-3、-6、-12、-24、-36、-60、全部)期出现过逾期>=X 的次数/占比
# 最近N(-1、-3、-6、-12、-24、-36、-60、全部)期最大逾期期数
# 最近N(-1、-3、-6、-12、-24、-36、-60、全部)期是否出现过逾期
# 当期（最近那期）的逾期状况
# 当期（最近那期）的逾期状况是否逾期
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
t1['SK_DPD_0_f'] = np.where(t1['SK_DPD'] == 0, 1, 0)
for N in [-1, -3, -6, -12, -24, -36, -60, -120]:
    nf_dpd = 'DPD_0_cnt_include_'+str(-N)
    nf_tol = 'total_cnt_include_'+str(-N)
    print("最近 %d 期" % -N)
    t2 = t1[t1['MONTHS_BALANCE'] >= N].copy()
    t3 = t2.groupby(by='SK_ID_CURR')['SK_DPD_0_f'].agg(['sum', 'count']).reset_index().rename(str, columns={'sum':nf_dpd, 'count':nf_tol})
    t3 = div_f(t3, x=nf_dpd, y=nf_tol)
    t3 = t3.fillna(0)
    t4 = add_feas(ccd_base, t3[['SK_ID_CURR', nf_dpd+'/'+nf_tol]])
    t4 = t4.fillna(1)
    ccd_fe = add_feas(ccd_fe, t4)
    del t2, t3, t4

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
for X in [1, 7, 14, 30, 60, 90, 120, 240, 360, 480, 600]:
    nf_X = 'SK_DPD_bg_' + str(X) + '_f'
    print(nf_X)
    t1[nf_X] = np.where(t1['SK_DPD'] >= X, 1, 0)
    for N in [-1, -3, -6, -12, -24, -36, -60, -120]:
        nf_dpd = 'DPD_cnt_include_'+str(-N)+'_bg_'+str(X)
        nf_tol = 'total_cnt_include_'+str(-N)+'_bg_'+str(X)
        print("最近 %d 期" % -N)
        t2 = t1[t1['MONTHS_BALANCE'] >= N].copy()
        t3 = t2.groupby(by='SK_ID_CURR')[nf_X].agg(['sum', 'count']).reset_index().rename(str, columns={'sum':nf_dpd, 'count':nf_tol})
        t3 = div_f(t3, x=nf_dpd, y=nf_tol)
        t3 = t3.fillna(0)
        t4 = add_feas(ccd_base, t3[['SK_ID_CURR', nf_dpd+'/'+nf_tol]])
        t4 = t4.fillna(0)
        ccd_fe = add_feas(ccd_fe, t4)
        del t2, t3, t4

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
for N in [-1, -3, -6, -12, -24, -36, -60, -120]:
    nf_dpd = 'DPD_max_dpd_include_'+str(-N)
    print(nf_dpd)
    print("最近 %d 期" % -N)
    t2 = t1[t1['MONTHS_BALANCE'] >= N].copy()
    t3 = t2.groupby(by='SK_ID_CURR')['SK_DPD'].agg(['max']).reset_index().rename(str, columns={'max':nf_dpd})
    t4 = add_feas(ccd_base, t3[['SK_ID_CURR', nf_dpd]])
    t4 = t4.fillna(0)
    ccd_fe = add_feas(ccd_fe, t4)
    del t2, t3, t4

t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
for N in [-1, -3, -6, -12, -24, -36, -60, -120]:
    nf_dpd = 'DPD_max_dpd_include_'+str(-N)
    nf_dpd_flag = 'DPD_dpd_include_'+str(-N)+'_flag'
    print(nf_dpd_flag)
    print("最近 %d 期" % -N)
    t2 = t1[t1['MONTHS_BALANCE'] >= N].copy()
    t3 = t2.groupby(by='SK_ID_CURR')['SK_DPD'].agg(['max']).reset_index().rename(str, columns={'max':nf_dpd})
    t3[nf_dpd_flag] = np.where(t3[nf_dpd] > 0, 1, 0)
    t4 = add_feas(ccd_base, t3[['SK_ID_CURR', nf_dpd_flag]])
    t4 = t4.fillna(0)
    ccd_fe = add_feas(ccd_fe, t4)
    del t2, t3, t4

# 当期（最近那期）的逾期状况
# 当期（最近那期）的逾期状况是否逾期
t1 = t0[['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD']].copy()
t2 = first_last_f(t1, x='SK_DPD')[['SK_ID_CURR', 'SK_DPD_lt']].rename(str, columns={'SK_DPD_lt':'SK_DPD_latest'})
t2['SK_DPD_latest_flag'] = np.where(t2['SK_DPD_latest'] > 0, 1, 0)
t3 = add_feas(ccd_base, t2)
t3 = t3.fillna(0)
ccd_fe = add_feas(ccd_fe, t3)
del t1, t2, t3

# 49. 错位 +-*/ todo

# to_hdf
ccd_fe = ccd_fe.fillna(-999)
print("pos_fe : ", ccd_fe.shape[1] - 1)
ccd_fe.to_hdf('Data_/Pos/credict_card_balance.hdf', 'credict_card_balance', mode='w', format='table')



