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

sns.set(style="ticks")


def plot_categorical(data, col, size=[8, 4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize=size)
    sns.barplot(x=plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle != 0:
        plt.xticks(rotation=xlabel_angle)  # 轴坐标刻度-角度
    plt.show()
def plot_numerical(data, col, size=[8, 4], bins=50):
    """ use this for ploting the distribution of numercial features """
    plt.figure(figsize=size)
    plt.title("Distribution of %s" % col)
    sns.distplot(data[col].dropna(), kde=True, bins=bins)  # 排除缺失值
    plt.show()
def iv_table(data, y, x, point: str = "NULL", labels='NULL', right=False):
    df = data[[y, x]]

    # 连续性特征处理
    if point != 'NULL':
        x_bin = x + '_bin'
        df[x_bin] = pd.cut(df[x], point, right=right, labels=labels)
        df[x_bin] = df[x_bin].cat.add_categories(['Missing', 'SUM'])
    else:
        x_bin = x + '_bin'
        df[x_bin] = df[x].astype('category')
        df[x_bin] = df[x_bin].cat.add_categories(['Missing', 'SUM'])

    # 表格计算
    iv_tb = df.groupby([x_bin])[y].agg(['count', 'sum']).reset_index().rename(
        columns={'count': 'Total', 'sum': 'Bad_cnt'})
    iv_tb['Good_cnt'] = iv_tb['Total'] - iv_tb['Bad_cnt']

    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Total'] = len(df[x])
    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Bad_cnt'] = (df[y] == 1).sum()
    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Good_cnt'] = (df[y] == 0).sum()

    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Total'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Total'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Total'].sum())
    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Bad_cnt'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Bad_cnt'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Bad_cnt'].sum())
    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Good_cnt'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Good_cnt'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Good_cnt'].sum())

    iv_tb['Bad_%'] = iv_tb['Bad_cnt'] / iv_tb['Total']
    iv_tb['Good_%'] = iv_tb['Good_cnt'] / iv_tb['Total']
    iv_tb['WOE'] = np.log(iv_tb['Good_%'] / iv_tb['Bad_%'])
    iv_tb['IV'] = (iv_tb['Good_%'] - iv_tb['Bad_%']) * iv_tb['WOE']
    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'IV'] = iv_tb.loc[iv_tb[x_bin] != 'SUM', 'IV'].sum()

    iv_tb = pd.concat([iv_tb.iloc[:, 0], iv_tb.iloc[:, 1:].fillna(0)], axis=1)
    iv_tb = pd.concat([iv_tb.iloc[:, 0], iv_tb.iloc[:, 1:].replace([np.inf, -np.inf], 0)], axis=1)

    print(iv_tb)
    print(iv_tb['IV'].sum().round(2))

    # 画图
    plt.figure(figsize=[12, 8])
    sns.pointplot(x=x_bin, y="WOE", data=iv_tb)
    plt.xticks(rotation=30)
    plt.show()
def iv_filter(data, y, x, point: str = "NULL", labels='NULL', right=False):
    df = data[[y, x]]

    # 连续性特征处理
    if point != 'NULL':
        x_bin = x + '_bin'
        df[x_bin] = pd.cut(df[x], point, right=right, labels=labels)
        df[x_bin] = df[x_bin].cat.add_categories(['Missing', 'SUM'])
    else:
        x_bin = x + '_bin'
        df[x_bin] = df[x].astype('category')
        df[x_bin] = df[x_bin].cat.add_categories(['Missing', 'SUM'])

    # 表格计算
    iv_tb = df.groupby([x_bin])[y].agg(['count', 'sum']).reset_index().rename(
        columns={'count': 'Total', 'sum': 'Bad_cnt'})
    iv_tb['Good_cnt'] = iv_tb['Total'] - iv_tb['Bad_cnt']

    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Total'] = len(df[x])
    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Bad_cnt'] = (df[y] == 1).sum()
    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Good_cnt'] = (df[y] == 0).sum()

    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Total'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Total'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Total'].sum())
    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Bad_cnt'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Bad_cnt'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Bad_cnt'].sum())
    iv_tb.loc[iv_tb[x_bin] == 'Missing', 'Good_cnt'] = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'Good_cnt'].values - (
        iv_tb.loc[np.arange(iv_tb.shape[0] - 2), 'Good_cnt'].sum())

    iv_tb['Bad_%'] = iv_tb['Bad_cnt'] / iv_tb['Total']
    iv_tb['Good_%'] = iv_tb['Good_cnt'] / iv_tb['Total']
    iv_tb['WOE'] = np.log(iv_tb['Good_%'] / iv_tb['Bad_%'])
    iv_tb['IV'] = (iv_tb['Good_%'] - iv_tb['Bad_%']) * iv_tb['WOE']

    iv_tb = pd.concat([iv_tb.iloc[:, 0], iv_tb.iloc[:, 1:].fillna(0)], axis=1)
    iv_tb = pd.concat([iv_tb.iloc[:, 0], iv_tb.iloc[:, 1:].replace([np.inf, -np.inf], 0)], axis=1)

    iv_tb.loc[iv_tb[x_bin] == 'SUM', 'IV'] = iv_tb.loc[iv_tb[x_bin] != 'SUM', 'IV'].sum()
    IV = iv_tb.loc[iv_tb[x_bin] == 'SUM', 'IV'].values

    if IV >= 1:
        print(IV.round(2), x)
        return x
def labelencoding(data, x):
    le = preprocessing.LabelEncoder()
    le.fit(data[x].tolist())
    print(x, le.classes_)
    data[x] = le.transform(data[x])
    return data
def re_astype(data, x):
    if (data[x].dtype in ['int8', 'int16', 'int32', 'int64']) & (data[x].min() < 0):
        if (data[x].max() <= 127) | (data[x].min() >= -128):
            data[x] = data[x].astype('int8')
        else:
            if (data[x].max() <= 32767) | (data[x].min() >= -32768):
                data[x] = data[x].astype('int16')
            else:
                if (data[x].max() <= 2147483647) | (data[x].min() >= -2147483648):
                    data[x] = data[x].astype('int32')
                else:
                    data[x] = data[x].astype('int64')
    if (data[x].dtype in ['int8', 'int16', 'int32', 'int64']) & (data[x].min() >= 0):
        if data[x].max() <= 255:
            data[x] = data[x].astype('uint8')
        else:
            if data[x].max() <= 65535:
                data[x] = data[x].astype('uint16')
            else:
                if data[x].max() <= 4294967295:
                    data[x] = data[x].astype('uint32')
                else:
                    data[x] = data[x].astype('uint64')
    return data[x]
def ratio_2_fetures_f(data, x, y):
    new_feature = x + '/' + y
    temp = np.where(data[y] == 0, 0.00000001, data[y])
    data[new_feature] = (data[x] / temp).astype('float64')
    return data, new_feature
def ratio_2_fetures(data, x, y):
    new_feature = x + '/' + y
    temp = np.where(data[y] == 0, 0.00000001, data[y])
    data[new_feature] = data[x] / temp
    print(new_feature)
    return data
def combines_cate_featuer(data, x, y, z='NULL'):
    x_cates = data[x].unique()
    y_cates = data[y].unique()
    if z == 'NULL':
        comb = itertools.product(x_cates, y_cates)
        for x_cat, y_cat in comb:
            new_feature = 'f_' + x + '_' + str(x_cat) + y + '_' + str(y_cat)
            flag_1_index = ((data[x] == x_cat) & (data[y] == y_cat) & (data['Set'] == 1))
            flag_1_rate = (data.loc[flag_1_index, 'TARGET'] == 1).sum() / data.loc[flag_1_index].shape[0]
            flag_0_rate = (data.loc[~flag_1_index, 'TARGET'] == 1).sum() / data.loc[(~flag_1_index) & (data['Set'] == 1)].shape[0]
            if (flag_1_index.sum() >= data.shape[0]*0.0001) & (abs(flag_1_rate - flag_0_rate) >= 0.05):
                data[new_feature] = np.where((data[x] == x_cat) & (data[y] == y_cat), 1, 0).astype('uint8')
                print(new_feature)
    else:
        z_cates = data[z].unique()
        comb = itertools.product(x_cates, y_cates, z_cates)
        for x_cat, y_cat, z_cat in comb:
            new_feature = 'f_' + x + '_' + str(x_cat) + y + '_' + str(y_cat) + z + '_' + str(z_cat)
            flag_1_index = ((data[x] == x_cat) & (data[y] == y_cat) & (data[z] == z_cat) & (data['Set'] == 1))
            flag_1_rate = (data.loc[flag_1_index, 'TARGET'] == 1).sum() / data.loc[flag_1_index].shape[0]
            flag_0_rate = (data.loc[~flag_1_index, 'TARGET'] == 1).sum() / data.loc[(~flag_1_index) & (data['Set'] == 1)].shape[0]
            if (flag_1_index.sum() >= data.shape[0]*0.0001) & (abs(flag_1_rate - flag_0_rate) >= 0.05):
                data[new_feature] = np.where((data[x] == x_cat) & (data[y] == y_cat) & (data[z] == z_cat), 1, 0).astype('uint8')
                print(new_feature)
    return data
def division_class_mean_medi(data, x, y):
    print((x, y))
    new_feature_mean = 'dc_mean_' + x + '_' + y
    new_feature_medi = 'dc_medi_' + x + '_' + y
    data[new_feature_mean] = np.nan
    data[new_feature_medi] = np.nan
    data_stat = data[[x, y]].groupby(y).agg(['mean', 'median']).reset_index()
    data_stat.columns = [y, 'mean', 'median']
    cates = data[y].unique()
    for cate in cates:
        data.loc[data[y] == cate, new_feature_mean] = data.loc[data[y] == cate, x] / data_stat.loc[data_stat[y] == cate, 'mean'].values
        data.loc[data[y] == cate, new_feature_medi] = data.loc[data[y] == cate, x] / data_stat.loc[data_stat[y] == cate, 'median'].values
    return data
def rank_feature(data, x):
    new_feature = 'Rank_' + x
    data[new_feature] = stats.rankdata(np.multiply(-1, data[x]))
    return data
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
def div_f(data, x, y, add_1=False, x_neg=False, y_neg=False):
    t1 = data.copy()
    nf = x + '_div_' + y
    print(nf)

    if x_neg:
        t1[x] = -t1[x]
    if y_neg:
        t1[y] = -t1[y]

    if add_1:
        t1[nf] = t1[x] / (t1[y]+1)
    else:
        t1[nf] = t1[x] / t1[y]
    return t1, nf
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

#
installments_payments = pd.read_csv("Data/installments_payments.csv").sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'NUM_INSTALMENT_VERSION'])
installments_payments['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']
installments_payments['AMT_INSTALMENT_AMT_PAYMENT_diff'] = installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']
ip_base = installments_payments[['SK_ID_CURR', 'SK_ID_PREV']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV'], keep='first').reset_index(drop=True)
ip_base_CURR = ip_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first').reset_index(drop=True)
ip_fe = ip_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first').reset_index(drop=True)
t0 = installments_payments.copy()

def count_f(data, feature, groupby_='SK_ID_CURR', nf=''):
    t1 = data.copy()
    new_feature = feature+nf+'_cnt'
    t2 = t1[[groupby_, feature]].groupby(by=groupby_)[feature].count().reset_index().rename(str, columns={feature: new_feature})
    return t2, new_feature

SK_ID_PREV_cnt, new_feature = count_f(data=ip_base, feature='SK_ID_PREV', nf='_ip')

t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION']].copy()
t2 = t1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].agg('max').reset_index().rename(str, columns={'NUM_INSTALMENT_VERSION': 'INSTALMENT_VERSION'})
t2['INSTALMENT_VERSION_0_flag'] = np.where(t2['INSTALMENT_VERSION'] == 0, 1, 0)
t2['INSTALMENT_VERSION_1_flag'] = np.where(t2['INSTALMENT_VERSION'] == 1, 1, 0)
t3 = t2.groupby('SK_ID_CURR')['INSTALMENT_VERSION_0_flag'].agg(['max']).reset_index().rename(str, columns={'max': 'INSTALMENT_VERSION_0_flag'})
t4 = t2.groupby('SK_ID_CURR')['INSTALMENT_VERSION_1_flag'].agg(['max']).reset_index().rename(str, columns={'max': 'INSTALMENT_VERSION_1_flag'})
t5 = add_feas_cont(ip_base_CURR, [t3, t4])
t5['status_0_1'] = np.where(t5['INSTALMENT_VERSION_0_flag']+t5['INSTALMENT_VERSION_1_flag']>=1,1,0)
t0 = add_feas(t0, t5[['SK_ID_CURR', 'status_0_1']], on='SK_ID_CURR')
del t1, t2, t3, t4, t5

# 1. 信用分期账户个数
ip_fe = add_feas(ip_fe, SK_ID_PREV_cnt)

# 2. 已结清（存在2）、未结清（只有1）、信用卡账户（0）、大于2的个数、占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION']].copy()
t2 = t1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].agg('max').reset_index().rename(str, columns={'NUM_INSTALMENT_VERSION': 'INSTALMENT_VERSION'})
t2['INSTALMENT_VERSION_0_flag'] = np.where(t2['INSTALMENT_VERSION'] == 0, 1, 0)
t2['INSTALMENT_VERSION_1_flag'] = np.where(t2['INSTALMENT_VERSION'] == 1, 1, 0)
t2['INSTALMENT_VERSION_2_flag'] = np.where(t2['INSTALMENT_VERSION'] == 2, 1, 0)
t2['INSTALMENT_VERSION_bg_flag'] = np.where(t2['INSTALMENT_VERSION'] > 2, 1, 0)
for x in ['INSTALMENT_VERSION_0_flag', 'INSTALMENT_VERSION_1_flag', 'INSTALMENT_VERSION_2_flag', 'INSTALMENT_VERSION_bg_flag']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'sum']).reset_index().rename(str, columns={'max': x, 'sum':x+'_cnt'})
    t4 = add_feas_cont(ip_base_CURR, [t3, SK_ID_PREV_cnt])
    t4[x+'_rto'] = t4[x+'_cnt'] / t4['SK_ID_PREV_ip_cnt']
    del t4['SK_ID_PREV_ip_cnt']
    ip_fe = add_feas(ip_fe, t4)
    del t3, t4
del t1, t2

# 3. 总期数：version=2 大于2取最大值 的时候的instalment number => stat
t1 = t0[t0['NUM_INSTALMENT_VERSION'] >= 2].copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
t3 = t2.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max': 'NUM_INSTALMENT_NUMBER'+'_max_v2', 'min': 'NUM_INSTALMENT_NUMBER'+'_min_v2', 'mean': 'NUM_INSTALMENT_NUMBER'+'_mean_v2', 'sum': 'NUM_INSTALMENT_NUMBER'+'_sum_v2', 'std': 'NUM_INSTALMENT_NUMBER'+'_std_v2'})
t4 = add_feas(ip_base_CURR, t3, on='SK_ID_CURR')
t4 = t4.fillna(0)
ip_fe = add_feas(ip_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# 4. 总期数(包含1)：instalment number最大值 => stat
t1 = t0.copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
t3 = t2.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max': 'NUM_INSTALMENT_NUMBER'+'_max_v', 'min': 'NUM_INSTALMENT_NUMBER'+'_min_v', 'mean': 'NUM_INSTALMENT_NUMBER'+'_mean_v', 'sum': 'NUM_INSTALMENT_NUMBER'+'_sum_v', 'std': 'NUM_INSTALMENT_NUMBER'+'_std_v'})
t4 = add_feas(ip_base_CURR, t3, on='SK_ID_CURR')
t4 = t4.fillna(0)
ip_fe = add_feas(ip_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# 5. 期数个数：instalment number count() => stat
t1 = t0.copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].count().reset_index().rename(str, columns={'NUM_INSTALMENT_NUMBER': 'NUM_INSTALMENT_NUMBER_num'})
t3 = t2.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER_num'].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max': 'NUM_INSTALMENT_NUMBER_num'+'_max', 'min': 'NUM_INSTALMENT_NUMBER_num'+'_min', 'mean': 'NUM_INSTALMENT_NUMBER_num'+'_mean', 'sum': 'NUM_INSTALMENT_NUMBER_num'+'_sum', 'std': 'NUM_INSTALMENT_NUMBER_num'+'_std'})
t4 = add_feas(ip_base_CURR, t3, on='SK_ID_CURR')
t4 = t4.fillna(0)
ip_fe = add_feas(ip_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# 6. 最早应还日期距今时间
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['DAYS_INSTALMENT'].min().reset_index().rename(str, columns={'DAYS_INSTALMENT':'DAYS_INSTALMENT_ft'})
t2['DAYS_INSTALMENT_ft'] = -t2['DAYS_INSTALMENT_ft']
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 7. 最近应还日期距今时间
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['DAYS_INSTALMENT'].max().reset_index().rename(str, columns={'DAYS_INSTALMENT':'DAYS_INSTALMENT_lt'})
t2['DAYS_INSTALMENT_lt'] = -t2['DAYS_INSTALMENT_lt']
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

t1 = t0.loc[t0['status_0_1'] == 1, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['DAYS_INSTALMENT'].max().reset_index().rename(str, columns={'DAYS_INSTALMENT':'DAYS_INSTALMENT_lt_01'})
t2['DAYS_INSTALMENT_lt_01'] = -t2['DAYS_INSTALMENT_lt_01']
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 8. 最近实还日期距今时间
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['DAYS_ENTRY_PAYMENT'].max().reset_index().rename(str, columns={'DAYS_ENTRY_PAYMENT':'DAYS_ENTRY_PAYMENT_lt'})
t2['DAYS_ENTRY_PAYMENT_lt'] = -t2['DAYS_ENTRY_PAYMENT_lt']
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

t1 = t0.loc[t0['status_0_1'] == 1, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']].copy()
t2 = t1.groupby(by='SK_ID_CURR')['DAYS_ENTRY_PAYMENT'].max().reset_index().rename(str, columns={'DAYS_ENTRY_PAYMENT':'DAYS_ENTRY_PAYMENT_lt_01'})
t2['DAYS_ENTRY_PAYMENT_lt_01'] = -t2['DAYS_ENTRY_PAYMENT_lt_01']
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 9. 应还日期在最近30、60、90、180、360、540、720、1080的账户数、占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT']].copy()
t2 = t1.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['DAYS_INSTALMENT'].max().reset_index().rename(str, columns={'DAYS_INSTALMENT':'DAYS_INSTALMENT_lt'})
for x in [-30, -60, -90, -180, -360, -540, -720, -1080]:
    t3 = t2[t2['DAYS_INSTALMENT_lt'] >= x]
    t4 = t3.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV':'SK_ID_PREV_cnt_including_'+str(-x)})
    t5 = add_feas_cont(ip_base_CURR, [t4, SK_ID_PREV_cnt])
    t5= t5.fillna(0)
    t5['SK_ID_PREV_rto_including_'+str(-x)] = t5['SK_ID_PREV_cnt_including_'+str(-x)] / t5['SK_ID_PREV_ip_cnt']
    del t5['SK_ID_PREV_ip_cnt']
    ip_fe = add_feas(ip_fe, t5)
    del t3, t4, t5
del t1, t2

# 10. 未结清（只有1）, 应还日期在最近30、60、90、180、360、540、720、1080的账户数、占比
t1 = t0.loc[t0['status_0_1'] == 1, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT']].copy()
t2 = t1.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['DAYS_INSTALMENT'].max().reset_index().rename(str, columns={'DAYS_INSTALMENT':'DAYS_INSTALMENT_lt'})
for x in [-30, -60, -90, -180, -360, -540, -720, -1080]:
    t3 = t2[t2['DAYS_INSTALMENT_lt'] >= x]
    t4 = t3.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV':'SK_ID_PREV_status01_cnt_including_'+str(-x)})
    t5 = add_feas_cont(ip_base_CURR, [t4, SK_ID_PREV_cnt])
    t5= t5.fillna(0)
    t5['SK_ID_PREV_status01_rto_including_'+str(-x)] = t5['SK_ID_PREV_status01_cnt_including_'+str(-x)] / t5['SK_ID_PREV_ip_cnt']
    del t5['SK_ID_PREV_ip_cnt']
    ip_fe = add_feas(ip_fe, t5)
    del t3, t4, t5
del t1, t2

gc.collect()

# 11. 应还日期-实还日期 >0 =0 <0 个数、占比（/总期数(包含1） /期数个数) => stat
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT','DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff']].copy()
t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_0'] = np.where(t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] == 0, 1, 0)
t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_bg0'] = np.where(t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] > 0, 1, 0)
t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_le0'] = np.where(t1['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] < 0, 1, 0)
t2 = t1.groupby(by=['SK_ID_PREV'])['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_0', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_bg0', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_le0'].sum().reset_index()

t3 = t0.copy()
t4 = t3.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
t5 = t3.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].count().reset_index().rename(str, columns={'NUM_INSTALMENT_NUMBER': 'NUM_INSTALMENT_NUMBER_num'})

t6 = add_feas_cont(ip_base, [t2, t4, t5], on='SK_ID_PREV')
for x in ['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_0', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_bg0', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff_le0']:
    t6, nf = div_f(data=t6, x=x, y='NUM_INSTALMENT_NUMBER')
    t6, nf = div_f(data=t6, x=x, y='NUM_INSTALMENT_NUMBER_num')
del t6['NUM_INSTALMENT_NUMBER'], t6['NUM_INSTALMENT_NUMBER_num']

for x in [x for x in t6.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV']]:
    t7 = t6.groupby('SK_ID_CURR')[x].agg(['max', 'min','mean','sum','std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    t8 = add_feas(ip_base_CURR, t7)
    ip_fe = add_feas(ip_fe, t8)
    del t7, t8
del t1, t2, t3, t4, t5, t6

# 12. 应还日期-实还日期 groupby(SK_ID_CURR) => max  最大提前还款天数 （取正数）
t1 = t0.loc[t0['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] >= 0, ['SK_ID_CURR', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff']].copy()
t2 = t1.groupby('SK_ID_CURR')['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'].max().reset_index().rename(columns={'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff':'max_repay_early'})
t3 = add_feas(ip_base_CURR, t2)
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 13. 应还日期-实还日期 groupby(SK_ID_CURR) => min 最大逾期天数（取正数）
t1 = t0.loc[t0['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] < 0, ['SK_ID_CURR', 'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff']].copy()
t2 = t1.groupby('SK_ID_CURR')['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'].max().reset_index().rename(columns={'DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff':'max_ovd_days'})
t2['max_ovd_days'] = -t2['max_ovd_days']
t3 = add_feas(ip_base_CURR, t2)
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 14. 存在逾期的账户个数、占比    是否逾期_还款日期_flag
t1 = t0.loc[t0['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] < 0, ['SK_ID_CURR', 'SK_ID_PREV']].copy()
t2 = t1.sort_values(['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(['SK_ID_CURR', 'SK_ID_PREV'])
t3 = t2.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'SK_ID_PREV_ovd_cnt'})
t4 = add_feas_cont(ip_base_CURR, [t3, SK_ID_PREV_cnt])
t4 = t4.fillna(0)
t4, nf = div_f(t4, x='SK_ID_PREV_ovd_cnt', y='SK_ID_PREV_ip_cnt')
del t4['SK_ID_PREV_ip_cnt']
ip_fe = add_feas(ip_fe, t4)
del t1, t2, t3, t4

ip_fe['CURR_ovd_flag'] = np.where(ip_fe['SK_ID_PREV_ovd_cnt'] > 0, 1, 0)

# 15. 存在提前还款的账户个数、占比   是否提前还款_flag     提前还款且未逾期_flag
t1 = t0.loc[t0['DAYS_INSTALMENT_DAYS_ENTRY_PAYMENT_diff'] > 0, ['SK_ID_CURR', 'SK_ID_PREV']].copy()
t2 = t1.sort_values(['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(['SK_ID_CURR', 'SK_ID_PREV'])
t3 = t2.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'SK_ID_PREV_repay_early_cnt'})
t4 = add_feas_cont(ip_base_CURR, [t3, SK_ID_PREV_cnt])
t4 = t4.fillna(0)
t4, nf = div_f(t4, x='SK_ID_PREV_repay_early_cnt', y='SK_ID_PREV_ip_cnt')
del t4['SK_ID_PREV_ip_cnt']
ip_fe = add_feas(ip_fe, t4)
del t1, t2, t3, t4

ip_fe['CURR_repay_early_flag'] = np.where(ip_fe['SK_ID_PREV_repay_early_cnt'] > 0, 1, 0)
ip_fe['CURR_repay_early_and_no_ovd_flag'] = np.where((ip_fe['CURR_ovd_flag'] == 0) & (ip_fe['CURR_repay_early_flag'] == 1), 1, 0)

# 19. 同一期还款的还款次数 groupby(SK_ID_CURR) => max
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).size().reset_index()
t3 = t2.groupby(['SK_ID_CURR', 'SK_ID_PREV'])[0].max().reset_index().rename(columns={0: 'max_sameinstalment_cnt'})
t4 = t3.groupby(['SK_ID_CURR'])['max_sameinstalment_cnt'].max().reset_index()
t5 = add_feas(ip_base_CURR, t4)
ip_fe = add_feas(ip_fe, t5)
del t1, t2, t3, t4, t5

# 20. 每期还款金额/应还金额 groupby(SK_ID_CURR) => max/min
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']].copy()
t1, nf = div_f(t1, x='AMT_PAYMENT', y='AMT_INSTALMENT')
t2 = t1.groupby(['SK_ID_CURR'])['AMT_PAYMENT_div_AMT_INSTALMENT'].agg(['max', 'min']).reset_index().rename(str,columns={'max': 'AMT_PAYMENT_div_AMT_INSTALMENT_max', 'min':'AMT_PAYMENT_div_AMT_INSTALMENT_min'})
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 21. 应还金额-每期还款金额/应还金额 groupby(SK_ID_CURR) => max/min
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']].copy()
t1 = sub_div_f(t1, x='AMT_INSTALMENT', y='AMT_PAYMENT', z='AMT_INSTALMENT')
t2 = t1.groupby(['SK_ID_CURR'])['AMT_INSTALMENT_sub_AMT_PAYMENT_div_AMT_INSTALMENT'].agg(['max', 'min']).reset_index().rename(str,columns={'max': 'AMT_INSTALMENT_sub_AMT_PAYMENT_div_AMT_INSTALMENT_max', 'min':'AMT_INSTALMENT_sub_AMT_PAYMENT_div_AMT_INSTALMENT_min'})
t3 = add_feas(ip_base_CURR, t2)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 22. 最小应还/最大应还 stat => stat
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT']].copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_INSTALMENT'].agg(['max', 'min']).reset_index().rename(str, columns={'max':'AMT_INSTALMENT_max', 'min':'AMT_INSTALMENT_min'})
t2, nf = div_f(t2, x='AMT_INSTALMENT_min', y='AMT_INSTALMENT_max')
for x in ['AMT_INSTALMENT_max', 'AMT_INSTALMENT_min', 'AMT_INSTALMENT_min_div_AMT_INSTALMENT_max']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'min','mean','sum','std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    t4 = add_feas(ip_base_CURR, t3)
    ip_fe = add_feas(ip_fe, t4)
    del t3, t4
del t1, t2

# 23. 最大应还-最小应还/最大应还 stat => stat
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT']].copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_INSTALMENT'].agg(['max', 'min']).reset_index().rename(str, columns={'max':'AMT_INSTALMENT_max', 'min':'AMT_INSTALMENT_min'})
t2 = sub_div_f(t2, x='AMT_INSTALMENT_max', y='AMT_INSTALMENT_min', z='AMT_INSTALMENT_max')
x = 'AMT_INSTALMENT_max_sub_AMT_INSTALMENT_min_div_AMT_INSTALMENT_max'
t3 = t2.groupby('SK_ID_CURR')[x].agg(['max', 'min', 'mean', 'sum', 'std']).reset_index().rename(str, columns={'max': x + '_max', 'min': x + '_min', 'mean': x + '_mean', 'sum': x + '_sum', 'std': x + '_std'})
t4 = add_feas(ip_base_CURR, t3)
ip_fe = add_feas(ip_fe, t4)
del t1, t2, t3, t4

gc.collect()

# -- 汇总逾期金额记录 => 都只有一期（实际还款日期以最后还款时间为准）=> DEF_flag/DEF_big_flag(>=50) --
t1 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION',
         'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
         'AMT_INSTALMENT', 'AMT_PAYMENT']].copy()
t1 = t1.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION',
                     'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT'])
t1['ovd_flag'] = np.where(t1['DAYS_ENTRY_PAYMENT'] - t1['DAYS_INSTALMENT'] > 0, 1, 0)

# 逾期金额
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'ovd_flag'])['AMT_INSTALMENT', 'AMT_PAYMENT'].agg({'AMT_INSTALMENT':'max', 'AMT_PAYMENT':'sum'}).reset_index()
t3 = t2.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION',
                     'NUM_INSTALMENT_NUMBER', 'ovd_flag']).drop_duplicates(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION',
                     'NUM_INSTALMENT_NUMBER'], keep='first')
t3['ovd_amt'] = np.where(t3['ovd_flag'] == 0, t3['AMT_INSTALMENT']-t3['AMT_PAYMENT'], t3['AMT_PAYMENT'])
t4 = t3.drop(['ovd_flag', 'AMT_INSTALMENT', 'AMT_PAYMENT'], axis=1)

t5 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']).agg({'DAYS_INSTALMENT':'max', 'DAYS_ENTRY_PAYMENT':'max', 'AMT_INSTALMENT':'max', 'AMT_PAYMENT':'sum'}).reset_index()
t6 = add_feas(t5, t4, on=['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER'], how='left')
t00 = t6.copy()
del t1, t2, t3, t4, t5, t6

t00 = t00.rename(index=str, columns={"DAYS_ENTRY_PAYMENT": "DAYS_ENTRY_PAYMENT_last", "AMT_PAYMENT": "AMT_PAYMENT_last"})
t00['ovd_days'] = t00['DAYS_ENTRY_PAYMENT_last'] - t00['DAYS_INSTALMENT']
t00['DEF_flag'] = np.where(t00['ovd_amt'] > 0, 1, 0)
t00['DEF_big_flag'] = np.where(t00['ovd_amt'] > 10, 1, 0)

# 24. 逾期期数占比
t1 = t00[t00['DEF_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_CURR')['DEF_flag'].sum().reset_index().rename(str, columns={'DEF_flag': 'DEF_cnt'})
t3 = t0.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'SK_ID_PREV_cnt_f'})
t4 = add_feas_cont(ip_base_CURR, [t2, t3])
t4 = t4.fillna(0)
t4, nf = div_f(t4, x='DEF_cnt', y='SK_ID_PREV_cnt_f')
ip_fe = add_feas(ip_fe, t4[['SK_ID_CURR', 'DEF_cnt', 'DEF_cnt_div_SK_ID_PREV_cnt_f']])
del t1, t2, t3, t4

# 24. DEF_flag 存在逾期（逾期金额>0）的那期 逾期天数 groupby(SK_ID_CURR) => max min mean sum
t1 = t00[t00['DEF_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_CURR')['ovd_days'].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max':'ovd_days'+'_max', 'min':'ovd_days'+'_min', 'mean':'ovd_days'+'_mean', 'sum':'ovd_days'+'_sum', 'std':'ovd_days'+'_std'})
t3 = add_feas_cont(ip_base_CURR, [t2])
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 25. DEF_big_flag 存在逾期（逾期金额>0）的那期 逾期天数 groupby(SK_ID_CURR) => max
t1 = t00[t00['DEF_big_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_CURR')['ovd_days'].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max':'ovd_days'+'_max_big', 'min':'ovd_days'+'_min_big', 'mean':'ovd_days'+'_mean_big', 'sum':'ovd_days'+'_sum_big', 'std':'ovd_days'+'_std_big'})
t3 = add_feas_cont(ip_base_CURR, [t2])
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 26. 逾期期数>0  占比（/总期数(包含1） /期数个数） => stat
t1 = t00[t00['DEF_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_PREV')['DEF_flag'].sum().reset_index().rename(str, columns={'DEF_flag': 'DEF_cnt'})
t3 = t00[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
t4 = t3.drop_duplicates(['SK_ID_PREV'], keep='last').rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_lt'})
t5 = t3.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].count().reset_index().rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_count'})
t6 = add_feas_cont(ip_base, [t2, t4, t5], on='SK_ID_PREV')
t6 = t6.fillna(0)
t6, nf = div_f(t6, x='DEF_cnt', y='NUM_INSTALMENT_NUMBER_lt')
t6, nf = div_f(t6, x='DEF_cnt', y='NUM_INSTALMENT_NUMBER_count')
t7 = t6[['SK_ID_CURR', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_lt', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_count']]
for x in ['DEF_cnt_div_NUM_INSTALMENT_NUMBER_lt', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_count']:
    t8 = t7.groupby('SK_ID_CURR')[x].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max':x+'_max', 'min':x+'_min', 'mean':x+'_mean', 'sum':x+'_sum', 'std':x+'_std'})
    ip_fe = add_feas(ip_fe, t8)
    del t8
del t1, t2, t3, t4, t5, t6, t7

t1 = t00[t00['DEF_big_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_PREV')['DEF_flag'].sum().reset_index().rename(str, columns={'DEF_flag': 'DEF_cnt'})
t3 = t00[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
t4 = t3.drop_duplicates(['SK_ID_PREV'], keep='last').rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_lt_big'})
t5 = t3.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].count().reset_index().rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_count_big'})
t6 = add_feas_cont(ip_base, [t2, t4, t5], on='SK_ID_PREV')
t6 = t6.fillna(0)
t6, nf = div_f(t6, x='DEF_cnt', y='NUM_INSTALMENT_NUMBER_lt_big')
t6, nf = div_f(t6, x='DEF_cnt', y='NUM_INSTALMENT_NUMBER_count_big')
t7 = t6[['SK_ID_CURR', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_lt_big', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_count_big']]
for x in ['DEF_cnt_div_NUM_INSTALMENT_NUMBER_lt_big', 'DEF_cnt_div_NUM_INSTALMENT_NUMBER_count_big']:
    t8 = t7.groupby('SK_ID_CURR')[x].agg(['max','min','mean','sum','std']).reset_index().rename(str, columns={'max':x+'_max', 'min':x+'_min', 'mean':x+'_mean', 'sum':x+'_sum', 'std':x+'_std'})
    ip_fe = add_feas(ip_fe, t8)
    del t8
del t1, t2, t3, t4, t5, t6, t7


# 27. 逾期金额  => max min sum      sum|max/总期数(包含1） sum|max/应还金额（最小值） sum|max/应还金额（最大值）
t1 = t00[t00['DEF_flag'] == 1].copy()
t2 = t1.groupby('SK_ID_PREV')['ovd_amt'].agg(['max','min','sum','mean']).reset_index().rename(str, columns={'max':'ovd_amt'+'_max', 'min':'ovd_amt'+'_min', 'mean':'ovd_amt'+'_mean', 'sum':'ovd_amt'+'_sum', 'std':'ovd_amt'+'_std'})
t3 = t00[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
t4 = t3.drop_duplicates(['SK_ID_PREV'], keep='last').rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_lt'})
t5 = t3.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].count().reset_index().rename(str, columns={'NUM_INSTALMENT_NUMBER':'NUM_INSTALMENT_NUMBER_count'})
t_5 = t00.groupby('SK_ID_PREV')['AMT_INSTALMENT'].agg(['max', 'min']).reset_index().rename(str, columns={'max':'AMT_INSTALMENT_max', 'min':'AMT_INSTALMENT_min'})
t6 = add_feas_cont(ip_base, [t2, t4, t5, t_5], on='SK_ID_PREV')
t6 = t6.fillna(0)
nfs = ['ovd_amt_max', 'ovd_amt_min', 'ovd_amt_sum', 'ovd_amt_mean']
for x in [x for x in t6.columns if x not in ['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER_lt', 'NUM_INSTALMENT_NUMBER_count']]:
    t6, nf1 = div_f(t6, x=x, y='NUM_INSTALMENT_NUMBER_lt')
    t6, nf2 = div_f(t6, x=x, y='NUM_INSTALMENT_NUMBER_count')
    t6, nf3 = div_f(t6, x=x, y='AMT_INSTALMENT_max')
    t6, nf4 = div_f(t6, x=x, y='AMT_INSTALMENT_min')
    nfs.extend([nf1, nf2, nf3, nf4])
for x in nfs:
    t7 = t6.groupby('SK_ID_CURR')[x].agg(['max','min','sum','mean']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ip_fe = add_feas(ip_fe, t7)
    del t7
del t1, t2, t3, t4, t5, t6, t_5

# 28. 当期：ovd/应还金额 groupby(SK_ID_CURR) => max min
t1 = t00.loc[t00['ovd_amt'] > 0, ['SK_ID_CURR', 'AMT_INSTALMENT', 'ovd_amt']].copy()
t1, nf = div_f(t1, x='ovd_amt', y='AMT_INSTALMENT')
t2 = t1.groupby('SK_ID_CURR')['ovd_amt_div_AMT_INSTALMENT'].agg(['max','min','sum','mean']).reset_index().rename(str, columns={'max': 'ovd_amt_div_AMT_INSTALMENT'+'_max', 'min': 'ovd_amt_div_AMT_INSTALMENT'+'_min', 'mean': 'ovd_amt_div_AMT_INSTALMENT'+'_mean', 'sum': 'ovd_amt_div_AMT_INSTALMENT'+'_sum', 'std': 'ovd_amt_div_AMT_INSTALMENT'+'_std'})
t3 = add_feas(ip_base_CURR, t2)
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

t1 = t00.loc[t00['ovd_amt'] > 10, ['SK_ID_CURR', 'AMT_INSTALMENT', 'ovd_amt']].copy()
t1, nf = div_f(t1, x='ovd_amt', y='AMT_INSTALMENT')
t2 = t1.groupby('SK_ID_CURR')['ovd_amt_div_AMT_INSTALMENT'].agg(['max','min','sum','mean']).reset_index().rename(str, columns={'max': 'ovd_amt_div_AMT_INSTALMENT'+'_max_big', 'min': 'ovd_amt_div_AMT_INSTALMENT'+'_min_big', 'mean': 'ovd_amt_div_AMT_INSTALMENT'+'_mean_big', 'sum': 'ovd_amt_div_AMT_INSTALMENT'+'_sum_big', 'std': 'ovd_amt_div_AMT_INSTALMENT'+'_std_big'})
t3 = add_feas(ip_base_CURR, t2)
t3 = t3.fillna(0)
ip_fe = add_feas(ip_fe, t3)
del t1, t2, t3

# 30. 总金额 sum(每期应还) 总还款金额  sum(每期实还)  sum(每期实还)/sum(每期应还)  sum(ovd)/sum(每期应还) => stat
t1 = t00[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT_last', 'ovd_amt']].copy()
t2 = t1.groupby(['SK_ID_CURR', 'SK_ID_PREV']).sum().reset_index().rename(str, columns={'AMT_INSTALMENT':'AMT_INSTALMENT_total', 'AMT_PAYMENT_last':'AMT_PAYMENT_last_total', 'ovd_amt':'ovd_amt_total'})
t2, nf = div_f(t2, x='AMT_PAYMENT_last_total', y='AMT_INSTALMENT_total')
t2, nf = div_f(t2, x='ovd_amt_total', y='AMT_INSTALMENT_total')
for x in ['AMT_INSTALMENT_total', 'AMT_PAYMENT_last_total', 'ovd_amt_total',
          'AMT_PAYMENT_last_total_div_AMT_INSTALMENT_total',
          'ovd_amt_total_div_AMT_INSTALMENT_total']:
    t3 = t2.groupby('SK_ID_CURR')[x].agg(['max','min','sum','mean','std']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean', 'sum': x+'_sum', 'std': x+'_std'})
    ip_fe = add_feas(ip_fe, t3)
    del t3
del t1, t2

# Write HDF
ip_base_CURR.shape
ip_fe.shape
ip_fe = ip_fe.fillna(-999)
print("ip_fe : ", ip_fe.shape[1] - 1)
ip_fe.to_hdf('Data_/Prev_app/installments_pay.hdf', 'installments_pay', mode='w', format='table')




