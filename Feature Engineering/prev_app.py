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

# 声明：无重复的PREV记录  根据PREV去重shape不变  So:groupby(by='SK_ID_CURR', left_on='SK_ID_CURR')
prev_app = pd.read_csv("Data/previous_application.csv")
prev_app_base = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).drop_duplicates(subset=['SK_ID_PREV'], keep='first')
prev_app_base_CURR = prev_app_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')
prev_app_fe = prev_app_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')
t0 = prev_app.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV']).copy()

def count_f(data, feature, groupby_='SK_ID_CURR', nf=''):
    t1 = data.copy()
    new_feature = feature+nf+'_cnt'
    t2 = t1[[groupby_, feature]].groupby(by=groupby_)[feature].count().reset_index().rename(str, columns={feature: new_feature})
    return t2, new_feature
def to_fe(data, features=[], base_=prev_app_base_CURR):
    t1 = data.copy()
    t = base_.copy()
    for feature in features:
        t2 = t1[['SK_ID_CURR', feature]].copy()
        t = add_feas(base=t, feature=t2)
        del t2
    return t

# 1. 贷款申请数量 (prev_app_count)
t1, ft = count_f(t0, feature='SK_ID_PREV', groupby_='SK_ID_CURR')
prev_app_fe = add_feas(prev_app_fe, t1, on='SK_ID_CURR')
prev_app_count = prev_app_fe.copy()
del t1

# 2. 各贷款类型申请数量/占比/Flag
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_TYPE']].copy()
t1['NAME_CONTRACT_TYPE'].value_counts()
for cato in t1['NAME_CONTRACT_TYPE'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_CONTRACT_TYPE'] == cato], feature='NAME_CONTRACT_TYPE', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' +cato+'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' +cato+'_flag']])
    del t2, t3
del t1

# 3. 贷款用途申请数量/占比 （Payments on other loans|Money for a third person|Refusal to name the goal|）
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CASH_LOAN_PURPOSE']].copy()
t1['NAME_CASH_LOAN_PURPOSE'].value_counts()
for cato in t1['NAME_CASH_LOAN_PURPOSE'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_CASH_LOAN_PURPOSE'] == cato], feature='NAME_CASH_LOAN_PURPOSE', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' +cato+'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' +cato+'_flag']])
    del t2, t3
del t1

# 4. 申请通过、取消、拒绝、unused数量/占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']].copy()
t1['NAME_CONTRACT_STATUS'].value_counts()
for cato in t1['NAME_CONTRACT_STATUS'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_CONTRACT_STATUS'] == cato], feature='NAME_CONTRACT_STATUS', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' +cato+'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' +cato+'_flag']])
    del t2, t3
del t1

# 5. 是否最后一次申请 | 是否是当天最后一次申请
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'FLAG_LAST_APPL_PER_CONTRACT']].copy()
t1['FLAG_LAST_APPL_PER_CONTRACT'].value_counts()
for cato in t1['FLAG_LAST_APPL_PER_CONTRACT'].unique():
    t2, cnt_feature = count_f(t1[t1['FLAG_LAST_APPL_PER_CONTRACT'] == cato], feature='FLAG_LAST_APPL_PER_CONTRACT', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' +cato+'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' +cato+'_flag']])
    del t2, t3
del t1

t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NFLAG_LAST_APPL_IN_DAY']].copy()
t1['NFLAG_LAST_APPL_IN_DAY'].value_counts()
for cato in t1['NFLAG_LAST_APPL_IN_DAY'].unique():
    t2, cnt_feature = count_f(t1[t1['NFLAG_LAST_APPL_IN_DAY'] == cato], feature='NFLAG_LAST_APPL_IN_DAY', nf='_'+str(cato))
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1
gc.collect()

# 6. NAME_YIELD_GROUP	利率级别 数量/占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_YIELD_GROUP']].copy()
t1['NAME_YIELD_GROUP'].value_counts()
for cato in t1['NAME_YIELD_GROUP'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_YIELD_GROUP'] == cato], feature='NAME_YIELD_GROUP', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

# 7. NAME_PORTFOLIO	CHANNEL_TYPE NAME_SELLER_INDUSTRY 资产类型 数量/占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_PORTFOLIO']].copy()
t1['NAME_PORTFOLIO'].value_counts()
for cato in t1['NAME_PORTFOLIO'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_PORTFOLIO'] == cato], feature='NAME_PORTFOLIO', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'CHANNEL_TYPE']].copy()
t1['CHANNEL_TYPE'].value_counts()
for cato in t1['CHANNEL_TYPE'].unique():
    t2, cnt_feature = count_f(t1[t1['CHANNEL_TYPE'] == cato], feature='CHANNEL_TYPE', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_SELLER_INDUSTRY']].copy()
t1['NAME_SELLER_INDUSTRY'].value_counts()
for cato in t1['NAME_SELLER_INDUSTRY'].unique():
    t2, cnt_feature = count_f(t1[t1['NAME_SELLER_INDUSTRY'] == cato], feature='NAME_SELLER_INDUSTRY', nf='_'+cato)
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

# 8. PRODUCT_COMBINATION	详细产品组合 数量/占比
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'PRODUCT_COMBINATION']].copy()
t1['PRODUCT_COMBINATION'].value_counts()
for cato in t1['PRODUCT_COMBINATION'].unique():
    t2, cnt_feature = count_f(t1[t1['PRODUCT_COMBINATION'] == cato], feature='PRODUCT_COMBINATION', nf='_'+str(cato))
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

# 9. 客户要求保险
t1 = t0[['SK_ID_CURR', 'SK_ID_PREV', 'NFLAG_INSURED_ON_APPROVAL']].copy()
t1['NFLAG_INSURED_ON_APPROVAL'] = t1['NFLAG_INSURED_ON_APPROVAL'].fillna(-1)
t1['NFLAG_INSURED_ON_APPROVAL'].value_counts()
for cato in t1['NFLAG_INSURED_ON_APPROVAL'].unique():
    t2, cnt_feature = count_f(t1[t1['NFLAG_INSURED_ON_APPROVAL'] == cato], feature='NFLAG_INSURED_ON_APPROVAL', nf='_'+str(cato))
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
    del t2, t3
del t1

# 10. 日期 最后应还距当前申请日期 是否为缺失或673065 数量/占比
# 11. 预计终止时间距当前申请日期 是否为缺失 数量/占比
t1 = t0[['SK_ID_CURR', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
         'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']].copy()
t1 = t1.replace({np.nan: 2, 365243.0: 1})
for day in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
         'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    print(day)
    for cato in [1, 2]:
        t2, cnt_feature = count_f(t1[t1[day] == cato], feature=day,
                                  nf='_' + str(cato))
        t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
        t3 = t3.fillna(0)
        t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
        t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
        prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
        del t2, t3
del t1

# 19. CNT_PAYMENT	期数 stat 最远 最近 额度最大 额度最小
# 21. NAME_YIELD_GROUP	利率级别  stat 最远 最近 额度最大 额度最小
# 14. 之前申请日期距当前贷款时间（其他时间） stat 最远 最近 额度最大 额度最小 diff-mean
# 25. 之前申请日期距当前贷款时间-首次支用距当前申请日期/之前申请日期距当前贷款时间 stat 最远 最近 额度最大 额度最小
# 26. 首次支用距当前申请日期-首次应还距当前申请日期 首次支用距当前申请日期-首次应还距当前申请日期/首次支用距当前申请日期 stat 最远 最近 额度最大 额度最小
# 28. 之前申请日期距当前贷款时间-最后应还距当前申请日期/之前申请日期距当前贷款时间 最后应还距当前申请日期/之前申请日期距当前贷款时间 stat 最远 最近 额度最大 额度最小
# 29. 之前申请日期距当前贷款时间-DAYS_TERMINATION/之前申请日期距当前贷款时间 DAYS_TERMINATION/之前申请日期距当前贷款时间 stat 最远 最近 额度最大 额度最小
# 27. 首次应还距当前申请日期-首次实还距当前申请日期 >0 =0 <0（dpd） 数量/占比
def stat_fl_mm(data, feature, succ=False, agg_list=['max', 'min', 'mean', 'sum', 'std']):
    if succ:
        t1 = data[data['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
        nf_base = str(feature) + '_ar'
    else:
        t1 = data.copy()
        nf_base = str(feature)
    t2 = t1.groupby('SK_ID_CURR')[feature].agg(agg_list).reset_index().rename(str, columns={'max': nf_base+'_max', 'min': nf_base+'_min', 'mean': nf_base+'_mean', 'sum': nf_base+'_sum', 'std': nf_base+'_std'}).fillna(0)
    t3 = t1[['SK_ID_CURR', 'DAYS_DECISION', feature]].sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).drop_duplicates(['SK_ID_CURR'], keep='first').rename(str, columns={feature: nf_base+'_ft'}).drop(['DAYS_DECISION'], axis=1)
    t4 = t1[['SK_ID_CURR', 'DAYS_DECISION', feature]].sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).drop_duplicates(['SK_ID_CURR'], keep='last').rename(str, columns={feature: nf_base+'_lt'}).drop(['DAYS_DECISION'], axis=1)
    if feature != 'AMT_CREDIT':
        t5 = t1[['SK_ID_CURR', 'AMT_CREDIT', feature]].sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(
            ['SK_ID_CURR'], keep='first').rename(str, columns={feature: nf_base + '_limit_min'}).drop(['AMT_CREDIT'], axis=1)
        t6 = t1[['SK_ID_CURR', 'AMT_CREDIT', feature]].sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(
            ['SK_ID_CURR'], keep='last').rename(str, columns={feature: nf_base + '_limit_max'}).drop(['AMT_CREDIT'], axis=1)
    else:
        t5 = t1[['SK_ID_CURR', 'AMT_CREDIT']].sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(
            ['SK_ID_CURR'], keep='first').rename(str, columns={feature: nf_base + '_limit_min'})
        t6 = t1[['SK_ID_CURR', 'AMT_CREDIT']].sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(
            ['SK_ID_CURR'], keep='last').rename(str, columns={feature: nf_base + '_limit_max'})
    t7 = add_feas_cont(prev_app_base_CURR, [t2, t3, t4, t5, t6])
    return t7

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


t1 = t0.copy()
t1 = t1.replace({365243.0: np.nan})
t1['NAME_YIELD_GROUP'] = t1['NAME_YIELD_GROUP'].replace({'XNA':0,
                                                         'low_normal':1,
                                                         'low_action':2,
                                                         'middle':3,
                                                         'high':4})
t1 = sub_div_f(data=t1, x='DAYS_DECISION', y='DAYS_FIRST_DRAWING', z='DAYS_DECISION')
t1 = substr(data=t1, x='DAYS_FIRST_DRAWING', y='DAYS_FIRST_DUE')
t1 = sub_div_f(data=t1, x='DAYS_FIRST_DRAWING', y='DAYS_FIRST_DUE', z='DAYS_FIRST_DRAWING')
t1 = sub_div_f(data=t1, x='DAYS_DECISION', y='DAYS_LAST_DUE', z='DAYS_DECISION')
t1 = sub_div_f(data=t1, x='DAYS_DECISION', y='DAYS_TERMINATION', z='DAYS_DECISION')
t1, temp = div_f(data=t1, x='DAYS_TERMINATION', y='DAYS_DECISION')
t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_bg_0'] = np.where(t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE'] > 0, 1, 0)
t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_eq_0'] = np.where(t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE'] == 0, 1, 0)
t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_ls_0'] = np.where(t1['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE'] < 0, 1, 0)
t1 = sub_div_f(t1, x='RATE_INTEREST_PRIMARY', y='RATE_INTEREST_PRIVILEGED', z='RATE_INTEREST_PRIMARY')

for fea in ['CNT_PAYMENT', 'NAME_YIELD_GROUP', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_LAST_DUE', 'DAYS_TERMINATION',
            'DAYS_DECISION_sub_DAYS_FIRST_DRAWING_div_DAYS_DECISION',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_div_DAYS_FIRST_DRAWING',
            'DAYS_DECISION_sub_DAYS_LAST_DUE_div_DAYS_DECISION',
            'DAYS_DECISION_sub_DAYS_TERMINATION_div_DAYS_DECISION',
            'DAYS_TERMINATION_div_DAYS_DECISION',
            'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
            'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
            'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
            'RATE_INTEREST_PRIVILEGED',
            'RATE_INTEREST_PRIMARY_sub_RATE_INTEREST_PRIVILEGED_div_RATE_INTEREST_PRIMARY']:
    print(fea)
    t2 = stat_fl_mm(t1, feature=fea, succ=False)
    prev_app_fe = add_feas(prev_app_fe, t2)
    del t2
for fea in ['CNT_PAYMENT', 'NAME_YIELD_GROUP', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_LAST_DUE', 'DAYS_TERMINATION',
            'DAYS_DECISION_sub_DAYS_FIRST_DRAWING_div_DAYS_DECISION',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_div_DAYS_FIRST_DRAWING',
            'DAYS_DECISION_sub_DAYS_LAST_DUE_div_DAYS_DECISION',
            'DAYS_DECISION_sub_DAYS_TERMINATION_div_DAYS_DECISION',
            'DAYS_TERMINATION_div_DAYS_DECISION', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
            'RATE_INTEREST_PRIMARY_sub_RATE_INTEREST_PRIVILEGED_div_RATE_INTEREST_PRIMARY']:
    print(fea)
    t2 = stat_fl_mm(t1, feature=fea, succ=True)
    prev_app_fe = add_feas(prev_app_fe, t2)
    del t2

for fea in ['DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_bg_0',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_eq_0',
            'DAYS_FIRST_DRAWING_sub_DAYS_FIRST_DUE_ls_0']:
    for cato in t1[fea].unique():
        t2, cnt_feature = count_f(t1[t1[fea] == cato], feature=fea, nf='_'+str(cato))
        t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
        t3 = t3.fillna(0)
        t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
        t3[cnt_feature + '_' + str(cato) +'_flag'] = np.where(t3[div_feature] > 0, 1, 0)
        prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato) +'_flag']])
        del t2, t3
del t1

# HDF
prev_app_fe = prev_app_fe.fillna(-999)
prev_app_fe.to_hdf('Data_/Prev_app/prev_app_2.hdf', 'prev_app_2', mode='w', table = True)

# 8. X/在该Y下均值/MAX
# （Y=贷款类型|贷款用途|商品类型|资产类型|渠道类型|销售行业|利率级别|期数类型）
# （X=年金|申请金额|授信金额|首付|商品价格|首付比例|基本利率|优惠利率）（先mean|max|sum）
t1 = t0.copy()
t1 = sub_div_f(t1, x='RATE_INTEREST_PRIMARY', y='RATE_INTEREST_PRIVILEGED', z='RATE_INTEREST_PRIMARY')
for X in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
          'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
          'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
          'RATE_INTEREST_PRIVILEGED', 'RATE_INTEREST_PRIMARY_sub_RATE_INTEREST_PRIVILEGED_div_RATE_INTEREST_PRIMARY']:
    for Y in ['NAME_CONTRACT_TYPE', 'NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY',
              'NAME_PORTFOLIO',
              'NAME_PRODUCT_TYPE',
              'CHANNEL_TYPE',
              'NAME_SELLER_INDUSTRY',
              'NAME_YIELD_GROUP',
              'CNT_PAYMENT']:
        print(X, ' / ', Y)
        t2 = t1[['SK_ID_CURR', Y, X]].groupby(by=['SK_ID_CURR', Y])[X].agg(['mean', 'max', 'sum']).reset_index().rename(str, columns={'mean':X+'_mean', 'max':X+'_max', 'sum':X+'_sum'})
        t3 = t1[[Y, X]].groupby(by=Y)[X].agg(['mean', 'max', 'sum']).reset_index().rename(str, columns={'mean':Y+'_mean', 'max':Y+'_max', 'sum':Y+'_sum'})
        t4 = add_feas(t2, t3, on=Y)
        t4, fea_mean = div_f(t4, x=X+'_mean', y=Y+'_mean')
        t4, fea_max = div_f(t4, x=X+'_max', y=Y+'_max')
        t4, fea_sum = div_f(t4, x=X+'_sum', y=Y+'_sum')
        t5 = t4[['SK_ID_CURR', fea_mean, fea_max, fea_sum]]
        for fea in [fea_mean, fea_max, fea_sum]:
            print(fea)
            t6 = t5.groupby(by='SK_ID_CURR')[fea].agg(['mean', 'max', 'min', 'std']).reset_index().rename(str,columns={'mean': fea+'_mean','max':fea+'_max','min':fea+'_min','std':fea+'_std'})
            t7 = add_feas(prev_app_base_CURR, t6).fillna(0)
            prev_app_fe = add_feas(prev_app_fe, t7)
            del t6, t7
        del t2, t3, t4, t5
del t1

t1 = t0[t0['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
t1 = sub_div_f(t1, x='RATE_INTEREST_PRIMARY', y='RATE_INTEREST_PRIVILEGED', z='RATE_INTEREST_PRIMARY')
for X in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
          'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
          'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
          'RATE_INTEREST_PRIVILEGED', 'RATE_INTEREST_PRIMARY_sub_RATE_INTEREST_PRIVILEGED_div_RATE_INTEREST_PRIMARY']:
    for Y in ['NAME_CONTRACT_TYPE', 'NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY',
              'NAME_PORTFOLIO',
              'NAME_PRODUCT_TYPE',
              'CHANNEL_TYPE',
              'NAME_SELLER_INDUSTRY',
              'NAME_YIELD_GROUP',
              'CNT_PAYMENT']:
        print(X, ' / ', Y)
        t2 = t1[['SK_ID_CURR', Y, X]].groupby(by=['SK_ID_CURR', Y])[X].agg(['mean', 'max', 'sum']).reset_index().rename(str, columns={'mean':X+'_mean_act', 'max':X+'_max_act', 'sum':X+'_sum_act'})
        t3 = t1[[Y, X]].groupby(by=Y)[X].agg(['mean', 'max', 'sum']).reset_index().rename(str, columns={'mean':Y+'_mean_act', 'max':Y+'_max_act', 'sum':Y+'_sum_act'})
        t4 = add_feas(t2, t3, on=Y)
        t4, fea_mean = div_f(t4, x=X+'_mean_act', y=Y+'_mean_act')
        t4, fea_max = div_f(t4, x=X+'_max_act', y=Y+'_max_act')
        t4, fea_sum = div_f(t4, x=X+'_sum_act', y=Y+'_sum_act')
        t5 = t4[['SK_ID_CURR', fea_mean, fea_max, fea_sum]]
        for fea in [fea_mean, fea_max, fea_sum]:
            print(fea)
            t6 = t5.groupby(by='SK_ID_CURR')[fea].agg(['mean', 'max', 'min', 'std']).reset_index().rename(str,columns={'mean': fea+'_mean','max':fea+'_max','min':fea+'_min','std':fea+'_std'})
            t7 = add_feas(prev_app_base_CURR, t6).fillna(0)
            prev_app_fe = add_feas(prev_app_fe, t7)
            del t6, t7
        del t2, t3, t4, t5
del t1

# HDF
prev_app_fe = prev_app_fe.fillna(-999)
prev_app_fe.to_hdf('Data_/Prev_app/prev_app_3.hdf', 'prev_app_3', mode='w', table = True)

# 9. X-Y X/Y X-Y/X （X=年金|申请金额|授信金额|首付|商品价格|首付比例|基本利率|优惠利率）（先mean|max|sum） （最后stat）
t1 = t0.copy()
for X in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
          'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
          'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
          'RATE_INTEREST_PRIVILEGED']:
    for Y in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
              'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']:
        if X != Y:
            print(X, ' and ', Y, '------------------------')
            t2 = t1[['SK_ID_CURR', X]].groupby(by='SK_ID_CURR')[X].agg(['max','mean','sum']).reset_index().rename(str, columns={'max':X+'_max', 'mean':X+'_mean', 'sum':X+'_sum'})
            t3 = t1[['SK_ID_CURR', Y]].groupby(by='SK_ID_CURR')[Y].agg(['max','mean','sum']).reset_index().rename(str, columns={'max':Y+'_max', 'mean':Y+'_mean', 'sum':Y+'_sum'})
            t4 = add_feas(t2, t3)
            t4 = substr(t4, x=X+'_max', y=Y+'_max')
            t4 = substr(t4, x=X+'_mean', y=Y+'_mean')
            t4 = substr(t4, x=X+'_sum', y=Y+'_sum')
            t4, name1 = div_f(t4, x=X+'_max', y=Y+'_max')
            t4, name2 = div_f(t4, x=X+'_mean', y=Y+'_mean')
            t4, name3 = div_f(t4, x=X+'_sum', y=Y+'_sum')
            t4 = sub_div_f(t4, x=X+'_max', y=Y+'_max', z=X+'_max')
            t4 = sub_div_f(t4, x=X+'_mean', y=Y+'_mean',  z=X+'_mean')
            t4 = sub_div_f(t4, x=X+'_sum', y=Y+'_sum', z=X+'_sum')
            t5 = t4[['SK_ID_CURR',
                     X + '_max' + '_sub_' +Y + '_max',
                     X + '_mean' + '_sub_' + Y + '_mean',
                     X + '_sum' + '_sub_' + Y + '_sum',
                     name1, name2, name3,
                     X + '_max' + '_sub_' + Y + '_max' + '_div_' + X + '_max',
                     X + '_mean' + '_sub_' + Y + '_mean' + '_div_' + X + '_mean',
                     X + '_sum' + '_sub_' + Y + '_sum' + '_div_' + X + '_sum']]
            t6 = add_feas(prev_app_base_CURR, t5)
            prev_app_fe = add_feas(prev_app_fe, t6)
            del t2, t3, t4, t5, t6

t1 = t0[t0['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
for X in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
          'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
          'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
          'RATE_INTEREST_PRIVILEGED']:
    for Y in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
              'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']:
        if X != Y:
            print(X, ' and ', Y, '------------------------')
            t2 = t1[['SK_ID_CURR', X]].groupby(by='SK_ID_CURR')[X].agg(['max','mean','sum']).reset_index().rename(str, columns={'max':X+'_max_act', 'mean':X+'_mean_act', 'sum':X+'_sum_act'})
            t3 = t1[['SK_ID_CURR', Y]].groupby(by='SK_ID_CURR')[Y].agg(['max','mean','sum']).reset_index().rename(str, columns={'max':Y+'_max_act', 'mean':Y+'_mean_act', 'sum':Y+'_sum_act'})
            t4 = add_feas(t2, t3)
            t4 = substr(t4, x=X+'_max_act', y=Y+'_max_act')
            t4 = substr(t4, x=X+'_mean_act', y=Y+'_mean_act')
            t4 = substr(t4, x=X+'_sum_act', y=Y+'_sum_act')
            t4, name1 = div_f(t4, x=X+'_max_act', y=Y+'_max_act')
            t4, name2 = div_f(t4, x=X+'_mean_act', y=Y+'_mean_act')
            t4, name3 = div_f(t4, x=X+'_sum_act', y=Y+'_sum_act')
            t4 = sub_div_f(t4, x=X+'_max_act', y=Y+'_max_act', z=X+'_max_act')
            t4 = sub_div_f(t4, x=X+'_mean_act', y=Y+'_mean_act',  z=X+'_mean_act')
            t4 = sub_div_f(t4, x=X+'_sum_act', y=Y+'_sum_act', z=X+'_sum_act')
            t5 = t4[['SK_ID_CURR',
                     X + '_max_act' + '_sub_' +Y + '_max_act',
                     X + '_mean_act' + '_sub_' + Y + '_mean_act',
                     X + '_sum_act' + '_sub_' + Y + '_sum_act',
                     name1, name2, name3,
                     X + '_max_act' + '_sub_' + Y + '_max_act' + '_div_' + X + '_max_act',
                     X + '_mean_act' + '_sub_' + Y + '_mean_act' + '_div_' + X + '_mean_act',
                     X + '_sum_act' + '_sub_' + Y + '_sum_act' + '_div_' + X + '_sum_act']]
            t6 = add_feas(prev_app_base_CURR, t5)
            prev_app_fe = add_feas(prev_app_fe, t6)
            del t2, t3, t4, t5, t6


# 近15、30、60、90、180、360、540、720、1800的申请、首次支用、首次实还、最后应还、预计终止 数量/占比
t1 = t0.loc[t0['NAME_CONTRACT_STATUS'] == 'Approved', ['SK_ID_CURR', 'DAYS_DECISION',
                                                       'DAYS_FIRST_DRAWING',
                                                       'DAYS_FIRST_DUE',
                                                       'DAYS_LAST_DUE_1ST_VERSION',
                                                       'DAYS_LAST_DUE',
                                                       'DAYS_TERMINATION']].copy()
t1 = t1.replace({365243.0: np.nan})
for X in ['DAYS_DECISION', 'DAYS_FIRST_DRAWING',
          'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
          'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    for N in [-15, -30, -60, -90, -180, -360, -540, -720, -1800]:
        print(N)
        t2 = t1.loc[t1[X] > N, ['SK_ID_CURR', X]].copy()
        t3 = t2.groupby(by='SK_ID_CURR')[X].count().reset_index().rename(str, columns={X:X+'_including_'+str(-N)})
        t4 = add_feas_cont(prev_app_base_CURR, [t3, prev_app_count]).fillna(0)
        t4, nf = div_f(t4, x=X+'_including_'+str(-N), y='SK_ID_PREV_cnt')
        t4 = t4.drop(['SK_ID_PREV_cnt'], axis=1)
        prev_app_fe = add_feas(prev_app_fe, t4)
        del t2, t3, t4
del t1

# HDF
prev_app_fe = prev_app_fe.fillna(-999)
prev_app_fe.to_hdf('Data_/Prev_app/prev_app_4.hdf', 'prev_app_4', mode='w', table = True)


# 11. 周几申请 是否不固定
# 12. 申请时间 是否不固定
t1 = t0[['SK_ID_CURR', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']].copy()
t1['WEEKDAY_APPR_PROCESS_START_n'] = np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'MONDAY', 1,
                                              np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'TUESDAY', 2,
                                                       np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'WEDNESDAY', 3,
                                                                np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'THURSDAY', 4,
                                                                         np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'FRIDAY', 5,
                                                                                np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'SATURDAY', 6,
                                                                                         np.where(t1['WEEKDAY_APPR_PROCESS_START'] == 'SUNDAY', 7, 8)))))))
del t1['WEEKDAY_APPR_PROCESS_START']
for cato in t1['WEEKDAY_APPR_PROCESS_START_n'].unique():
    t2, cnt_feature = count_f(t1[t1['WEEKDAY_APPR_PROCESS_START_n'] == cato], feature='WEEKDAY_APPR_PROCESS_START_n', nf='_'+str(cato))
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato)+'_WEEKDAY_APPR_PROCESS_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato)+'_WEEKDAY_APPR_PROCESS_flag']])
    del t2, t3
for cato in t1['HOUR_APPR_PROCESS_START'].unique():
    t2, cnt_feature = count_f(t1[t1['HOUR_APPR_PROCESS_START'] == cato], feature='HOUR_APPR_PROCESS_START', nf='_'+str(cato))
    t3 = add_feas_cont(prev_app_base_CURR, [t2, prev_app_count])
    t3 = t3.fillna(0)
    t3, div_feature = div_f(t3, x=cnt_feature, y='SK_ID_PREV_cnt')
    t3[cnt_feature + '_' + str(cato)+'_HOUR_APPR_PROCESS_START_flag'] = np.where(t3[div_feature] > 0, 1, 0)
    prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', cnt_feature, div_feature, cnt_feature + '_' + str(cato)+'_HOUR_APPR_PROCESS_START_flag']])
    del t2, t3

t2 = t1.groupby(by='SK_ID_CURR')['WEEKDAY_APPR_PROCESS_START_n'].std().reset_index().rename(str, columns={'WEEKDAY_APPR_PROCESS_START_n':'WEEKDAY_APPR_PROCESS_START_std'})
t2['WEEKDAY_APPR_PROCESS_START_same_flag'] = np.where(t2['WEEKDAY_APPR_PROCESS_START_std'] > 0, 0, 1)
t3 = add_feas(prev_app_base_CURR, t2)
t3 = t3.fillna(0)
prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', 'WEEKDAY_APPR_PROCESS_START_same_flag']])
del t2, t3
t2 = t1.groupby(by='SK_ID_CURR')['HOUR_APPR_PROCESS_START'].std().reset_index().rename(str, columns={'HOUR_APPR_PROCESS_START':'HOUR_APPR_PROCESS_START_std'})
t2['HOUR_APPR_PROCESS_START_same_flag'] = np.where(t2['HOUR_APPR_PROCESS_START_std'] > 0, 0, 1)
t3 = add_feas(prev_app_base_CURR, t2)
t3 = t3.fillna(0)
prev_app_fe = add_feas(prev_app_fe, t3[['SK_ID_CURR', 'HOUR_APPR_PROCESS_START_same_flag']])
del t2, t3

# FE2 => catogory features	类别型变量各类别的数量-违约率  |  OH encoding  （众数、最远、最近、额度最大、额度最小、预计终止时间距当前申请日期为缺失、）
# (X=贷款类型|周几申请|申请时点|贷款用途|申请结果|支付方式|拒绝原因|新旧客户类型|商品类型|资产类型|产品类型|渠道类型|销售地区|销售行业|利率级别|详细产品组合|)
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# Read Data
app_all = pd.read_hdf("Data_/Application/app_train_test.hdf")
app_target = app_all[['SK_ID_CURR', 'TARGET']]
del app_all

ALL = ['SK_ID_CURR', 'DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT','NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
     'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS',
     'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
     'NAME_CLIENT_TYPE', 'NAME_TYPE_SUITE',
     'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
     'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
     'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']
X = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
     'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS',
     'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
     'NAME_CLIENT_TYPE', 'NAME_TYPE_SUITE',
     'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
     'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
     'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']


t1 = t0[ALL].copy()
t1.loc[t1['NAME_TYPE_SUITE'].isnull(), 'NAME_TYPE_SUITE'] = 'XNA'
t1.loc[t1['PRODUCT_COMBINATION'].isnull(), 'PRODUCT_COMBINATION'] = 'XNA'


# 最远 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).drop_duplicates(['SK_ID_CURR'], keep='first').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_ft'
t4 = t4.rename(str, columns={'SK_ID_CURR_ft':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_ft'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_ft':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_ft'
t2 = t2.rename(str, columns={'SK_ID_CURR_ft':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t


# 最近 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).drop_duplicates(['SK_ID_CURR'], keep='last').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_lt'
t4 = t4.rename(str, columns={'SK_ID_CURR_lt':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_lt'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_lt':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_lt'
t2 = t2.rename(str, columns={'SK_ID_CURR_lt':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t

# 额度最小 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(['SK_ID_CURR'], keep='first').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_limitmin'
t4 = t4.rename(str, columns={'SK_ID_CURR_limitmin':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_limitmin'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_limitmin':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_limitmin'
t2 = t2.rename(str, columns={'SK_ID_CURR_limitmin':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t

# 额度最大 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'AMT_CREDIT']).drop_duplicates(['SK_ID_CURR'], keep='last').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_limitmax'
t4 = t4.rename(str, columns={'SK_ID_CURR_limitmax':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_limitmax'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_limitmax':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_limitmax'
t2 = t2.rename(str, columns={'SK_ID_CURR_limitmax':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t

# 预计终止最远 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'DAYS_TERMINATION']).drop_duplicates(['SK_ID_CURR'], keep='first').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_ter_ft'
t4 = t4.rename(str, columns={'SK_ID_CURR_ter_ft':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_ter_ft'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_ter_ft':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_ter_ft'
t2 = t2.rename(str, columns={'SK_ID_CURR_ter_ft':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t


# 预计终止最近 ---------
t2 = t1.sort_values(by=['SK_ID_CURR', 'DAYS_TERMINATION']).drop_duplicates(['SK_ID_CURR'], keep='last').drop(['DAYS_DECISION', 'DAYS_TERMINATION', 'AMT_CREDIT'], axis=1)
t2 = add_feas(t2, app_target)

# oh encoding
t3 = t2.copy().drop(['TARGET'], axis=1)
t4, tp = one_hot_encoder(t3, categorical_features=[i for i in X if i != 'SK_ID_CURR'], nan_as_category=True)
t4.columns = t4.columns+'_ter_lt'
t4 = t4.rename(str, columns={'SK_ID_CURR_ter_lt':'SK_ID_CURR'})
t5 = add_feas(prev_app_base_CURR, t4)
prev_app_fe = add_feas(prev_app_fe, t5)
del t3, t4, t5

# 违约率
from sklearn.utils import shuffle
train = t2.loc[~t2['TARGET'].isna(), ]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / (train.shape[0]/10))
test = t2.loc[t2['TARGET'].isna(), ]
fe2_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in X[1:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in [x for x in train[feature].unique() if (train[feature] == x).sum() >= 50]:
        cate_kfold_values = dict()
        for chuncks_index in train['chunks_index'].unique():
            # chuncks_target = train[train['chunks_index'] == chuncks_index]

            # stat K-1 folds
            chuncks_for_statistics = train.loc[
                (train['chunks_index'] != chuncks_index) & (train[feature] == cate), [feature, 'TARGET']]
            cate_kfold_values[chuncks_index] = (chuncks_for_statistics['TARGET'] == 1).sum() / \
                                               chuncks_for_statistics.shape[0]

            # fill K fold
            train.loc[(train['chunks_index'] == chuncks_index) & (train[feature] == cate), new_feature] = \
                cate_kfold_values[chuncks_index]

        test.loc[test[feature] == cate, new_feature] = np.mean(list(cate_kfold_values.values()))
        feature_cate_kfoldmean[cate] = np.mean(list(cate_kfold_values.values()))

    fe2_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

fe2 = pd.concat([train, test]).sort_values(by=['SK_ID_CURR']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
fe2.columns = fe2.columns+'_ter_lt'
fe2 = fe2.rename(str, columns={'SK_ID_CURR_ter_lt':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, fe2)
prev_app_fe = add_feas(prev_app_fe, t)
del fe2, t

# 类别个数-占比 （基于Train）
for feature in X[1:]:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    t2[new_feature_1] = np.nan
    t2[new_feature_2] = np.nan
    for cate in t2[feature].unique():
        t2.loc[t2[feature] == cate, new_feature_1] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum()
        t2.loc[t2[feature] == cate, new_feature_2] = (t2.loc[~t2['TARGET'].isna(), feature] == cate).sum() / \
                                                               t2[~t2['TARGET'].isna()].shape[0]
    t2 = t2.drop([feature], axis=1)

t2 = t2.drop(['TARGET'], axis=1)
t2.columns = t2.columns+'_ter_lt'
t2 = t2.rename(str, columns={'SK_ID_CURR_ter_lt':'SK_ID_CURR'})
t = add_feas(prev_app_base_CURR, t2)
prev_app_fe = add_feas(prev_app_fe, t)
del t2, t

# HDF
prev_app_fe = prev_app_fe.fillna(-999)
prev_app_fe.to_hdf('Data_/Prev_app/prev_app_5.hdf', 'prev_app_5', mode='w', table = True)

# Merge
prev_app_base_CURR.to_hdf('Data_/Prev_app/prev_app_base_CURR.hdf', 'prev_app_base_CURR', mode='w', table = True)












