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
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# Read Data
bureau = pd.read_csv("Data/bureau.csv").sort_values(by=['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop=True)
print("Bureau Shape : ", bureau.shape) # Bureau Shape :  (1716428, 17)
bureau_base = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].drop_duplicates(keep='first').reset_index(drop=True)})
bureau_fe = bureau_base

def stat(data, x, base=bureau_base):
    feature_name_add = ''
    df = data[['SK_ID_CURR', x]].copy()
    df1 = df.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': x+'_mean'+feature_name_add, 'max': x+'_max'+feature_name_add, 'min': x+'_min'+feature_name_add, 'std': x+'std'+feature_name_add}).fillna(0)

    feature_name_add = 'Active'
    df_active = data.loc[data['CREDIT_ACTIVE'] == feature_name_add, ['SK_ID_CURR', x]]
    df2 = df_active.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={
        'mean': x + '_mean' + feature_name_add, 'max': x + '_max' + feature_name_add,
        'min': x + '_min' + feature_name_add, 'std': x + 'std' + feature_name_add}).fillna(0)

    feature_name_add = 'Closed'
    df_closed = data.loc[data['CREDIT_ACTIVE'] == feature_name_add, ['SK_ID_CURR', x]]
    df3 = df_closed.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': x + '_mean' + feature_name_add, 'max': x + '_max' + feature_name_add, 'min': x + '_min' + feature_name_add, 'std': x + 'std' + feature_name_add}).fillna(0)

    feature_name_add = 'Consumer credit'
    df_consumer = data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df4 = df_consumer.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={
                                                                                                              'mean': x + '_mean' + feature_name_add,
                                                                                                              'max': x + '_max' + feature_name_add,
                                                                                                              'min': x + '_min' + feature_name_add,
                                                                                                              'std': x + 'std' + feature_name_add}).fillna(0)

    feature_name_add = 'Credit card'
    df_credit = data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df5 = df_credit.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str,
                                                                                                            columns={
                                                                                                                'mean': x + '_mean' + feature_name_add,
                                                                                                                'max': x + '_max' + feature_name_add,
                                                                                                                'min': x + '_min' + feature_name_add,
                                                                                                                'std': x + 'std' + feature_name_add}).fillna(0)

    feature_name_add = 'Car loan'
    df_car= data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df6 = df_car.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str,
                                                                                                          columns={
                                                                                                              'mean': x + '_mean' + feature_name_add,
                                                                                                              'max': x + '_max' + feature_name_add,
                                                                                                              'min': x + '_min' + feature_name_add,
                                                                                                              'std': x + 'std' + feature_name_add}).fillna(0)
    feature_name_add = 'Mortgage'
    df_mortgage = data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df7 = df_mortgage.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str,
                                                                                                       columns={
                                                                                                           'mean': x + '_mean' + feature_name_add,
                                                                                                           'max': x + '_max' + feature_name_add,
                                                                                                           'min': x + '_min' + feature_name_add,
                                                                                                           'std': x + 'std' + feature_name_add}).fillna(0)

    feature_name_add = 'Microloan'
    df_microloan = data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df8 = df_microloan.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str,
                                                                                                            columns={
                                                                                                                'mean': x + '_mean' + feature_name_add,
                                                                                                                'max': x + '_max' + feature_name_add,
                                                                                                                'min': x + '_min' + feature_name_add,
                                                                                                                'std': x + 'std' + feature_name_add}).fillna(0)
    feature_name_add = 'Loan for working capital replenishment'
    df_replenishment = data.loc[data['CREDIT_TYPE'] == feature_name_add, ['SK_ID_CURR', x]]
    df9 = df_replenishment.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str,
                                                                                                             columns={
                                                                                                                 'mean': x + '_mean' + feature_name_add,
                                                                                                                 'max': x + '_max' + feature_name_add,
                                                                                                                 'min': x + '_min' + feature_name_add,
                                                                                                                 'std': x + 'std' + feature_name_add}).fillna(0)

    fe = add_feas(base, df1)
    fe = add_feas(fe, df2)
    fe = add_feas(fe, df3)
    fe = add_feas(fe, df4)
    fe = add_feas(fe, df5)
    fe = add_feas(fe, df6)
    fe = add_feas(fe, df7)
    fe = add_feas(fe, df8)
    fe = add_feas(fe, df9)
    return fe
def stat_simple(data, x, base=bureau_base):
    feature_name_add = ''
    df = data[['SK_ID_CURR', x]].copy()
    df1 = df.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': x+'_mean'+feature_name_add, 'max': x+'_max'+feature_name_add, 'min': x+'_min'+feature_name_add, 'std': x+'std'+feature_name_add}).fillna(0)
    # df1 = df.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min']).reset_index().rename(index=str, columns={'mean': x+'_mean'+feature_name_add, 'max': x+'_max'+feature_name_add, 'min': x+'_min'+feature_name_add, 'std': x+'std'+feature_name_add}).fillna(0)

    feature_name_add = 'Active'
    df_active = data.loc[data['CREDIT_ACTIVE'] == feature_name_add, ['SK_ID_CURR', x]]
    df2 = df_active.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={
        'mean': x + '_mean' + feature_name_add, 'max': x + '_max' + feature_name_add,
        'min': x + '_min' + feature_name_add, 'std': x + 'std' + feature_name_add}).fillna(0)
    # df2 = df_active.groupby(by=['SK_ID_CURR'])[x].agg(['mean', 'max', 'min']).reset_index().rename(index=str, columns={
    #     'mean': x + '_mean' + feature_name_add, 'max': x + '_max' + feature_name_add,
    #     'min': x + '_min' + feature_name_add, 'std': x + 'std' + feature_name_add}).fillna(0)

    fe = add_feas(base, df1)
    fe = add_feas(fe, df2)
    return fe
def add_feas(base, feature, on='SK_ID_CURR', how='left'):
    base = base.merge(feature, on=on, how=how)
    del feature
    return base
def first_last(data, x, base=bureau_base):
    feature_name_add = ''
    df = data[['SK_ID_CURR', 'DAYS_CREDIT', x]].copy()
    t1 = df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset='SK_ID_CURR', keep='first').drop(['DAYS_CREDIT'], axis=1).rename(index=str, columns={x: x + feature_name_add + '_latest'})
    t2 = df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset='SK_ID_CURR', keep='last').drop(['DAYS_CREDIT'], axis=1).rename(index=str, columns={x: x + feature_name_add + '_last'})

    feature_name_add = 'Active'
    df = data.loc[data['CREDIT_ACTIVE'] == feature_name_add, ['SK_ID_CURR', 'DAYS_CREDIT', x]]
    t3 = df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset='SK_ID_CURR', keep='first').drop(['DAYS_CREDIT'], axis=1).rename(index=str, columns={x: x + '_' + feature_name_add + '_latest'})
    t4 = df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset='SK_ID_CURR', keep='last').drop(['DAYS_CREDIT'], axis=1).rename(index=str, columns={x: x + '_' + feature_name_add + '_last'})

    ff = add_feas(base, t1)
    ff = add_feas(ff, t2)
    ff = add_feas(ff, t3)
    ff = add_feas(ff, t4)

    del t1, t2, t3, t4
    return ff

'''
cates :
CREDIT_ACTIVE : ['Closed', 'Active', 'Sold', 'Bad debt']
CREDIT_CURRENCY : ['currency 1', 'currency 2', 'currency 4', 'currency 3']

'''
'''
1. (未逾期的)记录的最大授信额度，与此次申请的授信额度/收入...
2. 关注存在过sold的记录，"核销"
3. 原始变量onehot encodeing
'''

'''
# 1. CREDIT_ACTIVE	信贷情况
* 信贷总数
* closed数量/比例
* active数量/比例
* Sold数量/比例
* Bad debt数量/比例
* 是否存在Sold (Todo List)
* 是否存在Bad debt  (Todo List)
'''
t1 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t2 = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Active', ['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE_Active'})
t3 = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Closed', ['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE_Closed'})
t4 = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Sold', ['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE_Sold'})
t5 = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Bad debt', ['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE_Baddebt'})

t6 = add_feas(t1, t2, on='SK_ID_CURR')
t6 = add_feas(t6, t3, on='SK_ID_CURR')
t6 = add_feas(t6, t4, on='SK_ID_CURR')
t6 = add_feas(t6, t5, on='SK_ID_CURR')
t6 = t6.fillna(0)

t6['ratio_of_CREDIT_ACTIVE_Active'] = t6['num_of_CREDIT_ACTIVE_Active']/t6['num_of_CREDIT_ACTIVE']
t6['ratio_of_CREDIT_ACTIVE_Closed'] = t6['num_of_CREDIT_ACTIVE_Closed']/t6['num_of_CREDIT_ACTIVE']
t6['ratio_of_CREDIT_ACTIVE_Sold'] = t6['num_of_CREDIT_ACTIVE_Sold']/t6['num_of_CREDIT_ACTIVE']
t6['ratio_of_CREDIT_ACTIVE_Baddebt'] = t6['num_of_CREDIT_ACTIVE_Baddebt']/t6['num_of_CREDIT_ACTIVE']

t6['flag_of_CREDIT_ACTIVE_Sold'] = np.where(t6['ratio_of_CREDIT_ACTIVE_Sold'] > 0, 1, 0)
t6['flag_of_CREDIT_ACTIVE_Baddebt'] = np.where(t6['ratio_of_CREDIT_ACTIVE_Baddebt'] > 0, 1, 0)

bureau_fe = add_feas(bureau_base, t6, on='SK_ID_CURR')
del t1, t2, t3, t4, t5, t6

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
# 2. CREDIT_CURRENCY	信用货币
* 总数
* 数量/比例
* 是否存在非1
'''
# bureau['CREDIT_CURRENCY'].value_counts()
t1 = bureau[['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by=['SK_ID_CURR'])['CREDIT_CURRENCY'].count().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'num_of_CREDIT_CURRENCY'})
t2 = bureau.loc[bureau['CREDIT_CURRENCY'] == 'currency 1', ['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by=['SK_ID_CURR'])['CREDIT_CURRENCY'].count().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'num_of_CREDIT_CURRENCY_1'})
t3 = bureau.loc[bureau['CREDIT_CURRENCY'] == 'currency 2', ['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by=['SK_ID_CURR'])['CREDIT_CURRENCY'].count().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'num_of_CREDIT_CURRENCY_2'})
t4 = bureau.loc[bureau['CREDIT_CURRENCY'] == 'currency 3', ['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by=['SK_ID_CURR'])['CREDIT_CURRENCY'].count().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'num_of_CREDIT_CURRENCY_3'})
t5 = bureau.loc[bureau['CREDIT_CURRENCY'] == 'currency 4', ['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by=['SK_ID_CURR'])['CREDIT_CURRENCY'].count().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'num_of_CREDIT_CURRENCY_4'})

t6 = add_feas(t1, t2, on='SK_ID_CURR')
t6 = add_feas(t6, t3, on='SK_ID_CURR')
t6 = add_feas(t6, t4, on='SK_ID_CURR')
t6 = add_feas(t6, t5, on='SK_ID_CURR')
t6 = t6.fillna(0)

t6['ratio_of_CREDIT_CURRENCY_1'] = t6['num_of_CREDIT_CURRENCY_1']/t6['num_of_CREDIT_CURRENCY']
t6['ratio_of_CREDIT_CURRENCY_2'] = t6['num_of_CREDIT_CURRENCY_2']/t6['num_of_CREDIT_CURRENCY']
t6['ratio_of_CREDIT_CURRENCY_3'] = t6['num_of_CREDIT_CURRENCY_3']/t6['num_of_CREDIT_CURRENCY']
t6['ratio_of_CREDIT_CURRENCY_4'] = t6['num_of_CREDIT_CURRENCY_4']/t6['num_of_CREDIT_CURRENCY']
t6['flag_have_CREDIT_CURRENCY_not_1'] = np.where(t6['ratio_of_CREDIT_CURRENCY_1'] < 1, 1, 0)

bureau_fe = add_feas(bureau_fe, t6, on='SK_ID_CURR')
del t1, t2, t3, t4, t5, t6

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
# 3. DAYS_CREDIT 距今申请天数	
* Stat   最小的和平均的有用 最小表示离得近，资金紧张程度更高
* 与上一次的间隔统计
* diff.Stat 把第一个diff=>第一个的DAYS_CREDIT_positive，表示跟这次申请的距离，也反应资金紧张情况
* Diff mean/最早那笔的长度（stat max）
* Active => Stat diff.Stat
* Closed = > Stat diff.Stat
* 是否存在同时申请（总体，活跃： 0天 7天内) 占比

* 距离申请日N天内的记录数/占比 N=3 7 30 90 180 360 

'''
def days_credit(data, CREDIT_ACTIVE='all', base=bureau_base):
    if CREDIT_ACTIVE == 'all':
        df = data[['SK_ID_CURR', 'DAYS_CREDIT']].copy()
        feature_name_add = ''
    else:
        df = data.loc[bureau['CREDIT_ACTIVE'] == CREDIT_ACTIVE, ['SK_ID_CURR', 'DAYS_CREDIT']].copy()
        feature_name_add = CREDIT_ACTIVE
    df['DAYS_CREDIT_positive'] = -df['DAYS_CREDIT']
    del df['DAYS_CREDIT']
    df = df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT_positive'])

    # Stat
    df1 = df.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': 'DAYS_CREDIT_mean'+feature_name_add, 'max': 'DAYS_CREDIT_max'+feature_name_add, 'min': 'DAYS_CREDIT_min'+feature_name_add, 'std': 'DAYS_CREDIT_std'+feature_name_add}).fillna(0)

    # diff
    df2 = df.copy()
    df2['DAYS_CREDIT_positive_diff'] = df2.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].diff()
    df2.loc[df2['DAYS_CREDIT_positive_diff'].isna(), 'DAYS_CREDIT_positive_diff'] = df2['DAYS_CREDIT_positive']

    # diff stat
    df3 = df2.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive_diff'].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': 'DAYS_CREDIT_positive_diff_mean'+feature_name_add, 'max': 'DAYS_CREDIT_positive_diff_max'+feature_name_add, 'min': 'DAYS_CREDIT_positive_diff_min'+feature_name_add, 'std': 'DAYS_CREDIT_positive_diff_std'+feature_name_add}).fillna(0)

    # diff <=0 | <=7
    df4 = df2.copy()
    new_fea1 = 'DAYS_CREDIT_sametime_app_0' + '_' + feature_name_add
    new_fea2 = 'DAYS_CREDIT_sametime_app_7' + '_' + feature_name_add
    df4[new_fea1] = np.where(df4['DAYS_CREDIT_positive_diff'] == 0, 1, 0)
    df4[new_fea2] = np.where(df4['DAYS_CREDIT_positive_diff'] <= 7, 1, 0)
    df5 = df4.groupby(by=['SK_ID_CURR'])[new_fea1, new_fea2].sum().reset_index().rename(index=str, columns={new_fea1: new_fea1, new_fea2: new_fea2})
    df_ex = df.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT_positive': 'num_of_CREDIT_ACTIVE'})
    df5 = add_feas(df5, df_ex)

    new_fea3 = 'DAYS_CREDIT_sametime_app_0_ratio' + '_' + feature_name_add
    new_fea4 = 'DAYS_CREDIT_sametime_app_7_ratio' + '_' + feature_name_add
    df5[new_fea3] = df5[new_fea1] / df5['num_of_CREDIT_ACTIVE']
    df5[new_fea4] = df5[new_fea2] / df5['num_of_CREDIT_ACTIVE']
    del df_ex
    del df5['num_of_CREDIT_ACTIVE']

    # diff
    df6 = df.copy()
    df6['DAYS_CREDIT_positive_lastdiff'] = df6.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].diff(-1).fillna(0)
    df6['DAYS_CREDIT_positive_lastdiff'] = -df6['DAYS_CREDIT_positive_lastdiff']

    # diff stat
    df7 = df6.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive_lastdiff'].agg(
        ['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={
        'mean': 'DAYS_CREDIT_positive_lastdiff_mean' + feature_name_add,
        'max': 'DAYS_CREDIT_positive_lastdiff_max' + feature_name_add,
        'min': 'DAYS_CREDIT_positive_lastdiff_min' + feature_name_add,
        'std': 'DAYS_CREDIT_positive_lastdiff_std' + feature_name_add}).fillna(0)

    ff = add_feas(base, df1, on='SK_ID_CURR')
    ff = add_feas(ff, df3, on='SK_ID_CURR')
    ff = add_feas(ff, df5, on='SK_ID_CURR')
    ff = add_feas(ff, df7, on='SK_ID_CURR')

    if CREDIT_ACTIVE == 'all':
        ff['DAYS_CREDIT_positive_diff_mean/DAYS_CREDIT_max'] = ff['DAYS_CREDIT_positive_diff_mean']/ff['DAYS_CREDIT_max']

    del df, df1, df2, df3, df4, df5, df6, df7
    return ff

t1 = days_credit(data=bureau)
t2 = days_credit(data=bureau, CREDIT_ACTIVE='Active')
t3 = days_credit(data=bureau, CREDIT_ACTIVE='Closed')
bureau_fe = add_feas(bureau_fe, t1, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t2, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
del t1, t2, t3
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau.copy()
t0['DAYS_CREDIT_pos'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
t0['less_3_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 3, 1, 0)
t0['less_7_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 7, 1, 0)
t0['less_30_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 30, 1, 0)
t0['less_90_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 90, 1, 0)
t0['less_180_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 180, 1, 0)
t0['less_360_credit_num'] = np.where(t0['DAYS_CREDIT_pos'] <= 360, 1, 0)
t1 = t0[['SK_ID_CURR', 'less_3_credit_num', 'less_7_credit_num', 'less_30_credit_num', 'less_90_credit_num', 'less_180_credit_num', 'less_360_credit_num']].groupby(by=['SK_ID_CURR'])['less_3_credit_num', 'less_7_credit_num', 'less_30_credit_num', 'less_90_credit_num', 'less_180_credit_num', 'less_360_credit_num'].sum().reset_index()
t2 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t3 = add_feas(t2, t1, on='SK_ID_CURR')
t3['less_3_credit_ratio'] = t3['less_3_credit_num']/t3['num_of_CREDIT_ACTIVE']
t3['less_7_credit_ratio'] = t3['less_7_credit_num']/t3['num_of_CREDIT_ACTIVE']
t3['less_30_credit_ratio'] = t3['less_30_credit_num']/t3['num_of_CREDIT_ACTIVE']
t3['less_90_credit_ratio'] = t3['less_90_credit_num']/t3['num_of_CREDIT_ACTIVE']
t3['less_180_credit_ratio'] = t3['less_180_credit_num']/t3['num_of_CREDIT_ACTIVE']
t3['less_360_credit_ratio'] = t3['less_360_credit_num']/t3['num_of_CREDIT_ACTIVE']
del t3['num_of_CREDIT_ACTIVE']
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
del t0, t1, t2, t3

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
# 4. CREDIT_DAY_OVERDUE 逾期天数	
* 逾期天数大于N的笔数占比(Todo)
* Stat   最大的和平均的有用 
* Active => Stat
* Closed = > Stat
* 贷款种类 stat 
* 逾期笔数 /记录数，是否存在逾期，活跃逾期笔数 /记录数，是否存在活跃逾期
* 逾期记录距今申请天数 stat - 活跃

* 应还日期在申请日期前N天 逾期天数 stat 
* 应还日期在申请日期后 逾期天数

* 距今申请天数在30/90/180/360天内的逾期天数 stat (Todo)
* 最新一笔/最老一笔 逾期天数 
* 最大逾期天数/该笔时长/最短申请市时长/最长申请时长/平均申请时长

'''
t1 = stat(data=bureau, x='CREDIT_DAY_OVERDUE')

t2 = bureau.copy()
t2['flag_OVERDUE'] = np.where(t2['CREDIT_DAY_OVERDUE'] >= 1, 1, 0)
t3 = t2[['SK_ID_CURR', 'flag_OVERDUE']].groupby(by=['SK_ID_CURR'])['flag_OVERDUE'].sum().reset_index().rename(index=str, columns={'flag_OVERDUE': 'CREDIT_DAY_OVERDUE_flag_num'})
t3['flag_OVERDUE'] = np.where(t3['CREDIT_DAY_OVERDUE_flag_num'] >= 1, 1, 0)
t_ex = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t3 = add_feas(t3, t_ex)
t3['CREDIT_DAY_OVERDUE_flag_ratio'] = t3['CREDIT_DAY_OVERDUE_flag_num'] / t3['num_of_CREDIT_ACTIVE']
del t3['num_of_CREDIT_ACTIVE']
del t_ex

t4 = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy()
t4['flag_OVERDUE'] = np.where(t4['CREDIT_DAY_OVERDUE'] >= 1, 1, 0)
t5 = t4[['SK_ID_CURR', 'flag_OVERDUE']].groupby(by=['SK_ID_CURR'])['flag_OVERDUE'].sum().reset_index().rename(index=str, columns={'flag_OVERDUE': 'CREDIT_DAY_OVERDUE_flag_num_active'})
t5['flag_OVERDUE_active'] = np.where(t5['CREDIT_DAY_OVERDUE_flag_num_active'] >= 1, 1, 0)
t_ex = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t5 = add_feas(t5, t_ex)
t5['CREDIT_DAY_OVERDUE_flag_ratio_active'] = t5['CREDIT_DAY_OVERDUE_flag_num_active'] / t5['num_of_CREDIT_ACTIVE']
del t5['num_of_CREDIT_ACTIVE']
del t_ex

t6 = t2.loc[t2['flag_OVERDUE'] == 1, ['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'CREDIT_TYPE']]
t6['DAYS_CREDIT_positive_flag_OVERDUE'] = -t6['DAYS_CREDIT']
del t6['DAYS_CREDIT']
t7 = stat(data=t6, x='DAYS_CREDIT_positive_flag_OVERDUE')
t7 = t7.fillna(0)

bureau_fe = add_feas(bureau_fe, t1, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t5, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t7, on='SK_ID_CURR')
del t1, t2, t3, t4, t5, t6, t7
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < 0].copy()
t1 = stat(data=t0, x='CREDIT_DAY_OVERDUE')
t2 = bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] >= 0].copy()
t3 = stat(data=t2, x='CREDIT_DAY_OVERDUE')

bureau_fe = add_feas(bureau_fe, t1, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
del t0, t1, t2, t3
print("bureau_fe Shape : ", bureau_fe.shape)

t1 = first_last(data=bureau, x='CREDIT_DAY_OVERDUE')
bureau_fe = add_feas(bureau_fe, t1)
del t1
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].copy()
t0['DAYS_CREDIT_positive'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
t1 = t0.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': 'DAYS_CREDIT_mean', 'max': 'DAYS_CREDIT_max', 'min': 'DAYS_CREDIT_min', 'std': 'DAYS_CREDIT_std'}).fillna(0)
t2 = bureau[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE']].copy()
t2 = t2.groupby(by=['SK_ID_CURR'])['CREDIT_DAY_OVERDUE'].max().reset_index().rename(index=str, columns={'CREDIT_DAY_OVERDUE': 'CREDIT_DAY_OVERDUE_max'})
t2 = add_feas(t2, t1)
t2['CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_mean'] = t2['CREDIT_DAY_OVERDUE_max'] / t2['DAYS_CREDIT_mean']
t2['CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_max'] = t2['CREDIT_DAY_OVERDUE_max'] / t2['DAYS_CREDIT_max']
t2['CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_min'] = t2['CREDIT_DAY_OVERDUE_max'] / t2['DAYS_CREDIT_min']
t2 = t2[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_mean', 'CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_max', 'CREDIT_DAY_OVERDUE_max/DAYS_CREDIT_min']]
bureau_fe = add_feas(bureau_fe, t2)
del t0, t1, t2
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE']].copy()
t0['DAYS_CREDIT_positive'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
t1 = t0.sort_values(by=['SK_ID_CURR', 'CREDIT_DAY_OVERDUE']).drop_duplicates(subset='SK_ID_CURR', keep='last')
t1['CREDIT_DAY_OVERDUE_max/DAYS_CREDIT'] = t1['CREDIT_DAY_OVERDUE'] / t1['DAYS_CREDIT_positive']
t1 = t1[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_max/DAYS_CREDIT']]
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1
print("bureau_fe Shape : ", bureau_fe.shape)

# 逾期天数 用应还-实还替代 max
bureau['CREDIT_DAY_OVERDUE_t'] = np.where((bureau['DAYS_CREDIT_ENDDATE']-bureau['DAYS_ENDDATE_FACT']) >= 0, 0, -(bureau['DAYS_CREDIT_ENDDATE']-bureau['DAYS_ENDDATE_FACT']))
bureau['CREDIT_DAY_OVERDUE_t'] = bureau['CREDIT_DAY_OVERDUE_t'].fillna(0)
bureau['CREDIT_DAY_OVERDUE_t'] = bureau[['CREDIT_DAY_OVERDUE_t', 'CREDIT_DAY_OVERDUE']].max(axis=1)

t1 = stat(data=bureau, x='CREDIT_DAY_OVERDUE_t')

t2 = bureau.copy()
t2['flag_OVERDUE'] = np.where(t2['CREDIT_DAY_OVERDUE_t'] >= 1, 1, 0)
t3 = t2[['SK_ID_CURR', 'flag_OVERDUE']].groupby(by=['SK_ID_CURR'])['flag_OVERDUE'].sum().reset_index().rename(index=str, columns={'flag_OVERDUE': 'CREDIT_DAY_OVERDUE_t_flag_num'})
t3['flag_OVERDUE'] = np.where(t3['CREDIT_DAY_OVERDUE_t_flag_num'] >= 1, 1, 0)
t_ex = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t3 = add_feas(t3, t_ex)
t3['CREDIT_DAY_OVERDUE_t_flag_ratio'] = t3['CREDIT_DAY_OVERDUE_t_flag_num'] / t3['num_of_CREDIT_ACTIVE']
del t3['num_of_CREDIT_ACTIVE']
del t_ex

t4 = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy()
t4['flag_OVERDUE'] = np.where(t4['CREDIT_DAY_OVERDUE_t'] >= 1, 1, 0)
t5 = t4[['SK_ID_CURR', 'flag_OVERDUE']].groupby(by=['SK_ID_CURR'])['flag_OVERDUE'].sum().reset_index().rename(index=str, columns={'flag_OVERDUE': 'CREDIT_DAY_OVERDUE_t_flag_num_active'})
t5['flag_OVERDUE_active'] = np.where(t5['CREDIT_DAY_OVERDUE_t_flag_num_active'] >= 1, 1, 0)
t_ex = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t5 = add_feas(t5, t_ex)
t5['CREDIT_DAY_OVERDUE_t_flag_ratio_active'] = t5['CREDIT_DAY_OVERDUE_t_flag_num_active'] / t5['num_of_CREDIT_ACTIVE']
del t5['num_of_CREDIT_ACTIVE']
del t_ex

t6 = t2.loc[t2['flag_OVERDUE'] == 1, ['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'CREDIT_TYPE']]
t6['DAYS_CREDIT_positive_flag_OVERDUE'] = -t6['DAYS_CREDIT']
del t6['DAYS_CREDIT']
t7 = stat(data=t6, x='DAYS_CREDIT_positive_flag_OVERDUE')
t7 = t7.fillna(0)

bureau_fe = add_feas(bureau_fe, t1, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t5, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t7, on='SK_ID_CURR')
del t1, t2, t3, t4, t5, t6, t7
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < 0].copy()
t1 = stat(data=t0, x='CREDIT_DAY_OVERDUE_t')
t2 = bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] >= 0].copy()
t3 = stat(data=t2, x='CREDIT_DAY_OVERDUE_t')

bureau_fe = add_feas(bureau_fe, t1, on='SK_ID_CURR')
bureau_fe = add_feas(bureau_fe, t3, on='SK_ID_CURR')
del t0, t1, t2, t3
print("bureau_fe Shape : ", bureau_fe.shape)

t1 = first_last(data=bureau, x='CREDIT_DAY_OVERDUE_t')
bureau_fe = add_feas(bureau_fe, t1)
del t1
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].copy()
t0['DAYS_CREDIT_positive'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
t1 = t0.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_positive'].agg(['mean', 'max', 'min', 'std']).reset_index().rename(index=str, columns={'mean': 'DAYS_CREDIT_mean', 'max': 'DAYS_CREDIT_max', 'min': 'DAYS_CREDIT_min', 'std': 'DAYS_CREDIT_std'}).fillna(0)
t2 = bureau[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_t']].copy()
t2 = t2.groupby(by=['SK_ID_CURR'])['CREDIT_DAY_OVERDUE_t'].max().reset_index().rename(index=str, columns={'CREDIT_DAY_OVERDUE_t': 'CREDIT_DAY_OVERDUE_t_max'})
t2 = add_feas(t2, t1)
t2['CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_mean'] = t2['CREDIT_DAY_OVERDUE_t_max'] / t2['DAYS_CREDIT_mean']
t2['CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_max'] = t2['CREDIT_DAY_OVERDUE_t_max'] / t2['DAYS_CREDIT_max']
t2['CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_min'] = t2['CREDIT_DAY_OVERDUE_t_max'] / t2['DAYS_CREDIT_min']
t2 = t2[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_mean', 'CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_max', 'CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT_min']]
bureau_fe = add_feas(bureau_fe, t2)
del t0, t1, t2
print("bureau_fe Shape : ", bureau_fe.shape)

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE_t']].copy()
t0['DAYS_CREDIT_positive'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
t1 = t0.sort_values(by=['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_t']).drop_duplicates(subset='SK_ID_CURR', keep='last')
t1['CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT'] = t1['CREDIT_DAY_OVERDUE_t'] / t1['DAYS_CREDIT_positive']
t1 = t1[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_t_max/DAYS_CREDIT']]
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
# 5. DAYS_CREDIT_ENDDATE （应还日期）
* stat diff 贷款种类 

* 应还日期在申请日期后 flag 笔数/占比  N=3， 7, 30, 90
* 应还日期在申请日期后 且逾期 flag 笔数/占比
* 应还日期在申请日期前N天  flag 笔数/占比 N=3， 7, 30, 90
* 应还日期在申请日期前N天 且逾期  flag 笔数/占比
* 应还日期在申请日期前N天 还未实际还款 flag 笔数/占比 N=7, 30, 90

* Diff mean/申请时间 stat
* Diff mean/应还日期 stat

* 是否存在同时还款（活跃： 0天）占比 (Todo)
* 应还日期-距今申请天数 stat / 活跃 关闭 sold  应还日期-距今申请天数/距今申请天数 应还日期/距今申请天数
'''
def enddate(data, x, base=bureau_base):
    df0 = data[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})

    df = data.loc[data[x] >= 0, ['SK_ID_CURR', x, 'CREDIT_DAY_OVERDUE']]
    df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x+'_laterapp_num'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
    df1[x+'_laterapp_ratio'] = df1[x+'_laterapp_num'] / df1['num_of_CREDIT_ACTIVE']
    del df1['num_of_CREDIT_ACTIVE']
    ff = add_feas(base, df1)
    del df, df1

    for N in [3, 7, 30, 90]:
        nf = '_include' + str(N)
        df = data.loc[(data[x] >= 0) & (data[x] <= N), ['SK_ID_CURR', x, 'CREDIT_DAY_OVERDUE']]
        df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x + nf + '_laterapp_num'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
        df1[x + nf + '_laterapp_ratio'] = df1[x + nf + '_laterapp_num'] / df1['num_of_CREDIT_ACTIVE']
        del df1['num_of_CREDIT_ACTIVE']
        ff = add_feas(ff, df1)
        del df, df1

    for N in [-3, -7, -30, -90]:
        nf = '_include' + str(N)
        df = data.loc[(data[x] < 0) & (data[x] >= N), ['SK_ID_CURR', x, 'CREDIT_DAY_OVERDUE']]
        df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x + nf + '_beforeapp_num'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
        df1[x + nf + '_beforeapp_ratio'] = df1[x + nf + '_beforeapp_num'] / df1['num_of_CREDIT_ACTIVE']
        del df1['num_of_CREDIT_ACTIVE']
        ff = add_feas(ff, df1)
        del df, df1

    df = data.loc[(data[x] >= 0) & (data['CREDIT_DAY_OVERDUE'] > 0), ['SK_ID_CURR', x, 'CREDIT_DAY_OVERDUE']]
    df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x + '_laterapp_num_overdue'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
    df1[x + '_laterapp_ratio_overdue'] = df1[x + '_laterapp_num_overdue'] / df1['num_of_CREDIT_ACTIVE']
    del df1['num_of_CREDIT_ACTIVE']
    ff = add_feas(bureau_base, df1)
    del df, df1

    for N in [-30, -90]:
        nf = '_include' + str(N)
        df = data.loc[(data[x] < 0) & (data[x] >= N) & (data['CREDIT_DAY_OVERDUE'] > 0), ['SK_ID_CURR', x, 'CREDIT_DAY_OVERDUE']]
        df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x + nf + '_beforeapp_num_overdue'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
        df1[x + nf + '_beforeapp_ratio_overdue'] = df1[x + nf + '_beforeapp_num_overdue'] / df1['num_of_CREDIT_ACTIVE']
        del df1['num_of_CREDIT_ACTIVE']
        ff = add_feas(ff, df1)
        del df, df1

    if x == 'DAYS_CREDIT_ENDDATE':
        for N in [-7, -30, -90]:
            nf = '_include' + str(N)
            df = data.loc[(data[x] < 0) & (data[x] >= N) & (data['DAYS_ENDDATE_FACT'].isnull()), ['SK_ID_CURR', x, 'DAYS_ENDDATE_FACT']]
            df1 = df.groupby(by='SK_ID_CURR')[x].count().reset_index().merge(df0, on='SK_ID_CURR', how='right').rename(str, columns={x: x + nf + '_beforeapp_num_norepay'}).fillna(0).sort_values(by='SK_ID_CURR').reset_index(drop=True)
            df1[x + nf + '_beforeapp_ratio_norepay'] = df1[x + nf + '_beforeapp_num_norepay'] / df1['num_of_CREDIT_ACTIVE']
            del df1['num_of_CREDIT_ACTIVE']
            ff = add_feas(ff, df1)
            del df, df1

    df = data[['SK_ID_CURR', x, 'DAYS_CREDIT']].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'], ascending=[True, False])
    df[x+'_diff'] = df.groupby(by='SK_ID_CURR')[x].diff()
    df1 = df.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].agg(['max','min','mean']).reset_index().rename(str, columns={'max': 'DAYS_CREDIT_max', 'min': 'DAYS_CREDIT_min', 'mean': 'DAYS_CREDIT_mean'})
    df2 = df.groupby(by=['SK_ID_CURR'])[x].agg(['max', 'min', 'mean']).reset_index().rename(str, columns={'max': x+'_max', 'min': x+'_min', 'mean': x+'_mean'})
    df3 = df.groupby(by='SK_ID_CURR')[x+'_diff'].mean().reset_index().rename(str, columns={x+'_diff': x+'_diff_mean'})
    df3 = add_feas(df3, df1)
    df3 = add_feas(df3, df2)
    df3[x +'_diff_mean/' + 'DAYS_CREDIT_max'] = df3[x +'_diff_mean'] / df3['DAYS_CREDIT_max']
    df3[x +'_diff_mean/' + 'DAYS_CREDIT_min'] = df3[x +'_diff_mean'] / df3['DAYS_CREDIT_min']
    df3[x +'_diff_mean/' + 'DAYS_CREDIT_mean'] = df3[x +'_diff_mean'] / df3['DAYS_CREDIT_mean']
    df3[x +'_diff_mean/'  + x+'_max'] = df3[x +'_diff_mean'] / df3[x+'_max']
    df3[x +'_diff_mean/'  + x+'_min'] = df3[x +'_diff_mean'] / df3[x+'_min']
    df3[x +'_diff_mean/'  + x+'_mean'] = df3[x +'_diff_mean'] / df3[x+'_mean']
    ff = add_feas(ff, df3)
    del df0, df1, df2, df3

    return ff

t1 = stat(data=bureau, x='DAYS_CREDIT_ENDDATE', base=bureau_base)
t1 = t1.fillna(0)
bureau_fe = add_feas(bureau_fe, t1)
del t1

t1 = enddate(data=bureau, x='DAYS_CREDIT_ENDDATE', base=bureau_base)
t1 = t1.fillna(0)
bureau_fe = add_feas(bureau_fe, t1)
del t1

t0 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE']].copy()
t0['DAYS_CREDIT_ENDDATE-DAYS_CREDIT'] = t0['DAYS_CREDIT_ENDDATE'] - t0['DAYS_CREDIT']
t0['DAYS_CREDIT_ENDDATE/DAYS_CREDIT'] = t0['DAYS_CREDIT_ENDDATE'] / t0['DAYS_CREDIT']
t0['DAYS_CREDIT_ENDDATE-DAYS_CREDIT/DAYS_CREDIT'] = (t0['DAYS_CREDIT_ENDDATE'] - t0['DAYS_CREDIT']) / -t0['DAYS_CREDIT']

t1 = stat_simple(data=t0, x='DAYS_CREDIT_ENDDATE-DAYS_CREDIT', base=bureau_base)
t2 = stat_simple(data=t0, x='DAYS_CREDIT_ENDDATE/DAYS_CREDIT', base=bureau_base)
t3 = stat_simple(data=t0, x='DAYS_CREDIT_ENDDATE-DAYS_CREDIT/DAYS_CREDIT', base=bureau_base)

bureau_fe = add_feas(bureau_fe, t1)
bureau_fe = add_feas(bureau_fe, t2)
bureau_fe = add_feas(bureau_fe, t3)
del t0, t1, t2, t3

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
实际完成贷款距今天数（实还日期）（不能变负号）
* stat diff 贷款种类 
* 实还日期在申请日期前N天  flag 笔数/占比 N=0， 1， 3， 7  （金额）
* 实还日期在申请日期前N天 且逾期  flag 笔数/占比  N=0， 1， 3， 7
* Diff mean/记录次数
* Diff mean/记录次数 (与下一次间隔)
* 是否存在同时实际还款（活跃： 0天）占比
* 实还日期-距今申请天数 stat / 活跃 关闭 sold 

* 应还日期-实还日期 STAT 负数逾期
* 应还日期-实还日期 <0 -15 -30 -60 -90 次数/次数占比 Flag
* 有应还，无实还 flag 笔数/占比 (Todo)
* 提前还款 应还日期-实还日期>0 Flag/笔数/占比 >0 15 30 60 90
'''
t1 = stat(data=bureau, x='DAYS_ENDDATE_FACT', base=bureau_base)
t1 = t1.fillna(0)
bureau_fe = add_feas(bureau_fe, t1)
del t1

t1 = enddate(data=bureau, x='DAYS_ENDDATE_FACT', base=bureau_base)
t1 = t1.fillna(0)
bureau_fe = add_feas(bureau_fe, t1)
del t1

t0 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].copy()
t0['DAYS_ENDDATE_FACT-DAYS_CREDIT'] = t0['DAYS_ENDDATE_FACT'] - t0['DAYS_CREDIT']
t0['DAYS_ENDDATE_FACT/DAYS_CREDIT'] = t0['DAYS_ENDDATE_FACT'] / t0['DAYS_CREDIT']
t0['DAYS_ENDDATE_FACT-DAYS_CREDIT/DAYS_CREDIT'] = (t0['DAYS_ENDDATE_FACT'] - t0['DAYS_CREDIT']) / -t0['DAYS_CREDIT']
t1 = stat_simple(data=t0, x='DAYS_ENDDATE_FACT-DAYS_CREDIT', base=bureau_base)
t2 = stat_simple(data=t0, x='DAYS_ENDDATE_FACT/DAYS_CREDIT', base=bureau_base)
t3 = stat_simple(data=t0, x='DAYS_ENDDATE_FACT-DAYS_CREDIT/DAYS_CREDIT', base=bureau_base)
bureau_fe = add_feas(bureau_fe, t1)
bureau_fe = add_feas(bureau_fe, t2)
bureau_fe = add_feas(bureau_fe, t3)
del t0, t1, t2, t3


t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].copy()
t0['DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT'] = t0['DAYS_CREDIT_ENDDATE'] - t0['DAYS_ENDDATE_FACT']
t1 = t0[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT']].groupby(by='SK_ID_CURR').agg(['max','min','mean','std']).reset_index().rename(str, columns={'max': 'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_max', 'min':'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_min', 'mean':'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_mean', 'std':'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_std'})
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1


t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].copy()
t0['DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT'] = t0['DAYS_CREDIT_ENDDATE'] - t0['DAYS_ENDDATE_FACT']
num = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
for N in [0, -15, -30, -60, -90]:
    nf = 'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_' + str(N)
    t0[nf] = np.where(t0['DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT'] < N, 1, 0)
    t1 = t0[['SK_ID_CURR', nf]].groupby(by='SK_ID_CURR').max().reset_index().rename(str, columns={nf: nf+'_flag'})
    t2 = t0[['SK_ID_CURR', nf]].groupby(by='SK_ID_CURR').sum().reset_index().rename(str, columns={nf: nf + '_sum'})
    t3 = num.merge(t2, on='SK_ID_CURR', how='left')
    t3[nf + '_ratio'] = t3[nf + '_sum'] / t3['num_of_CREDIT_ACTIVE']
    del t3['num_of_CREDIT_ACTIVE']
    bureau_fe = add_feas(bureau_fe, t1)
    bureau_fe = add_feas(bureau_fe, t3)
    del t1, t2, t3

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].copy()
t0['DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT'] = t0['DAYS_CREDIT_ENDDATE'] - t0['DAYS_ENDDATE_FACT']
num = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
for N in [0, 7, 30, 60, 90]:
    nf = 'DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT_qrepay' + str(N)
    t0[nf] = np.where(t0['DAYS_CREDIT_ENDDATE-DAYS_ENDDATE_FACT'] > N, 1, 0)
    t1 = t0[['SK_ID_CURR', nf]].groupby(by='SK_ID_CURR').max().reset_index().rename(str, columns={nf: nf+'_flag'})
    t2 = t0[['SK_ID_CURR', nf]].groupby(by='SK_ID_CURR').sum().reset_index().rename(str, columns={nf: nf + '_sum'})
    t3 = num.merge(t2, on='SK_ID_CURR', how='left')
    t3[nf + '_ratio'] = t3[nf + '_sum'] / t3['num_of_CREDIT_ACTIVE']
    del t3['num_of_CREDIT_ACTIVE']
    bureau_fe = add_feas(bureau_fe, t1)
    bureau_fe = add_feas(bureau_fe, t3)
    del t1, t2, t3

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
信用延长次数
* stat 简单统计
* 是否存在信用延长 存在信用延长的笔数/占比
* 存在信用延长的距今申请天数 stat (Todo)
* 存在信用延长的距今申请天数 stat (Todo)
* 存在信用延长的应还/实还 stat (Todo)
'''
t0 = bureau[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']]
t1 = t0.groupby(by='SK_ID_CURR')['CNT_CREDIT_PROLONG'].agg(['count', 'sum']).reset_index().rename(str, columns={'count': 'num', 'sum': 'num_CNT_CREDIT_PROLONG'})
t1['ratio_CNT_CREDIT_PROLONG'] = t1['num_CNT_CREDIT_PROLONG'] / t1['num']
t1['flag_CREDIT_PROLONG'] = np.where(t1['num_CNT_CREDIT_PROLONG'] > 0, 1, 0)
del t1['num']
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
贷款类型 CREDIT_TYPE
* 贷款种类数 
* 众数 | 最新一笔 | 最早一笔 | 转化率
* 是否存在 各贷款类型
* 每种种类的占比
* IV 重点贷款类型 -> Point Type => 新增维度 (Todo)
CREDIT_TYPE :
Consumer credit                                 1251615
Credit card                                      402195
Car loan                                          27690
Mortgage                                          18391 # 抵押贷款
Microloan                                         12413 # 小微贷款
Loan for business development                      1975 # 经营贷
Another type of loan                               1017 
Unknown type of loan                                555 
Loan for working capital replenishment              469 # 资本补充贷款
Cash loan (non-earmarked)                            56
Real estate loan                                     27 # 房贷
Loan for the purchase of equipment                   19 
Loan for purchase of shares (margin lending)          4
Mobile operator loan                                  1
Interbank credit                                      1
'''
num = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t0 = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].copy()
t1 = t0.drop_duplicates(keep='first').reset_index(drop=True).groupby(by='SK_ID_CURR')['CREDIT_TYPE'].count().reset_index().rename(str, columns={'CREDIT_TYPE': 'num_CREDIT_TYPE'}).merge(num, on='SK_ID_CURR', how='left')
t1['ratio_CREDIT_TYPE'] = t1['num_CREDIT_TYPE'] / t1['num_of_CREDIT_ACTIVE']
del t1['num_of_CREDIT_ACTIVE']
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1
print("bureau_fe Shape : ", bureau_fe.shape)


# 几种方式的单ID贷款种类  众数 | 最新一笔 | 最早一笔 | 最新活跃一笔 | 逾期的最新最远一笔 credit_type ---------
from scipy.stats import mode
t0 = bureau[['SK_ID_CURR', 'CREDIT_TYPE', 'DAYS_CREDIT']].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'])
t1 = t0.groupby(by='SK_ID_CURR')['CREDIT_TYPE'].apply(lambda x: x.mode()).reset_index().drop_duplicates(subset=['SK_ID_CURR'], keep='first').drop(['level_1'], axis=1).rename(str, columns={'CREDIT_TYPE': 'CREDIT_TYPE_mode'})
t2 = t0.groupby(by='SK_ID_CURR')['CREDIT_TYPE'].agg(['first', 'last']).reset_index().rename(str, columns={'first': 'CREDIT_TYPE_first', 'last': 'CREDIT_TYPE_last'})
credit_type = add_feas(bureau_base, t1)
credit_type = add_feas(credit_type, t2)
del t0, t1, t2
t0 = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Active', ['SK_ID_CURR', 'CREDIT_TYPE', 'DAYS_CREDIT']].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'])
t1 = t0.groupby(by='SK_ID_CURR')['CREDIT_TYPE'].agg(['first', 'last']).reset_index().rename(str, columns={'first': 'CREDIT_TYPE_active_first', 'last': 'CREDIT_TYPE_active_last'})
credit_type = add_feas(credit_type, t1)
del t0, t1
t0 = bureau.loc[(bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT'] < 0), ['SK_ID_CURR', 'CREDIT_TYPE', 'DAYS_CREDIT']].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'])
t1 = t0.groupby(by='SK_ID_CURR')['CREDIT_TYPE'].agg(['first', 'last']).reset_index().rename(str, columns={'first': 'CREDIT_TYPE_dpd_first', 'last': 'CREDIT_TYPE_dpd_last'})
credit_type = add_feas(credit_type, t1)
del t0, t1

# 转换率
app_base = pd.read_hdf('Data_/Application/app_base.hdf')
app_all = credit_type.merge(app_base, on='SK_ID_CURR', how='left').sort_values(by='All_ID')
credit_type_features = [x for x in app_all.columns if x not in ['SK_ID_CURR', 'All_ID', 'Set', 'TARGET']]

# 违约率
from sklearn.utils import shuffle

credit_type_features.insert(0, 'TARGET')
credit_type_features.insert(0, 'All_ID')
train = app_all.loc[app_all['Set'] == 1, credit_type_features]
train = shuffle(train, random_state=123).reset_index(drop=True)
train['chunks_index'] = np.floor(train.index.values / 30750)
test = app_all.loc[app_all['Set'] == 2, credit_type_features]
credit_type_dict = dict()

# 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
for feature in credit_type_features[2:]:
    print(feature)

    new_feature = 'ratio_' + feature
    train[new_feature] = np.nan
    test[new_feature] = np.nan

    feature_cate_kfoldmean = dict()
    for cate in train[feature].unique():
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

    credit_type_dict[feature] = feature_cate_kfoldmean

    train = train.drop([feature], axis=1)
    test = test.drop([feature], axis=1)

credit_type = pd.concat([train, test]).sort_values(by=['All_ID']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)
del train, test

# 类别个数-占比 （基于Train）
credit_type_cols = []
credit_type_features = [x for x in app_all.columns if x not in ['SK_ID_CURR', 'All_ID', 'Set', 'TARGET']]
for feature in credit_type_features:
    print(feature)
    new_feature_1 = 'classnum_' + feature
    new_feature_2 = 'classratio_' + feature
    app_all[new_feature_1] = np.nan
    app_all[new_feature_2] = np.nan
    for cate in app_all[feature].unique():
        app_all.loc[app_all[feature] == cate, new_feature_1] = (app_all.loc[app_all['Set'] == 1, feature] == cate).sum()
        app_all.loc[app_all[feature] == cate, new_feature_2] = (app_all.loc[
                                                                    app_all['Set'] == 1, feature] == cate).sum() / \
                                                               app_all[app_all['Set'] == 1].shape[0]
        credit_type_cols.extend([new_feature_1, new_feature_2])

app_all = app_all.merge(credit_type, on='All_ID', how='left')

# one_hot
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns
app_all, credit_type_ohe = one_hot_encoder(data=app_all, categorical_features=credit_type_features, nan_as_category=True)

app_all = app_all.fillna(0).drop(['All_ID', 'Set', 'TARGET'], axis=1).sort_values(by='SK_ID_CURR').reset_index(drop=True)
bureau_fe = add_feas(bureau_fe, app_all)
print("bureau_fe Shape : ", bureau_fe.shape)

# 是否存在 各贷款类型
num = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE'].count().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'num_of_CREDIT_ACTIVE'})
t0 = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].copy()
CREDIT_TYPE_list = t0['CREDIT_TYPE'].unique()
ff = bureau_base
for typ in CREDIT_TYPE_list:
    print(typ)
    nf = 'CREDIT_TYPE_' + typ
    t1 = t0.loc[t0['CREDIT_TYPE'] == typ].groupby(by='SK_ID_CURR')['CREDIT_TYPE'].count().reset_index().rename(str, columns={'CREDIT_TYPE': nf + '_count'})
    t1 = t1.merge(num, on='SK_ID_CURR', how='right').sort_values(by='SK_ID_CURR').reset_index(drop=True).fillna(0)
    t1[nf + '_ratio'] = t1[nf + '_count'] / t1['num_of_CREDIT_ACTIVE']
    t1[nf + '_flag'] = np.where(t1[nf + '_count'] > 0, 1, 0)
    del t1['num_of_CREDIT_ACTIVE']
    ff = add_feas(ff, t1)
    del t1
bureau_fe = add_feas(bureau_fe, ff)
del ff

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
贷款最新信息距今天数（更新日期）
* stat Max 为最近的一笔更新记录 diff stat 活跃 （实还日期=nan 排除）
* 应还/申请 stat 活跃 
* 实还/申请 stat 
* （更新-应还）/（更新-申请）stat
* （更新-实还）/（更新-申请）stat
'''
t0 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT_UPDATE']]
t0['DAYS_CREDIT_UPDATE_pos'] = -t0['DAYS_CREDIT_UPDATE']
del t0['DAYS_CREDIT_UPDATE']
t1 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE_pos', base=bureau_base)
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1

t0 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT_UPDATE']].sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT_UPDATE'], ascending=[True, False])
t0['DAYS_CREDIT_UPDATE_diff'] = t0.sort_values(by='DAYS_CREDIT_UPDATE').groupby(by='SK_ID_CURR')['DAYS_CREDIT_UPDATE'].diff(-1)
t0.loc[t0['DAYS_CREDIT_UPDATE_diff'].isnull(), 'DAYS_CREDIT_UPDATE_diff'] = t0['DAYS_CREDIT_UPDATE']
t1 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE_diff', base=bureau_base)
bureau_fe = add_feas(bureau_fe, t1)
del t0, t1

t0 = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']].copy()
t0['DAYS_CREDIT_UPDATE-DAYS_CREDIT_ENDDATE'] = t0['DAYS_CREDIT_UPDATE'] - t0['DAYS_CREDIT_ENDDATE']
t0['DAYS_CREDIT_UPDATE-DAYS_ENDDATE_FACT'] = t0['DAYS_CREDIT_UPDATE'] - t0['DAYS_ENDDATE_FACT']
t0['DAYS_CREDIT_UPDATE-DAYS_CREDIT_ENDDATE/DAYS_CREDIT_UPDATE-DAYS_CREDIT'] = t0['DAYS_CREDIT_UPDATE-DAYS_CREDIT_ENDDATE'] / (t0['DAYS_CREDIT_UPDATE'] - t0['DAYS_CREDIT'])
t0['DAYS_CREDIT_UPDATE-DAYS_ENDDATE_FACT/DAYS_CREDIT_UPDATE-DAYS_CREDIT'] = t0['DAYS_CREDIT_UPDATE-DAYS_ENDDATE_FACT'] / (t0['DAYS_CREDIT_UPDATE'] - t0['DAYS_CREDIT'])
t0['DAYS_CREDIT_ENDDATE/DAYS_CREDIT'] = t0['DAYS_CREDIT_ENDDATE'] / t0['DAYS_CREDIT']
t0['DAYS_ENDDATE_FACT/DAYS_CREDIT'] = t0['DAYS_ENDDATE_FACT'] / t0['DAYS_CREDIT']
t1 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE-DAYS_CREDIT_ENDDATE', base=bureau_base)
t2 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE-DAYS_ENDDATE_FACT', base=bureau_base)
t3 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE-DAYS_CREDIT_ENDDATE/DAYS_CREDIT_UPDATE-DAYS_CREDIT', base=bureau_base)
t4 = stat_simple(data=t0, x='DAYS_CREDIT_UPDATE-DAYS_ENDDATE_FACT/DAYS_CREDIT_UPDATE-DAYS_CREDIT', base=bureau_base)
t5 = stat_simple(data=t0, x='DAYS_CREDIT_ENDDATE/DAYS_CREDIT', base=bureau_base)
t6 = stat_simple(data=t0, x='DAYS_ENDDATE_FACT/DAYS_CREDIT', base=bureau_base)
bureau_fe = add_feas(bureau_fe, t1)
bureau_fe = add_feas(bureau_fe, t2)
bureau_fe = add_feas(bureau_fe, t3)
bureau_fe = add_feas(bureau_fe, t4)
bureau_fe = add_feas(bureau_fe, t5)
bureau_fe = add_feas(bureau_fe, t6)
del t0, t1, t2, t3, t4, t5, t6

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

'''
AMT_CREDIT_MAX_OVERDUE AMT_CREDIT_SUM AMT_CREDIT_SUM_DEBT 
AMT_CREDIT_SUM_LIMIT AMT_CREDIT_SUM_OVERDUE AMT_ANNUITY
距今最大逾期金额 （跨表）
当前授信金额（授信总额）（跨表）
当前负债金额（支用额）
当前信用卡额度（授信限额）
当前逾期金额
信贷年金（每期偿还金额）
* stat / 活跃 关闭 售出 | A/B A-B stat (归一化？)| 最新一笔 最老一笔 | 活跃| 
* 距今申请天数在30/90/180/360天内的最大逾期金额 stat | 
* 应还日期在申请日期前/后N天 (Todo)
* 除以该贷款类型下均值 'Consumer credit', 'Credit card'
(距今最大逾期金额/当前授信金额/当前负债金额/当前信用卡额度/当前逾期金额) 【跨表】
'''
def div(data, x, y):
    nf = x + '/' + y
    nf1 = x + '-' + y
    data[nf] = data[x].fillna(0) / (data[y].fillna(0)+0.0001)
    data[nf1] = data[x].fillna(0) - (data[y].fillna(0))
    return data

feas = ['SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']
to_feas = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']

# stat
t0 = bureau.copy()
for x in to_feas:
    t1 = stat(data=t0, x=x, base=bureau_base)
    bureau_fe = add_feas(bureau_fe, t1)
    del t1
del t0

t0 = bureau.copy()
originals = [x for x in t0.columns if x not in ['SK_ID_CURR', 'CREDIT_ACTIVE']]
for x in to_feas:
    for y in [y for y in to_feas if y != x]:
        t0 = div(data=t0, x=x, y=y)
t0 = t0.drop(originals, axis=1)
for x in [x for x in t0.columns if x not in ['SK_ID_CURR', 'CREDIT_ACTIVE']]:
    t1 = stat_simple(data=t0, x=x, base=bureau_base)
    bureau_fe = add_feas(bureau_fe, t1)
    del t1
del t0

for x in to_feas:
    t0 = bureau.copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset=['SK_ID_CURR'], keep='first').loc[:, ['SK_ID_CURR', x]].rename(str, columns={x: x + '_first'})
    bureau_fe = add_feas(bureau_fe, t0)
    del t0
for x in to_feas:
    t0 = bureau.copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'], ).drop_duplicates(subset=['SK_ID_CURR'], keep='last').loc[:, ['SK_ID_CURR', x]].rename(str, columns={x: x + '_last'})
    bureau_fe = add_feas(bureau_fe, t0)
    del t0

for x in to_feas:
    t0 = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT']).drop_duplicates(subset=['SK_ID_CURR'], keep='first').loc[:, ['SK_ID_CURR', x]].rename(str, columns={x: x + '_active_first'})
    bureau_fe = add_feas(bureau_fe, t0)
    del t0
for x in to_feas:
    t0 = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy().sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'], ).drop_duplicates(subset=['SK_ID_CURR'], keep='last').loc[:, ['SK_ID_CURR', x]].rename(str, columns={x: x + '_active_last'})
    bureau_fe = add_feas(bureau_fe, t0)
    del t0

t0 = bureau[['SK_ID_CURR', 'DAYS_CREDIT', 'AMT_CREDIT_SUM_OVERDUE']].copy()
t0['DAYS_CREDIT_pos'] = -t0['DAYS_CREDIT']
del t0['DAYS_CREDIT']
for N in [30, 90, 180, 360, 540, 720]:
    nf = 'AMT_CREDIT_SUM_OVERDUE_max_less' + str(N)
    nf1 = 'AMT_CREDIT_SUM_OVERDUE_sum_less' + str(N)
    nf2 = 'AMT_CREDIT_SUM_OVERDUE_mean_less' + str(N)
    t1 = t0.loc[t0['DAYS_CREDIT_pos'] <= N, ].groupby(by='SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].agg(['max', 'sum', 'mean']).reset_index().rename(str, columns={'max': nf, 'sum': nf1, 'mean': nf2})
    bureau_fe = add_feas(bureau_fe, t1)
    del t1
del t0

t0 = bureau.copy()
t1 = t0.groupby(by='CREDIT_TYPE')[to_feas].mean().reset_index().fillna(0)
for x in to_feas:
    for tp in ['Consumer credit', 'Credit card']:
        nf = x + '_mean_ratio_' + tp
        t2 = t0.loc[t0['CREDIT_TYPE'] == tp, ['SK_ID_CURR', x]].groupby(by='SK_ID_CURR')[x].mean().reset_index()
        t2['type_mean'] = np.repeat(t1.loc[t1['CREDIT_TYPE'] == tp, x].values+1, t2.shape[0])
        t2[nf] = t2[x] / t2['type_mean']
        del t2[x], t2['type_mean']
        bureau_fe = add_feas(bureau_fe, t2)
        del t2

bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)

# Features Selection
for feature in bureau_fe.columns:
    t1 = re.match(".*(_x)", feature)
    t2 = re.match(".*(_y)", feature)
    if (type(t1) != 'NoneType') | (type(t2) != 'NoneType'):
        print(feature)


# Write HDF
bureau_fe = bureau_fe.fillna(0)
print("bureau_fe Shape : ", bureau_fe.shape)  # (305811, 1575)
bureau_fe.to_hdf('Data_/Bureau/bureau.hdf', 'bureau', mode='w')










