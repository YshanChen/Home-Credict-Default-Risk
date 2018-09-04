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
def add_feas(base, feature, on='SK_ID_CURR', how='left'):
    base = base.merge(feature, on=on, how=how)
    del feature
    return base
def stat_simple(data, x, groupby, nf_base, agg_list=['max', 'min', 'mean', 'sum', 'std']):
    t1 = data[['SK_ID_PREV', 'SK_ID_CURR', x]].copy()
    t2 = t1.groupby(by=groupby)[x].agg(agg_list).reset_index().rename(str, columns={'max': nf_base+'_max', 'min': nf_base+'_min', 'mean': nf_base+'_mean', 'sum': nf_base+'_sum', 'std': nf_base+'_std'})
    return t2

# Read Data
pos = pd.read_csv("Data/POS_CASH_balance.csv").sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
pos.loc[(pos['NAME_CONTRACT_STATUS'] == 'Signed') | (pos['NAME_CONTRACT_STATUS'] == 'Returned to the store') | (pos['NAME_CONTRACT_STATUS'] == 'Approved'), 'NAME_CONTRACT_STATUS'] = 'Active'
pos = pos.loc[(pos['NAME_CONTRACT_STATUS'] != 'Canceled') & (pos['NAME_CONTRACT_STATUS'] != 'XNA')]

print("POS_CASH_balance Shape : ", pos.shape) # POS_CASH_balance Shape :  (10001358, 8)
pos_base = pos[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(subset=['SK_ID_PREV'], keep='first')
pos_fe = pos_base[['SK_ID_CURR']].drop_duplicates(subset=['SK_ID_CURR'], keep='first')

t0 = pos.copy()
# 标记 Completed1  Demand2  Amortized3  (Demand | Amortized)
t0['pos_status_t'] = np.where(t0['NAME_CONTRACT_STATUS'] == 'Active', 0,
                              np.where(t0['NAME_CONTRACT_STATUS'] == 'Completed', 1,
                                       np.where(t0['NAME_CONTRACT_STATUS'] == 'Demand', 2, 3)))
t1 = t0.groupby(by='SK_ID_PREV')['pos_status_t'].max().reset_index().rename(str, columns={'pos_status_t': 'pos_status'})
t0 = add_feas(t0, t1, on='SK_ID_PREV')
del t0['pos_status_t'], t1

def merge_to_CURR(tdata, x, pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe):
    t1 = tdata[['SK_ID_PREV', x]]
    t2 = add_feas(pos_base, t1, on='SK_ID_PREV')
    # t2 = t2.fillna(pos_base_fillna)

    for agg in agg_list:
        nf = x + '_' + agg
        print(nf)
        t3 = t2.groupby(by='SK_ID_CURR')[x].agg([agg]).reset_index().rename(str, columns={agg: nf})
        pos_fe_ = add_feas(pos_fe_, t3, on='SK_ID_CURR')
        # pos_fe_ = pos_fe_.fillna(pos_fe_base_fillna)
        del nf, t3

    return pos_fe_


# array(['Active', 'Completed', 'Signed=>Active', 'Returned to the store=>Active',
#        'Approved=>Active', 'Demand'核销, 'Amortized debt'摊平, 'Canceled'取消, 'XNA'],
#       dtype=object)

# 衍生变量 ————————————————————————————————————————————————
# 1. 笔数 未结清笔数 结清笔数 (Demand | Amortized)笔数 /占比 SK_ID_PREV stat (337252, 2) (181719, 2)
t1 = t0.drop_duplicates(subset=['SK_ID_PREV'], keep='first')
t2 = t1.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'pos_number'})
t3 = t1[t1['pos_status'] == 0].drop_duplicates(subset=['SK_ID_PREV'], keep='first')
t4 = t3.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'pos_active_number'})
t5 = t1[t1['pos_status'] == 1].drop_duplicates(subset=['SK_ID_PREV'], keep='first')
t6 = t5.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'pos_complete_number'})
t7 = t1[(t1['pos_status'] == 2) | (t1['pos_status'] == 3)].drop_duplicates(subset=['SK_ID_PREV'], keep='first') #  (360, 2)
t8 = t7.groupby(by='SK_ID_CURR')['SK_ID_PREV'].count().reset_index().rename(str, columns={'SK_ID_PREV': 'pos_demand_debt_number'})

pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
pos_fe = add_feas(pos_fe, t4, on='SK_ID_CURR')
pos_fe = add_feas(pos_fe, t6, on='SK_ID_CURR')
pos_fe = add_feas(pos_fe, t8, on='SK_ID_CURR')
pos_fe = pos_fe.fillna(0)
pos_fe['pos_active_ratio'] = pos_fe['pos_active_number'] / pos_fe['pos_number']
pos_fe['pos_complete_ratio'] = pos_fe['pos_complete_number'] / pos_fe['pos_number']
pos_fe['pos_demand_debt_ratio'] = pos_fe['pos_demand_debt_number'] / pos_fe['pos_number']
del t1, t2, t3, t4, t5, t6, t7, t8

pos_fe = pos_fe.fillna(0)

# 2. 针对SK_ID_PREV  Months_balance 个数 stat  =》 stat (未结清)
t1 = t0.groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_count'})
pos_fe = merge_to_CURR(tdata=t1, x='Months_balance_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1
t1 = t0[t0['pos_status'] == 0].groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_active_count'})
pos_fe = merge_to_CURR(tdata=t1, x='Months_balance_active_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1

pos_fe['Months_balance_active_count_max/Months_balance_count_max'] = np.where(pos_fe['Months_balance_count_max'] == 0, 0, pos_fe['Months_balance_active_count_max'] / pos_fe['Months_balance_count_max'])
pos_fe['Months_balance_active_count_min/Months_balance_count_min'] = np.where(pos_fe['Months_balance_count_min'] == 0, 0, pos_fe['Months_balance_active_count_min'] / pos_fe['Months_balance_count_min'])
pos_fe['Months_balance_active_count_mean/Months_balance_count_mean'] = np.where(pos_fe['Months_balance_count_mean'] == 0, 0, pos_fe['Months_balance_active_count_mean'] / pos_fe['Months_balance_count_mean'])
pos_fe['Months_balance_active_count_sum/Months_balance_count_sum'] = np.where(pos_fe['Months_balance_count_sum'] == 0, 0, pos_fe['Months_balance_active_count_sum'] / pos_fe['Months_balance_count_sum'])

# 3. 针对SK_ID_PREV  Months_balance 最早时间 最晚时间 stat  =》 stat (未结清)
t1 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t1['MONTHS_BALANCE'] = -t1['MONTHS_BALANCE']
t2 = stat_simple(data=t1, x='MONTHS_BALANCE', groupby='SK_ID_CURR', nf_base='Months_balance_first', agg_list=['max', 'min', 'mean', 'sum', 'std'])
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2

t1 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, True, False]).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t1['MONTHS_BALANCE'] = -t1['MONTHS_BALANCE']
t2 = stat_simple(data=t1, x='MONTHS_BALANCE', groupby='SK_ID_CURR', nf_base='Months_balance_nearest', agg_list=['max', 'min', 'mean', 'sum', 'std'])
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2

t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t1['MONTHS_BALANCE'] = -t1['MONTHS_BALANCE']
t2 = stat_simple(data=t1, x='MONTHS_BALANCE', groupby='SK_ID_CURR', nf_base='Months_balance_first_active', agg_list=['max', 'min', 'mean', 'sum', 'std'])
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2

t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, True, False]).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t1['MONTHS_BALANCE'] = -t1['MONTHS_BALANCE']
t2 = stat_simple(data=t1, x='MONTHS_BALANCE', groupby='SK_ID_CURR', nf_base='Months_balance_nearest_active', agg_list=['max', 'min', 'mean', 'sum', 'std'])
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2

# 4. 针对SK_ID_PREV  Months_balance 最早期数-最晚时间/最早期数（as 5/57） stat  =》 stat  (未结清)
t1 = t0[['SK_ID_PREV', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t1['MONTHS_BALANCE_first'] = t1['MONTHS_BALANCE']
t2 = t0[['SK_ID_PREV', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, False]).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True)
t2['MONTHS_BALANCE_nearest'] = t2['MONTHS_BALANCE']
t3 = add_feas(pos_base, t1, on='SK_ID_PREV')
t3 = add_feas(t3, t2, on='SK_ID_PREV')
t3['MONTHS_BALANCE_nearest-MONTHS_BALANCE_first'] = t3['MONTHS_BALANCE_nearest'] - t3['MONTHS_BALANCE_first']
t3['MONTHS_BALANCE_nearest-MONTHS_BALANCE_first/MONTHS_BALANCE_first'] = np.where(t3['MONTHS_BALANCE_first'] == 0, 0, t3['MONTHS_BALANCE_nearest-MONTHS_BALANCE_first']/(-t3['MONTHS_BALANCE_first']))

pos_fe = merge_to_CURR(tdata=t3, x='MONTHS_BALANCE_nearest-MONTHS_BALANCE_first', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(tdata=t3, x='MONTHS_BALANCE_nearest-MONTHS_BALANCE_first/MONTHS_BALANCE_first', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3

# 5. 针对SK_ID_PREV  CNT_INSTALMENT  计划分期总期数 最早时间那笔 stat =》 stat (未结清)
t1 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'})
t2 = stat_simple(data=t1, x='original_CNT_INSTALMENT', groupby='SK_ID_CURR', nf_base='original_CNT_INSTALMENT')
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2

t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT_active'})
t2 = stat_simple(data=t1, x='original_CNT_INSTALMENT_active', groupby='SK_ID_CURR', nf_base='original_CNT_INSTALMENT_active')
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2


# 6. 针对SK_ID_PREV  CNT_INSTALMENT  实际分期期 Complete那笔 （没有complete则为nan） stat =》 stat
t1 = t0.loc[t0['NAME_CONTRACT_STATUS'] == 'Completed', ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT']].rename(str, columns={'CNT_INSTALMENT': 'completed_CNT_INSTALMENT'})
t2 = stat_simple(data=t1, x='completed_CNT_INSTALMENT', groupby='SK_ID_CURR', nf_base='completed_CNT_INSTALMENT')
pos_fe = add_feas(pos_fe, t2, on='SK_ID_CURR')
del t1, t2


# 7. 针对SK_ID_PREV  CNT_INSTALMENT  计划分期数-实际分期数（提前几期结清）  计划分期数-实际分期数/计划分期数 stat/Flag（是否提前结清） =》 stat
t1 = t0[['SK_ID_PREV', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'}).drop(['MONTHS_BALANCE'], axis=1)
t2 = t0.loc[t0['NAME_CONTRACT_STATUS'] == 'Completed', ['SK_ID_PREV', 'CNT_INSTALMENT']].rename(str, columns={'CNT_INSTALMENT': 'completed_CNT_INSTALMENT'}).sort_values(by=['SK_ID_PREV'])
t3 = add_feas(pos_base, t1, on='SK_ID_PREV')
t3 = add_feas(t3, t2, on='SK_ID_PREV')
t3['early_completed_months'] = t3['original_CNT_INSTALMENT'] - t3['completed_CNT_INSTALMENT']
t3['early_completed_months_ratio'] = t3['early_completed_months'] / t3['original_CNT_INSTALMENT']
t3['early_completed_months_flag'] = np.where(t3['early_completed_months'] > 0, 1, 0)

t4 = stat_simple(data=t3, x='early_completed_months', groupby='SK_ID_CURR', nf_base='early_completed_months', agg_list=['max', 'min', 'mean', 'sum', 'std'])
t5 = stat_simple(data=t3, x='early_completed_months_ratio', groupby='SK_ID_CURR', nf_base='early_completed_months_ratio', agg_list=['max', 'min', 'mean', 'sum', 'std'])
t6 = stat_simple(data=t3, x='early_completed_months_flag', groupby='SK_ID_CURR', nf_base='early_completed_months_flag', agg_list=['max', 'min', 'mean', 'sum', 'std'])
pos_fe = add_feas(pos_fe, t4, on='SK_ID_CURR')
pos_fe = add_feas(pos_fe, t5, on='SK_ID_CURR')
pos_fe = add_feas(pos_fe, t6, on='SK_ID_CURR')
del t1, t2, t3, t4, t5, t6


# 8. 针对SK_ID_PREV  CNT_INSTALMENT_FUTURE  min剩余期数 stat/Flag（>0 未还清） =》 stat
t1 = t0.groupby(by='SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].min().reset_index().rename(str, columns={'CNT_INSTALMENT_FUTURE': 'CNT_INSTALMENT_FUTURE_min'})
t2 = add_feas(pos_base, t1, on='SK_ID_PREV')
t3 = stat_simple(data=t2, x='CNT_INSTALMENT_FUTURE_min', groupby='SK_ID_CURR', nf_base='CNT_INSTALMENT_FUTURE_min', agg_list=['max', 'min', 'mean', 'sum', 'std'])
t3['CNT_INSTALMENT_FUTURE_min_max_bigger0_flag'] = np.where(t3['CNT_INSTALMENT_FUTURE_min_max'] > 0, 1, 0)
pos_fe = add_feas(pos_fe, t3, on='SK_ID_CURR')
del t1, t2, t3

# 9. 针对SK_ID_PREV  NAME_CONTRACT_STATUS （针对未结清）计划分期总期数-（FUTURE！=0）期数和 /计划分期总期数  stat  =》 stat
t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT_active'})
t1 = t1[['SK_ID_PREV', 'original_CNT_INSTALMENT_active']]
# 包含了结清的
t2 = t0.loc[(t0['CNT_INSTALMENT_FUTURE'] != 0) & (~(t0['CNT_INSTALMENT_FUTURE'].isna())), ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT_FUTURE', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
t3 = t2.groupby(by='SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].count().reset_index().rename(str, columns={'CNT_INSTALMENT_FUTURE': 'CNT_INSTALMENT_FUTURE_count_active'})
t4 = add_feas(pos_base, t1, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
# 排除结清
t4 = t4[~t4['original_CNT_INSTALMENT_active'].isna()]
t4['original_CNT_INSTALMENT_active-CNT_INSTALMENT_FUTURE_count_active'] = t4['original_CNT_INSTALMENT_active'] - t4['CNT_INSTALMENT_FUTURE_count_active']
t4['CNT_INSTALMENT_FUTURE_count_active_ratio'] = t4['original_CNT_INSTALMENT_active-CNT_INSTALMENT_FUTURE_count_active'] / t4['original_CNT_INSTALMENT_active']

pos_fe = merge_to_CURR(t4, x='original_CNT_INSTALMENT_active-CNT_INSTALMENT_FUTURE_count_active', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t4, x='CNT_INSTALMENT_FUTURE_count_active_ratio', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

# 9. 针对SK_ID_PREV  NAME_CONTRACT_STATUS （针对未结清）计划分期总期数-Active期数和 /计划分期总期数  stat  =》 stat
t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT_active'})
t1 = t1[['SK_ID_PREV', 'original_CNT_INSTALMENT_active']]
# 包含了结清的
t2 = t0.loc[t0['NAME_CONTRACT_STATUS'] == 'Active', ['SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
t3 = t2.groupby(by='SK_ID_PREV')['NAME_CONTRACT_STATUS'].count().reset_index().rename(str, columns={'NAME_CONTRACT_STATUS': 'months_count_active'})
t4 = add_feas(pos_base, t1, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
# 排除结清
t4 = t4[~t4['original_CNT_INSTALMENT_active'].isna()]
t4['original_CNT_INSTALMENT_active-months_count_active'] = t4['original_CNT_INSTALMENT_active'] - t4['months_count_active']
t4['months_count_active_ratio'] = t4['original_CNT_INSTALMENT_active-months_count_active'] / t4['original_CNT_INSTALMENT_active']

pos_fe = merge_to_CURR(t4, x='original_CNT_INSTALMENT_active-months_count_active', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t4, x='months_count_active_ratio', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

# 10. 针对SK_ID_PREV  SK_DPD SK_DPD_DEF stat  =》 stat (未结清)
t1 = stat_simple(data=t0, x='SK_DPD', groupby='SK_ID_PREV', nf_base='SK_DPD')
pos_fe = merge_to_CURR(t1, x='SK_DPD_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1

t1 = stat_simple(data=t0, x='SK_DPD_DEF', groupby='SK_ID_PREV', nf_base='SK_DPD_DEF')
pos_fe = merge_to_CURR(t1, x='SK_DPD_DEF_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_DEF_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_DEF_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_DEF_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t1, x='SK_DPD_DEF_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1

t1 = t0[t0['pos_status'] == 0].copy()
t2 = stat_simple(data=t1, x='SK_DPD', groupby='SK_ID_PREV', nf_base='SK_DPD_active')
pos_fe = merge_to_CURR(t2, x='SK_DPD_active_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_active_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_active_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_active_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_active_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2

t1 = t0[t0['pos_status'] == 0].copy()
t2 = stat_simple(data=t1, x='SK_DPD_DEF', groupby='SK_ID_PREV', nf_base='SK_DPD_DEF_active')
pos_fe = merge_to_CURR(t2, x='SK_DPD_DEF_active_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_DEF_active_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_DEF_active_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_DEF_active_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_DEF_active_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2

# 11. 针对SK_ID_PREV  SK_DPD - SK_DPD_DEF stat  =》 stat (未结清)
t1 = t0.copy()
t1['SK_DPD_diff'] = t1['SK_DPD'] - t1['SK_DPD_DEF']
t2 = stat_simple(data=t1, x='SK_DPD_diff', groupby='SK_ID_PREV', nf_base='SK_DPD_diff')
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2

t1 = t0[t0['pos_status'] == 0].copy()
t1['SK_DPD_diff'] = t1['SK_DPD'] - t1['SK_DPD_DEF']
t2 = stat_simple(data=t1, x='SK_DPD_DEF', groupby='SK_ID_PREV', nf_base='SK_DPD_diff_active')
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_active_max', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_active_min', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_active_mean', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_active_sum', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t2, x='SK_DPD_diff_active_std', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2

# 12. 针对SK_ID_PREV  SK_DPD 逾期 flag =》 flag (未结清)
t1 = t0[['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].max().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_max'})
t3 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = t3.groupby(by='SK_ID_CURR')['SK_DPD_max'].max().reset_index()
t4['SK_DPD_flag'] = np.where(t4['SK_DPD_max'] > 0, 1, 0)
del t4['SK_DPD_max']
pos_fe = add_feas(pos_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].max().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_max_active'})
t3 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = t3.groupby(by='SK_ID_CURR')['SK_DPD_max_active'].max().reset_index()
t4['SK_DPD_active_flag'] = np.where(t4['SK_DPD_max_active'] > 0, 1, 0)
del t4['SK_DPD_max_active']
pos_fe = add_feas(pos_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# 13. 针对SK_ID_PREV  SK_DPD SK_DPD_DEF 逾期次数（SK_DPD>0 count） /Months_balance个数 /计划分期总期数 =》 stat (未结清)
t1 = t0.loc[t0['SK_DPD'] > 0, ['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].count().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_count'})
t3 = t0.groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_count'})
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t4 = t4.fillna(0)
t4['SK_DPD_count_ratio'] = t4['SK_DPD_count'] / t4['Months_balance_count']
pos_fe = merge_to_CURR(t4, x='SK_DPD_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t4, x='SK_DPD_count_ratio', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

t1 = t0.loc[t0['SK_DPD'] > 0, ['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].count().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_count'})
t3 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'})
t3 = t3[['SK_ID_PREV', 'original_CNT_INSTALMENT']]
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t4 = t4.fillna(0)
t4['SK_DPD_count_ratio_original_CNT_INSTALMENT'] = np.where(t4['original_CNT_INSTALMENT'] == 0, 0, t4['SK_DPD_count'] / t4['original_CNT_INSTALMENT'])
pos_fe = merge_to_CURR(t4, x='SK_DPD_count_ratio_original_CNT_INSTALMENT', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

t1 = t0.loc[(t0['SK_DPD'] > 0) & (t0['pos_status'] == 0), ['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].count().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_count_active'})
t3 = t0.groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_count'})
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t4 = t4.fillna(0)
t4['SK_DPD_count_active_ratio'] = t4['SK_DPD_count_active'] / t4['Months_balance_count']
pos_fe = merge_to_CURR(t4, x='SK_DPD_count_active', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t4, x='SK_DPD_count_active_ratio', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

t1 = t0.loc[(t0['SK_DPD'] > 0) & (t0['pos_status'] == 0), ['SK_ID_PREV', 'SK_DPD']]
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD'].count().reset_index().rename(str, columns={'SK_DPD': 'SK_DPD_count_active'})
t3 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'})
t3 = t3[['SK_ID_PREV', 'original_CNT_INSTALMENT']]
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t4 = t4.fillna(0)
t4['SK_DPD_active_ratio_original_CNT_INSTALMENT'] = np.where(t4['original_CNT_INSTALMENT'] == 0, 0, t4['SK_DPD_count_active'] / t4['original_CNT_INSTALMENT'])
pos_fe = merge_to_CURR(t4, x='SK_DPD_active_ratio_original_CNT_INSTALMENT', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4

# 14. 针对SK_ID_PREV  SK_DPD 逾期次数（连续逾期算为1次） /Months_balance个数 /计划分期总期数 =》 stat
t1 = t0[['SK_ID_PREV', 'SK_DPD']].copy()
t1['SK_DPD_t'] = np.where(t1['SK_DPD'] > 0, 1, 0)
t1['SK_DPD_diff'] = t1.groupby(by='SK_ID_PREV')['SK_DPD_t'].diff()
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'SK_DPD_diff_t'})
t2['continue_DPD_count'] = np.ceil(t2['SK_DPD_diff_t'] / 2)
del t2['SK_DPD_diff_t']
t3 = t0.groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_count'})
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t5 = t0[['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'})
t5 = t5[['SK_ID_PREV', 'original_CNT_INSTALMENT']]
t6 = add_feas(t4, t5, on='SK_ID_PREV')
t6 = t6.fillna(0)
t6['continue_DPD_ratio_Months_balance_count'] = np.where(t6['Months_balance_count'] == 0, 0, t6['continue_DPD_count'] / t6['Months_balance_count'])
t6['continue_DPD_ratio_original_CNT_INSTALMENT'] = np.where(t6['original_CNT_INSTALMENT'] == 0, 0, t6['continue_DPD_count'] / t6['original_CNT_INSTALMENT'])
pos_fe = merge_to_CURR(t6, x='continue_DPD_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t6, x='continue_DPD_ratio_Months_balance_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t6, x='continue_DPD_ratio_original_CNT_INSTALMENT', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4, t5, t6

t1 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_DPD']].copy()
t1['SK_DPD_t'] = np.where(t1['SK_DPD'] > 0, 1, 0)
t1['SK_DPD_diff'] = t1.groupby(by='SK_ID_PREV')['SK_DPD_t'].diff()
t2 = t1.groupby(by='SK_ID_PREV')['SK_DPD_diff'].agg([lambda x: np.abs(x).sum()]).reset_index().rename(str, columns={"<lambda>": 'SK_DPD_diff_t'})
t2['continue_DPD_count_active'] = np.ceil(t2['SK_DPD_diff_t'] / 2)
del t2['SK_DPD_diff_t']
t3 = t0[t0['pos_status'] == 0].groupby(by='SK_ID_PREV')['MONTHS_BALANCE'].count().reset_index().rename(str, columns={'MONTHS_BALANCE': 'Months_balance_count'})
t4 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = add_feas(t4, t3, on='SK_ID_PREV')
t5 = t0.loc[t0['pos_status'] == 0, ['SK_ID_PREV', 'SK_ID_CURR', 'CNT_INSTALMENT', 'MONTHS_BALANCE']].sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).drop_duplicates(subset=['SK_ID_PREV'], keep='first').reset_index(drop=True).rename(str, columns={'CNT_INSTALMENT': 'original_CNT_INSTALMENT'})
t5 = t5[['SK_ID_PREV', 'original_CNT_INSTALMENT']]
t6 = add_feas(t4, t5, on='SK_ID_PREV')
t6 = t6.fillna(0)
t6['continue_DPD_ratio_active_Months_balance_count'] = np.where(t6['Months_balance_count'] == 0, 0, t6['continue_DPD_count_active'] / t6['Months_balance_count'])
t6['continue_DPD_ratio_active_original_CNT_INSTALMENT'] = np.where(t6['original_CNT_INSTALMENT'] == 0, 0, t6['continue_DPD_count_active'] / t6['original_CNT_INSTALMENT'])
pos_fe = merge_to_CURR(t6, x='continue_DPD_count_active', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t6, x='continue_DPD_ratio_active_Months_balance_count', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
pos_fe = merge_to_CURR(t6, x='continue_DPD_ratio_active_original_CNT_INSTALMENT', pos_base=pos_base, agg_list=['max', 'min', 'mean', 'sum', 'std'], pos_fe_=pos_fe)
del t1, t2, t3, t4, t5, t6

# 15. 针对SK_ID_PREV  NAME_CONTRACT_STATUS flag demand =》 flag
t1 = t0[['SK_ID_PREV', 'pos_status']].copy()
t1['demand_flag'] = np.where(t1['pos_status'] == 2, 1, 0)
t1['debt_flag'] = np.where(t1['pos_status'] == 3, 1, 0)
t1['demand_debt_flag'] = np.where((t1['pos_status'] == 2) | (t1['pos_status'] == 3), 1, 0)
t2 = t1.groupby(by='SK_ID_PREV')['demand_flag', 'debt_flag', 'demand_debt_flag'].max().reset_index()
t3 = add_feas(pos_base, t2, on='SK_ID_PREV')
t4 = t3.groupby(by='SK_ID_CURR')['demand_flag', 'debt_flag', 'demand_debt_flag'].max().reset_index()
pos_fe = add_feas(pos_fe, t4, on='SK_ID_CURR')
del t1, t2, t3, t4

# to_hdf
pos_fe = pos_fe.fillna(-999)
print("pos_fe : ", pos_fe.shape[1] - 1)
pos_fe.to_hdf('Data_/Pos/POS_CASH_balance.hdf', 'POS_CASH_balance', mode='w', format='table')

