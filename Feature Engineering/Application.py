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
def rank_feature(data, x):
    new_feature = 'Rank_' + x
    data[new_feature] = stats.rankdata(np.multiply(-1, data[x]))
    return data



def get_application_toHDF():
    # ============================= Read Data =================================
    app_train = pd.read_csv("Data/application_train.csv")
    app_test = pd.read_csv("Data/application_test.csv")
    print('TARGET Class: \n', app_train.TARGET.value_counts()) # 0:282686 1:24825 rate:0.0807

    app_train['Set'] = 1
    app_test['Set'] = 2

    # 异常值
    app_test.loc[app_test['REGION_RATING_CLIENT_W_CITY'] == -1, 'REGION_RATING_CLIENT_W_CITY'] = 2
    app_train = app_train.loc[app_train['CODE_GENDER'] != 'XNA']

    # concat
    app_train_test = pd.concat([app_train, app_test])
    app_train_test['All_ID'] = np.arange(app_train_test.shape[0])
    original_feas = ['All_ID'] + app_train.columns.tolist()
    app_train_test = app_train_test[original_feas]
    cust_cols = ['All_ID', 'Set', 'SK_ID_CURR', 'TARGET']

    # 异常值
    app_train_test = app_train_test[app_train_test['AMT_INCOME_TOTAL'] < 6000000]
    app_train_test = app_train_test.loc[(app_train_test['Set'] == 2) | ((app_train_test['Set'] == 1) & ((app_train_test['OBS_30_CNT_SOCIAL_CIRCLE'].isnull()) | (app_train_test['OBS_30_CNT_SOCIAL_CIRCLE'] <= 50)))]
    app_train_test = app_train_test.loc[(app_train_test['Set'] == 2) | ((app_train_test['Set'] == 1) & ((app_train_test['AMT_REQ_CREDIT_BUREAU_QRT'].isnull()) | (app_train_test['AMT_REQ_CREDIT_BUREAU_QRT'] <= 200)))]

    del app_train, app_test, original_feas

    # ============================= Clear Data =================================
    #  时长特征
    app_train_test['DAYS_BIRTH'] = -(app_train_test['DAYS_BIRTH'] / 365)
    app_train_test['DAYS_EMPLOYED'] = -(app_train_test['DAYS_EMPLOYED'] / 30)
    app_train_test['DAYS_REGISTRATION'] = -(app_train_test['DAYS_REGISTRATION'] / 30)
    app_train_test['DAYS_ID_PUBLISH'] = -(app_train_test['DAYS_ID_PUBLISH'] / 30)
    app_train_test['DAYS_LAST_PHONE_CHANGE'] = -(app_train_test['DAYS_LAST_PHONE_CHANGE'] / 30)
    app_train_test.loc[app_train_test['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED'] = np.nan

    # 缺失率统计（样本层级） ---------------------
    miss_table = app_train_test.copy()
    miss_table = miss_table.drop(['Set', 'SK_ID_CURR', 'TARGET'], axis=1)
    raw_predict_features = [feature for feature in miss_table.columns.tolist() if feature != 'All_ID']
    print("Original Features Number : ", len(raw_predict_features))
    for feature in raw_predict_features:
        new_feature_missflag = feature + '_missflag'
        miss_table[new_feature_missflag] = np.where(app_train_test[feature].isnull(), 1, 0)
        del miss_table[feature]
    miss_table['miss_num_raw_predict_features_120'] = miss_table.iloc[:, 1:miss_table.shape[1]].sum(axis=1)
    miss_table['miss_num_building_info'] = miss_table.loc[:, "APARTMENTS_AVG_missflag":"EMERGENCYSTATE_MODE_missflag"].sum(axis=1)
    miss_table['miss_num_FLAG_DOCUMENT'] = miss_table.loc[:, "FLAG_DOCUMENT_2_missflag":"FLAG_DOCUMENT_21_missflag"].sum(axis=1)
    miss_table['miss_ratio_raw_predict_features_120'] = (miss_table['miss_num_raw_predict_features_120'] / 120).round(4)
    miss_table['miss_ratio_building_info'] = (miss_table['miss_num_building_info'] / 47).round(4)
    miss_table['miss_ratio_FLAG_DOCUMENT'] = (miss_table['miss_num_FLAG_DOCUMENT'] / 20).round(4)
    miss_table['flag_miss_ratio_building_info'] = np.where(miss_table['miss_ratio_building_info'] == 0, 1, 0)
    miss_table = miss_table[['All_ID', 'miss_num_raw_predict_features_120', 'miss_num_building_info',
                             'miss_num_FLAG_DOCUMENT', 'miss_ratio_raw_predict_features_120', 'miss_ratio_building_info',
                             'miss_ratio_FLAG_DOCUMENT', 'flag_miss_ratio_building_info']]

    app_train_test = app_train_test.merge(miss_table, on='All_ID', how='left')
    del miss_table

    # ============================= Categorical Features =================================
    # categorical_features
    categorical_features = pd.read_excel("Data_/Application/categorical_features.xlsx").Categorial_features.tolist()

    # Todo Bin:Bin based IV ['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE']
    categorical_features = [feature for feature in categorical_features if feature not in ['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE']]
    categorical_features.extend(['CNT_CHILDREN_bin', 'DAYS_BIRTH_bin', 'DAYS_EMPLOYED_bin', 'OWN_CAR_AGE_bin'])

    app_train_test['CNT_CHILDREN_bin'] = pd.cut(app_train_test['CNT_CHILDREN'], [0, 1, 3, 5, 10, np.inf], right=False, labels=['0', '[1-2]', '[3-5)', '[5,10)', '10+'])
    app_train_test['DAYS_BIRTH_bin'] = pd.cut(app_train_test['DAYS_BIRTH'], [20, 25, 30, 35, 40, 45, 50, 60, np.inf], labels=['[20,25)', '[25,30)', '[30,35)', '[35,40)', '[40,45)', '[45,50)', '[50,60)', '[60+)'], right=False)
    app_train_test['DAYS_EMPLOYED_bin'] = pd.cut(app_train_test['DAYS_EMPLOYED'], [0, 12, 36, 48, 60, 120, 180, 240, 300, 360, 480, np.inf], labels=['[0,1Y)', '[1Y,3Y)', '[3Y,4Y)', '[4Y,5Y)', '[5Y,10Y)', '[10Y,15Y)', '[15Y,20Y)', '[20Y,25Y)', '[25Y,30Y)', '[30Y,40Y)', '[40Y+)'], right=False)
    app_train_test['DAYS_EMPLOYED_bin'] = app_train_test['DAYS_EMPLOYED_bin'].cat.add_categories(['Missing'])
    app_train_test['DAYS_EMPLOYED_bin'].loc[app_train_test['DAYS_EMPLOYED_bin'].isnull()] = "Missing"
    app_train_test['OWN_CAR_AGE_bin'] = pd.cut(app_train_test['OWN_CAR_AGE'], [0, 1, 2, 3, 4, 6, 10, 15, 20, 25, 40, 60, np.inf], labels=['0', '1', '2', '3', '4-5', '6-9', '10-14', '15-19', '20-24', '25-39', '40-59', '60+'], right=False)
    app_train_test['OWN_CAR_AGE_bin'] = app_train_test['OWN_CAR_AGE_bin'].cat.add_categories(['Missing'])
    app_train_test['OWN_CAR_AGE_bin'].loc[app_train_test['OWN_CAR_AGE_bin'].isnull()] = "Missing"
    app_train_test[['CNT_CHILDREN_bin', 'DAYS_BIRTH_bin', 'DAYS_EMPLOYED_bin', 'OWN_CAR_AGE_bin']].isnull().any()

    # categorical_features fillna("Missing")
    app_train_test[categorical_features].isnull().any()
    app_train_test[['NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'FONDKAPREMONT_MODE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE']] = app_train_test[['NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'FONDKAPREMONT_MODE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE']].fillna("Missing")

    # Pickle
    with open('Data_/Application/categorical_features', 'wb') as fp:
        pickle.dump(categorical_features, fp)

    # ============================= Numerical Features =================================
    numerical_features = [feature for feature in app_train_test.columns.tolist() if feature not in (categorical_features+cust_cols)]
    print("Original Numerical Features Number : ", len(numerical_features))
    app_train_test[numerical_features] = app_train_test[numerical_features].astype('float32')

    # Pickle
    with open('Data_/Application/numerical_features', 'wb') as fp:
        pickle.dump(numerical_features, fp)

    # Todo : fillna()

    # ============================= Write HDF =================================
    app_train_test.to_hdf('Data_/Application/app_train_test.hdf', 'Application', mode='w', format='table')

def get_application_base():
    app_train_test = pd.read_hdf('Data_/Application/app_train_test.hdf')
    app_base = app_train_test[['All_ID', 'Set', 'SK_ID_CURR', 'TARGET']]
    app_base.to_hdf('Data_/Application/app_base.hdf', 'app_base', mode='w', format='table')

def get_application_original():
    app_train_test = pd.read_hdf('Data_/Application/app_train_test.hdf')
    app_original_feats = app_train_test.drop(['Set', 'SK_ID_CURR', 'TARGET'], axis=1)

    with open('Data_/Application/categorical_features', 'rb') as fp:
        categorical_features = pickle.load(fp)
    with open('Data_/Application/numerical_features', 'rb') as fp:
        numerical_features = pickle.load(fp)

    # del unvaluerable categorical features
    new_categorical_features = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
                                'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                                'FLAG_DOCUMENT_3', 'DAYS_EMPLOYED_bin', 'DAYS_BIRTH_bin', 'OWN_CAR_AGE_bin']
    del_cats_list = list(set(categorical_features) - set(new_categorical_features))
    print("Before del cates : ", app_original_feats.shape)
    app_original_feats = app_original_feats.drop(del_cats_list, axis=1)
    print("After del cates : ", app_original_feats.shape)
    del del_cats_list, categorical_features

    # Pickle
    with open('Data_/Application/categorical_features_Original', 'wb') as fp:
        pickle.dump(new_categorical_features, fp)

    # del unvaluerable numerical features
    del_numerical_feats = ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY']
    app_original_feats = app_original_feats.drop(del_numerical_feats, axis=1)

    # Pickle
    with open('Data_/Application/numerical_features_Original', 'wb') as fp:
        pickle.dump(numerical_features, fp)

    print("Application Original Features Number : ", app_original_feats.shape[1]-1)
    app_original_feats.to_hdf('Data_/Application/app_original_fe.hdf', 'original_fe', mode='w', format='table')

def get_application_fe2(): # 类别型变量各类别的数量-违约率 (165)
    # Read Data
    app_all = pd.read_hdf("Data_/Application/app_train_test.hdf")
    with open('Data_/Application/categorical_features', 'rb') as fp:
        categorical_features = pickle.load(fp)
    with open('Data_/Application/numerical_features', 'rb') as fp:
        numerical_features = pickle.load(fp)

    # 违约率
    from sklearn.utils import shuffle
    categorical_features.insert(0, 'TARGET')
    categorical_features.insert(0, 'All_ID')
    train = app_all.loc[app_all['Set'] == 1, categorical_features]
    train = shuffle(train, random_state=123).reset_index(drop=True)
    train['chunks_index'] = np.floor(train.index.values / 30750)  # 10 Folds
    test = app_all.loc[app_all['Set'] == 2, categorical_features]
    fe2_dict = dict()

    # 1） 拿K折平均填充测试集(Now) 2）拿训练集每个样本平均测试集
    for feature in categorical_features[2:]:
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

        fe2_dict[feature] = feature_cate_kfoldmean

        train = train.drop([feature], axis=1)
        test = test.drop([feature], axis=1)

    fe2 = pd.concat([train, test]).sort_values(by=['All_ID']).reset_index(drop=True).drop(['chunks_index', 'TARGET'], axis=1)

    # 类别个数-占比 （基于Train）
    fe2_1_cols = []
    for feature in categorical_features:
        print(feature)
        new_feature_1 = 'classnum_' + feature
        new_feature_2 = 'classratio_' + feature
        app_all[new_feature_1] = np.nan
        app_all[new_feature_2] = np.nan
        for cate in app_all[feature].unique():
            app_all.loc[app_all[feature] == cate, new_feature_1] = (app_all.loc[app_all['Set'] == 1, feature] == cate).sum()
            app_all.loc[app_all[feature] == cate, new_feature_2] = (app_all.loc[app_all['Set'] == 1, feature] == cate).sum() / app_all[app_all['Set'] == 1].shape[0]
        fe2_1_cols.extend([new_feature_1, new_feature_2])
    fe2_1_cols.insert(0, 'All_ID')
    fe2_2 = app_all[fe2_1_cols]

    fe2 = pd.merge(fe2, fe2_2, on='All_ID', how='left')

    fe2.to_hdf('Data_/Application/df_fe2.hdf', 'app_fe2', mode='w')

def get_application_fe3(): # 数值型变量 两两相除
    # Read Data
    app_all = pd.read_hdf("Data_/Application/app_train_test.hdf")

    # division_features
    division_features = ['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                         'AMT_GOODS_PRICE', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1',
                         'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                         'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
                         'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                         'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    # FE3
    fe3 = []
    for feature_i in division_features:
        for feature_j in division_features:
            if feature_i != feature_j:
                print(feature_i, feature_j)
                app_all, new_feature = ratio_2_fetures_f(data=app_all, x=feature_i, y=feature_j)
                fe3.append(new_feature)
    len(fe3)
    fe3.extend(['All_ID', 'TARGET'])
    fe3 = app_all[fe3]

    # 随机森林降维
    from sklearn.ensemble import RandomForestClassifier
    train = fe3[~fe3['TARGET'].isnull()]
    train = train.fillna(train.median())
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456, verbose=True)
    X = train.drop(['All_ID', 'TARGET'], axis=1)
    rf.fit(X=train.drop(['All_ID', 'TARGET'], axis=1), y=train['TARGET'])
    len(rf.feature_importances_)
    imp_tb = pd.DataFrame({'features': X.columns.tolist(), 'Importance': rf.feature_importances_}).reset_index(drop=True).sort_values(by=['Importance'], ascending=False)
    fe3_cols_f = imp_tb.loc[imp_tb['Importance'] >= 0.0035, 'features'].values.tolist()
    len(fe3_cols_f)

    fe3_cols_f.insert(0, 'All_ID')
    fe3 = fe3[fe3_cols_f]

    fe3 = fe3.drop(['AMT_CREDIT/AMT_ANNUITY', 'AMT_GOODS_PRICE/AMT_CREDIT', 'AMT_GOODS_PRICE/AMT_ANNUITY', 'AMT_CREDIT/AMT_INCOME_TOTAL'], axis=1)
    fe3.to_hdf('Data_/Application/app_fe3.hdf', 'app_fe3', mode='w', format='table')

def get_application_fe4(): # 衍生变量
    # Read Data
    from sklearn import preprocessing
    app_all = pd.read_hdf("Data_/Application/app_train_test.hdf")
    fe4_not = [x for x in app_all.columns.tolist() if x not in ['All_ID']]

    # 1. 收入/子女个数
    # 2. 商品价格/授信金额
    # 3. 商品价格/年金
    # 4. 商品价格/收入
    # 5. 授信额度/收入
    # 6. 授信额度/年金
    # 7. 收入/年金
    # 8. 授信额度/子女个数
    # 9. 年金/子女个数
    # 10. 收入/家庭人数
    # 11. 商品价格/家庭人数
    # 12. 授信额度/家庭人数
    # 13. 年金/家庭人数
    # df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    app_all = ratio_2_fetures(data=app_all, x='AMT_INCOME_TOTAL', y='CNT_CHILDREN')
    app_all = ratio_2_fetures(data=app_all, x='AMT_GOODS_PRICE', y='AMT_CREDIT')
    app_all = ratio_2_fetures(data=app_all, x='AMT_GOODS_PRICE', y='AMT_ANNUITY')
    app_all = ratio_2_fetures(data=app_all, x='AMT_GOODS_PRICE', y='AMT_INCOME_TOTAL')
    app_all = ratio_2_fetures(data=app_all, x='AMT_CREDIT', y='AMT_INCOME_TOTAL')
    app_all = ratio_2_fetures(data=app_all, x='AMT_CREDIT', y='AMT_ANNUITY')
    app_all = ratio_2_fetures(data=app_all, x='AMT_CREDIT', y='CNT_CHILDREN')
    app_all = ratio_2_fetures(data=app_all, x='AMT_ANNUITY', y='CNT_CHILDREN')
    app_all = ratio_2_fetures(data=app_all, x='AMT_INCOME_TOTAL', y='CNT_FAM_MEMBERS')
    app_all = ratio_2_fetures(data=app_all, x='AMT_GOODS_PRICE', y='CNT_FAM_MEMBERS')
    app_all = ratio_2_fetures(data=app_all, x='AMT_CREDIT', y='CNT_FAM_MEMBERS')
    app_all = ratio_2_fetures(data=app_all, x='AMT_ANNUITY', y='CNT_FAM_MEMBERS')

    # 14. 有无房-有无车-子女个数组合
    # 15. 婚姻-年龄-子女个数组合
    # 16. 婚姻为‘单身/离异/寡妇’的子女个数
    # 17. 成年人个数（家庭人数-子女个数）
    # 18. 年龄-子女个数组合 (排除)
    # 19. 年龄-陪同父母组合
    # 20. 年龄-是否学生组合(排除)
    # 21. 年龄-居住情况组合
    # 22. 年龄-婚姻组合(排除)
    # 23. 居住情况-陪同父母组合
    # 24. 居住情况-婚姻组合
    # 25. 居住情况-子女个数组合
    # 26. 居住情况-收入类型
    # 27. 工作年限-年龄 = 教育水平
    # 28. 工作年限-是够陪同父母
    # 29. 工作年限-居住情况-withparents
    # 30. 工作年限-是否有房
    # 31. 工作年限-是否有车
    app_all = combines_cate_featuer(data=app_all, x='FLAG_OWN_CAR', y='FLAG_OWN_REALTY', z='CNT_CHILDREN_bin')
    app_all = combines_cate_featuer(data=app_all, x='NAME_FAMILY_STATUS', y='DAYS_BIRTH_bin', z='CNT_CHILDREN_bin')
    app_all = combines_cate_featuer(data=app_all, x='NAME_FAMILY_STATUS', y='CNT_CHILDREN_bin')
    app_all['adult_numer'] = app_all['CNT_FAM_MEMBERS'] - app_all['CNT_CHILDREN']
    app_all = combines_cate_featuer(data=app_all, x='DAYS_BIRTH_bin', y='NAME_TYPE_SUITE')
    app_all = combines_cate_featuer(data=app_all, x='DAYS_BIRTH_bin', y='NAME_HOUSING_TYPE')
    app_all = combines_cate_featuer(data=app_all, x='NAME_TYPE_SUITE', y='NAME_HOUSING_TYPE')
    app_all = combines_cate_featuer(data=app_all, x='NAME_FAMILY_STATUS', y='NAME_HOUSING_TYPE')
    app_all = combines_cate_featuer(data=app_all, x='CNT_CHILDREN_bin', y='NAME_HOUSING_TYPE')
    app_all = combines_cate_featuer(data=app_all, x='NAME_INCOME_TYPE', y='NAME_HOUSING_TYPE')
    app_all['edu_level'] = app_all['DAYS_EMPLOYED'] - app_all['DAYS_BIRTH']
    app_all = combines_cate_featuer(data=app_all, x='DAYS_EMPLOYED_bin', y='NAME_TYPE_SUITE')
    app_all = combines_cate_featuer(data=app_all, x='DAYS_EMPLOYED_bin', y='NAME_HOUSING_TYPE')
    app_all = combines_cate_featuer(data=app_all, x='DAYS_EMPLOYED_bin', y='FLAG_OWN_CAR')
    app_all = combines_cate_featuer(data=app_all, x='DAYS_EMPLOYED_bin', y='FLAG_OWN_REALTY')

    # 32. 工作年限/车龄
    # 33. 车龄-是否有车组合
    # 34. 收入/车龄
    # 35. 授信额度/车龄
    # 36. 子女个数/车龄
    # 37. 注册时间是否刚刚注册
    # 38. 年龄-注册时间
    # 39. 工作年限-注册时间
    # 40. 周几申请-申请时点
    app_all = ratio_2_fetures(data=app_all, x='DAYS_EMPLOYED', y='OWN_CAR_AGE')
    app_all = combines_cate_featuer(data=app_all, x='OWN_CAR_AGE_bin', y='FLAG_OWN_CAR')
    app_all = ratio_2_fetures(data=app_all, x='AMT_INCOME_TOTAL', y='OWN_CAR_AGE')
    app_all = ratio_2_fetures(data=app_all, x='AMT_CREDIT', y='OWN_CAR_AGE')
    app_all = ratio_2_fetures(data=app_all, x='CNT_CHILDREN', y='OWN_CAR_AGE')
    app_all['f_DAYS_REGISTRATION_0'] = np.where(app_all['DAYS_REGISTRATION'] == 0, 1, 0).astype('uint8')
    app_all['REGISTRATION_BIRTH'] = app_all['DAYS_BIRTH'] - (app_all['DAYS_REGISTRATION'] / 12)
    app_all['REGISTRATION_EMPLOYED'] = (app_all['DAYS_EMPLOYED'] / 12) - (app_all['DAYS_REGISTRATION'] / 12)
    # app_all = combines_cate_featuer(data=app_all, x='f_WEEKDAY_APPR_PROCESS_START_3HOUR_APPR_PROCESS_START_6', y='HOUR_APPR_PROCESS_START')

    # 41. 收入情况/在该收入类型下均值
    # 42. 收入情况/在该收入类型下中位数
    # {收入情况/商品价格/授信额度/年金}/{收入类型/学历类型/公司类型/工作年限/年龄/居住地等级/城市等级}
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='NAME_INCOME_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='NAME_INCOME_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='NAME_INCOME_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='NAME_INCOME_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='NAME_EDUCATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='NAME_EDUCATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='NAME_EDUCATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='NAME_EDUCATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='ORGANIZATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='ORGANIZATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='ORGANIZATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='ORGANIZATION_TYPE')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='DAYS_EMPLOYED_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='DAYS_EMPLOYED_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='DAYS_EMPLOYED_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='DAYS_EMPLOYED_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='DAYS_BIRTH_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='DAYS_BIRTH_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='DAYS_BIRTH_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='DAYS_BIRTH_bin')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='REGION_RATING_CLIENT')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='REGION_RATING_CLIENT')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='REGION_RATING_CLIENT')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='REGION_RATING_CLIENT')
    app_all = division_class_mean_medi(data=app_all, x='AMT_INCOME_TOTAL', y='REGION_RATING_CLIENT_W_CITY')
    app_all = division_class_mean_medi(data=app_all, x='AMT_GOODS_PRICE', y='REGION_RATING_CLIENT_W_CITY')
    app_all = division_class_mean_medi(data=app_all, x='AMT_CREDIT', y='REGION_RATING_CLIENT_W_CITY')
    app_all = division_class_mean_medi(data=app_all, x='AMT_ANNUITY', y='REGION_RATING_CLIENT_W_CITY')

    # 43. 工作年限/年龄
    # 44. 车龄/年龄
    # 45. 所在公司类型工资平均工资水平
    app_all['DAYS_EMPLOYED/DAYS_BIRTH'] = app_all['DAYS_EMPLOYED'] / app_all['DAYS_BIRTH']
    app_all['OWN_CAR_AGE/DAYS_BIRTH'] = app_all['OWN_CAR_AGE'] / app_all['DAYS_BIRTH']
    inc_by_org = app_all[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    app_all['INC_BY_ORG'] = app_all['ORGANIZATION_TYPE'].map(inc_by_org)

    # 客户社交网络有30DPD个数
    # 客户社交网络有60DPD个数
    app_all['DEF_60_CNT_SOCIAL_CIRCLE_bin'] = np.where(app_all['DEF_60_CNT_SOCIAL_CIRCLE'] == 0, 0, 1)
    app_all['DEF_60_CNT_SOCIAL_CIRCLE_bin'] = np.where(app_all['DEF_60_CNT_SOCIAL_CIRCLE'] == 0, 0, 1)

    # 申请前一周内 征信局查询次数
    # 申请前一月内 征信局查询次数
    # 申请前一年内 征信局查询次数
    app_all['AMT_REQ_CREDIT_BUREAU_WEEK_bin'] = np.where(app_all['AMT_REQ_CREDIT_BUREAU_WEEK'].isnull(), 2,
                                                     np.where(app_all['AMT_REQ_CREDIT_BUREAU_WEEK'] <= 1, 0, 1))
    app_all['AMT_REQ_CREDIT_BUREAU_MON_bin'] = np.where(app_all['AMT_REQ_CREDIT_BUREAU_MON'].isnull(), 4,
                                                         np.where(app_all['AMT_REQ_CREDIT_BUREAU_MON'] <= 1, 0,
                                                                  np.where(app_all['AMT_REQ_CREDIT_BUREAU_MON'] <= 5, 1,
                                                                           np.where(app_all['AMT_REQ_CREDIT_BUREAU_MON'] <= 10, 2, 3))))
    app_all['AMT_REQ_CREDIT_BUREAU_YEAR_bin'] = np.where(app_all['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull(), 4,
                                                        np.where(app_all['AMT_REQ_CREDIT_BUREAU_YEAR'] <= 1, 0,
                                                                 np.where(app_all['AMT_REQ_CREDIT_BUREAU_YEAR'] <= 5, 1, 2)))

    # EXT_SOURCE.stat
    app_all['EXT_SOURCE_Min'] = app_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    app_all['EXT_SOURCE_Max'] = app_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    app_all['EXT_SOURCE_MEAN'] = app_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    app_all['EXT_SOURCE_STD'] = app_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    app_all['EXT_SOURCE_STD'] = app_all['EXT_SOURCE_STD'].fillna(app_all['EXT_SOURCE_STD'].mean())
    app_all['EXT_SOURCE_MissingNumber'] = app_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isnull().sum(axis=1)

    # Del FE4
    del_fe4 = ['f_NAME_FAMILY_STATUS_0DAYS_BIRTH_bin_2CNT_CHILDREN_bin_3',
               'f_NAME_FAMILY_STATUS_2DAYS_BIRTH_bin_1CNT_CHILDREN_bin_0',
               'f_NAME_FAMILY_STATUS_2DAYS_BIRTH_bin_0CNT_CHILDREN_bin_2', 'f_NAME_FAMILY_STATUS_3CNT_CHILDREN_bin_3',
               'f_NAME_FAMILY_STATUS_0CNT_CHILDREN_bin_3', 'f_NAME_FAMILY_STATUS_0DAYS_BIRTH_bin_0CNT_CHILDREN_bin_2',
               'f_NAME_FAMILY_STATUS_0DAYS_BIRTH_bin_3CNT_CHILDREN_bin_3',
               'f_NAME_FAMILY_STATUS_1DAYS_BIRTH_bin_1CNT_CHILDREN_bin_3',
               'f_NAME_FAMILY_STATUS_3DAYS_BIRTH_bin_0CNT_CHILDREN_bin_2', 'f_DAYS_BIRTH_bin_1NAME_TYPE_SUITE_5',
               'f_DAYS_BIRTH_bin_3NAME_TYPE_SUITE_3', 'f_FLAG_OWN_CAR_0FLAG_OWN_REALTY_1CNT_CHILDREN_bin_4',
               'f_DAYS_BIRTH_bin_2NAME_TYPE_SUITE_2', 'f_DAYS_BIRTH_bin_4NAME_TYPE_SUITE_2',
               'f_DAYS_BIRTH_bin_7NAME_TYPE_SUITE_3', 'f_DAYS_BIRTH_bin_5NAME_HOUSING_TYPE_4',
               'f_NAME_FAMILY_STATUS_0NAME_HOUSING_TYPE_4', 'f_DAYS_BIRTH_bin_0NAME_HOUSING_TYPE_4',
               'f_NAME_TYPE_SUITE_6NAME_HOUSING_TYPE_4', 'f_DAYS_EMPLOYED_bin_9FLAG_OWN_CAR_1',
               'f_DAYS_EMPLOYED_bin_9FLAG_OWN_REALTY_1', 'f_DAYS_EMPLOYED_bin_9FLAG_OWN_REALTY_0',
               'f_DAYS_EMPLOYED_bin_3NAME_HOUSING_TYPE_0', 'f_DAYS_EMPLOYED_bin_3NAME_HOUSING_TYPE_3',
               'f_DAYS_EMPLOYED_bin_5NAME_HOUSING_TYPE_4', 'f_DAYS_EMPLOYED_bin_9NAME_TYPE_SUITE_1',
               'f_DAYS_EMPLOYED_bin_9NAME_TYPE_SUITE_7', 'f_DAYS_EMPLOYED_bin_7NAME_TYPE_SUITE_0',
               'f_DAYS_EMPLOYED_bin_6NAME_TYPE_SUITE_6', 'f_DAYS_EMPLOYED_bin_5NAME_TYPE_SUITE_6',
               'f_DAYS_EMPLOYED_bin_0NAME_TYPE_SUITE_2', 'f_DAYS_EMPLOYED_bin_10NAME_TYPE_SUITE_5',
               'f_DAYS_EMPLOYED_bin_1NAME_TYPE_SUITE_5', 'f_DAYS_EMPLOYED_bin_4NAME_TYPE_SUITE_2',
               'f_DAYS_EMPLOYED_bin_4NAME_TYPE_SUITE_5', 'f_NAME_INCOME_TYPE_4NAME_HOUSING_TYPE_0',
               'f_CNT_CHILDREN_bin_3NAME_HOUSING_TYPE_5', 'f_NAME_FAMILY_STATUS_2NAME_HOUSING_TYPE_0',
               'f_NAME_FAMILY_STATUS_2NAME_HOUSING_TYPE_4', 'f_DAYS_EMPLOYED_bin_9FLAG_OWN_CAR_0',
               'f_NAME_TYPE_SUITE_5NAME_HOUSING_TYPE_5', 'f_NAME_TYPE_SUITE_4NAME_HOUSING_TYPE_2',
               'f_DAYS_EMPLOYED_bin_9NAME_HOUSING_TYPE_1']
    print('Before Drop fe4 shape : ', app_all.shape)
    fe4 = app_all.drop(fe4_not, axis=1)
    # fe4 = fe4.drop([feature for feature in fe4.columns.tolist() if feature in del_fe4], axis=1)
    print('After Drop fe4 shape : ', fe4.shape)

    # to_hdf
    print("App_FE4 Number : ", fe4.shape[1]-1)
    fe4.columns.tolist()
    fe4.to_hdf('Data_/Application/app_fe4.hdf', 'app_fe4', mode='w', format='table')



def get_application_fe5(): # Rank Variables
    # Read Data
    from scipy import stats
    app_all = pd.read_hdf("Data_/Application/app_train_test.hdf")

    # Rank Features
    rank_features = ['All_ID', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    fe5 = app_all[rank_features].fillna(0)

    for feature in [x for x in rank_features if x not in 'All_ID']:
        fe5 = rank_feature(data=fe5, x=feature)
        del fe5[feature]

    # to_hdf
    print("App_FE5 Number : ", fe5.shape[1] - 1)
    fe5.columns.tolist()
    fe5.to_hdf('Data_/Application/app_fe5.hdf', 'app_fe5', mode='w', format='table')





















