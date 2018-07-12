# -*- coding:utf8 -*-
import pandas as pd
import sys
import logging
from pandas import Series,DataFrame
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn import  feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,make_scorer,explained_variance_score
import xgboost as xgb
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_val_predict,KFold,StratifiedKFold
from scipy.special import boxcox1p
from scipy.stats import skew,boxcox_normmax
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

logging.basicConfig(level=logging.INFO,format='%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')
path_train = "/data/dm/train.csv"  # 训练文件
# path_train = "./train_new_nonsen.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
pay = 0.3 #赔付率阈值
cv = -1
metrics = 0
isensemble = 0
istrain = 0
isadddata = 0
mn = 1

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
# lgb_params['max_bin'] = 10
# lgb_params['subsample'] = 0.8
# lgb_params['subsample_freq'] = 10
# lgb_params['colsample_bytree'] = 0.8   
# lgb_params['min_child_samples'] = 500
# lgb_params['seed'] = 99
lgb_params['verbose'] = -1

xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 490
xgb_params['max_depth'] = 4
# xgb_params['subsample'] = 0.9
# xgb_params['colsample_bytree'] = 0.9  
# xgb_params['min_child_weight'] = 10

mparams = [{'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls','random_state':1976},
           {'max_depth':6, 'min_child_weight': 4,'learning_rate':0.01, 'n_estimators':500, 'silent':True, 'objective':'reg:linear'},
           {'boosting_type': "dart" , 'learning_rate':0.0021},
           {'n_estimators' : 1300, 'max_features' : 0.5, 'min_samples_leaf' : 30,'random_state' : 1},
           {'max_depth':4, 'learning_rate':0.005, 'n_estimators':1000, 'silent':True, 'objective':'reg:linear',
            'bagging_fraction':0.8,'bagging_freq':5, 'feature_fraction':0.2319,'feature_fraction_seed':9,
            'bagging_seed':9,'min_data_in_leaf':6, 'min_sum_hessian_in_leaf':11}]
mff = {}#是否已生成特征
fl = {'p1':['t31221_mean','h11202_mean','h11301_mean','c11022__2','t21002_mean','s11323_mean'],
      'p2':['t11200_mean','t11201_std','t31200_6'],
      'p3':['t31221_mean','t11001_mean','t11401_mean']}
fl = {'p1':['t31221','h11202_std','t21001_mean','t21001_std']}
afl = [ 't10000', 't10001', 't10301', 't11000', 't11001','t11301','t10010',
        't10011', 't10311', 't11010', 't11011', 't11311','t21201','t21200',
        't21002', 't21211', 't21210', 't31200', 't31201','t31210','t31211',
        't11200', 't11201', 't11210', 't11211', 't11401','t11601','t41201',
        't41211', 'h11000', 'h11010', 'h11101', 'h11111','h11301','h11311',
        's11000', 's11010', 's11101', 's11301', 's11103','s11323','s11111',
        's11311', 's11113', 's11313', 'l11001', 'l11000','l11003','l21001',
        'l21000', 'l21003', 'l11011', 'l11010', 'l11013','l21011','l21010',
        'l21013', 'c11021', 'c11022', 'h11202', 't31221','t21001']
afl = ['t10000_mean','t10000_std','t10001_mean','t10001_std','t10301_mean','t10301_std'
,'t11000_mean','t11000_std','t11001_mean','t11001_std','t11301_mean','t11301_std'
,'t10010_mean','t10010_std','t10011_mean','t10011_std','t10311_mean','t10311_std'
,'t11010_mean','t11010_std','t11011_mean','t11011_std','t11311_mean'
,'t11311_std','t21201_mean','t21201_std','t21200_mean','t21200_std'
,'t21002_mean','t21002_std','t21211_mean','t21211_std','t21210_mean'
,'t21210_std','t31200_0','t31200_1','t31200_2','t31200_3','t31200_4'
,'t31200_5','t31200_6','t31200_7','t31200_8','t31200_9','t31200_10'
,'t31200_11','t31200_12','t31200_13','t31200_14','t31200_15','t31200_16'
,'t31200_17','t31200_18','t31200_19','t31200_20','t31200_21','t31200_22'
,'t31200_23','t31201_0','t31201_1','t31201_2','t31201_3','t31201_4'
,'t31201_5','t31201_6','t31201_7','t31201_8','t31201_9','t31201_10'
,'t31201_11','t31201_12','t31201_13','t31201_14','t31201_15','t31201_16'
,'t31201_17','t31201_18','t31201_19','t31201_20','t31201_21','t31201_22'
,'t31201_23','t31210_0','t31210_1','t31210_2','t31210_4','t31210_5'
,'t31210_6','t31210_7','t31210_8','t31210_9','t31210_10','t31210_11'
,'t31210_12','t31210_13','t31210_14','t31210_15','t31210_16','t31210_17'
,'t31210_18','t31210_19','t31210_20','t31210_21','t31210_22','t31210_23'
,'t31211_0','t31211_1','t31211_2','t31211_4','t31211_5','t31211_6'
,'t31211_7','t31211_8','t31211_9','t31211_10','t31211_11','t31211_12'
,'t31211_13','t31211_14','t31211_15','t31211_16','t31211_17','t31211_18'
,'t31211_19','t31211_20','t31211_21','t31211_22','t31211_23','t11200_mean'
,'t11200_std','t11201_mean','t11201_std','t11210_mean','t11210_std'
,'t11211_mean','t11211_std','t11401_mean','t11401_std','t11601_hminm'
,'t11601_hminstd','t11601_hmaxm','t11601_hmaxstd','t41201_mean'
,'t41201_std','t41211_mean','t41211_std','h11000_mean','h11000_std'
,'h11010_mean','h11010_std','h11101_mean','h11101_std','h11111_mean'
,'h11111_std','h11301_mean','h11301_std','h11311_mean','h11311_std'
,'s11000_mean','s11000_std','s11010_mean','s11010_std','s11101_mean'
,'s11101_std','s11301_mean','s11301_std','s11103_mean','s11103_std'
,'s11323_mean','s11323_std','s11111_mean','s11111_std','s11311_mean'
,'s11311_std','s11113_mean','s11113_std','s11313_mean','s11313_std'
,'l11001_mean','l11001_std','l11000_mean','l11000_std','l11003_mean'
,'l11003_std','l21001_mean','l21001_std','l21000_mean','l21000_std'
,'l21003_mean','l21003_std','l11011_mean','l11011_std','l11010_mean'
,'l11010_std','l11013_mean','l11013_std','l21011_mean','l21011_std'
,'l21010_mean','l21010_std','l21013_mean','l21013_std','c11020__1'
,'c11020__2','c11020__3','c11020__4','c11021__1','c11021__2','c11021__3'
,'c11021__4','c11022__1','c11022__2','c11022__3','c11022__4']

afl = ['t10000_mean','t10000_std','t10001_mean','t10001_std','t10301_mean','t10301_std'
,'t11000_mean','t11000_std','t11001_mean','t11001_std','t11301_mean','t11301_std'
,'t10010_mean','t10010_std','t10011_mean','t10011_std','t10311_mean','t10311_std'
,'t31221_mean','t31221_std','t11010_mean','t11010_std','t11011_mean','t11011_std','t11311_mean'
,'t11311_std','t21201_mean','t21201_std','t21200_mean','t21200_std'
,'t21002_mean','t21002_std','t21211_mean','t21211_std','t21210_mean'
,'t21210_std','t31200_0','t31200_1','t31200_2','t31200_3','t31200_4'
,'t31200_5','t31200_6','t31200_7','t31200_8','t31200_9','t31200_10'
,'t31200_11','t31200_12','t31200_13','t31200_14','t31200_15','t31200_16'
,'t31200_17','t31200_18','t31200_19','t31200_20','t31200_21','t31200_22'
,'t31200_23','t31210_0','t11200_mean'
,'t11200_std','t11201_mean','t11201_std','t11210_mean','t11210_std'
,'t11211_mean','t11211_std','t11401_mean','t11401_std','t11601_hminm'
,'t11601_hminstd','t11601_hmaxm','t11601_hmaxstd','t41201_mean'
,'t41201_std','t41211_mean','t41211_std','h11000_mean','h11000_std'
,'h11010_mean','h11010_std','h11101_mean','h11101_std','h11111_mean'
,'h11111_std','h11301_mean','h11301_std','h11311_mean','h11311_std'
,'s11000_mean','s11000_std','s11010_mean','s11010_std','s11101_mean'
,'s11101_std','s11301_mean','s11301_std','s11103_mean','s11103_std'
,'s11323_mean','s11323_std','s11111_mean','s11111_std','s11311_mean'
,'s11311_std','s11113_mean','s11113_std','s11313_mean','s11313_std'
,'l11001_mean','l11001_std','l11000_mean','l11000_std','l11003_mean'
,'l11003_std','l21001_mean','l21001_std','l21000_mean','l21000_std'
,'l21003_mean','l21003_std','l11011_mean','l11011_std','l11010_mean'
,'l11010_std','l11013_mean','l11013_std','l21011_mean','l21011_std'
,'l21010_mean','l21010_std','l21013_mean','l21013_std','c11020__1'
,'c11020__2','c11020__3','c11020__4','c11021__1','c11021__2','c11021__3'
,'c11021__4','c11022__1','c11022__2','c11022__3','c11022__4','h11202_mean'
,'h11202_std','t21001_mean','t21001_std']


dfn = 'df_tfc1'
statsf = 'stats2'
hci = 0.8
tci = 0.8
lci = 0.8
sci = 0.8

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def stats1(df,df_base,col,pre,ci):
    l1 = pre + 'max'
    l2 = pre + 'min'
    l3 = pre + 'std'
    l4 = pre + 'mean'
    l5 = pre + 'pc1'
    l6 = pre + 'pc2'
    rate1 = (1-(1 -ci)/2)*100
    rate2 = ((1 -ci)/2)*100
    tn = df.groupby(['TERMINALNO'])[col].agg(
        {l1:np.max,l4:np.mean,l2:np.min,l3:np.std,
         l5:lambda x: np.percentile(x,rate1),
         l6:lambda x: np.percentile(x,rate2) }).reset_index()
    df_temp = pd.merge(df_base, tn, how='left', on=['TERMINALNO'])
    return df_temp

def stats2(df,df_base,col,pre,ci):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    tn = df
    tn = df.groupby(['TERMINALNO'])[col].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, tn, how='left', on=['TERMINALNO'])
    return df_temp

def tfc24r(df,pre,level=0): #计算24小时占比
    t1 = df
    t1.rename(columns={'minute':'c1'}, inplace = True)
    if level == 0:
        t2 = df.groupby(['TERMINALNO'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO'])
    elif level == 1:
        t2 = df.groupby(['TERMINALNO','month','day'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO','month','day'])
    else:
        t2 = df.groupby(['TERMINALNO','TRIP_ID'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO','TRIP_ID'])
    t1['ratio'] = t1['c1']/t1['c2']
    t1.fillna(0,inplace = True)
    t1 = t1.groupby(['TERMINALNO','hour'])['ratio'].mean().reset_index()
    t2 = pd.get_dummies(t1['hour'], prefix= pre)
    t1 = pd.concat([t1, t2], axis=1)
    startcol = t2.columns.values[0]
    t1.loc[:,startcol:] = t1.loc[:,startcol:].mul(t1['ratio'],axis=0)
    t1 = t1.groupby('TERMINALNO').sum().reset_index()
    t1 = t1.drop(['hour','ratio'], axis=1)
    return t1

def cfcr(df,pre,level=0): #计算各状态占比
    t1 = df
    if level == 0:
        t2 = df.groupby(['TERMINALNO'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO'])
    elif level == 1:
        t2 = df.groupby(['TERMINALNO','month','day'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO','month','day'])
    else:
        t2 = df.groupby(['TERMINALNO','TRIP_ID'])['c1'].agg({'c2':'sum'}).reset_index()
        t1 = pd.merge(t1, t2, how='left', on=['TERMINALNO','TRIP_ID'])
    t1['ratio'] = t1['c1']/t1['c2']
    t1.fillna(0,inplace = True)
    t1 = t1.groupby(['TERMINALNO','CALLSTATE'])['ratio'].mean().reset_index()
    t2 = pd.get_dummies(t1['CALLSTATE'], prefix= pre)
    t1 = pd.concat([t1, t2], axis=1)
    startcol = t2.columns.values[0]
    t1.loc[:,startcol:] = t1.loc[:,startcol:].mul(t1['ratio'],axis=0)
    t1 = t1.groupby('TERMINALNO').sum().reset_index()
    t1 = t1.drop(['CALLSTATE','ratio'], axis=1)
    return t1


# 时间特征:
'''
第一位：特征类别：1：时刻，2：时长，3：占比,4:次数
第二位：是否加权：0：加权,1：不加权，
第三位：统计方法：0：原值，1：mean,2:count,3:std,4:sum
第四位：统计范围：0：周一至周五，1：双休日,2:全部
第五位：粒度：0：车主，1：日，2：trip
'''
def t10000(df,df_base):#周一至周五出行时间加权值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t10001(df,df_base):#周一至周五出行时间加权值，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t10301(df,df_base):#周一至周五出行时间加权STD值，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11000(df,df_base):#周一至周五出行时刻
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['minute'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11001(df,df_base):#周一至周五出行时刻,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11301(df,df_base):#周一至周五出行时刻std,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t10010(df,df_base):#周六、日出行时刻加权值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t10011(df,df_base):#周六、日出行时刻加权值，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t10311(df,df_base):#周六、日出行时刻加权值STD，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11010(df,df_base):#双休日出行时刻
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['minute'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11011(df,df_base):#双休日出行时刻,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11311(df,df_base):#双休日出行时刻STD,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t21201(df,df_base):#周一至周五出行时长,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t21200(df,df_base):#周一至周五出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t21002(df,df_base):#周一至周五出行时长（TIME）,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['TIME'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'TIME',pre,tci)

def t21001(df,df_base):#周一至周五出行时长（TIME）,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','TRIP_ID'])['TIME'].agg(lambda x:max(x)-min(x)).reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['TIME'].sum().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'TIME',pre,tci)

def t21211(df,df_base):#周六、日出行时长,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t21210(df,df_base):#周六、日出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t31200(df,df_base):#周一至周五24小时出行时间占比 
    fname = sys._getframe().f_code.co_name
    pre = fname
    mff[fname] = 1
    
    t1 = df[(df['weekday'] < 5)]
    t1 = df.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','hour'])['minute'].count().reset_index()
    t1 = tfc24r(t1,pre)
    return t1

def t31201(df,df_base): #周一至周五24小时出行时间占比，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()   
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = tfc24r(t1,pre,1)
    return t1

def t31221(df,df_base): #22-2出行占比，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    
    t1 = df
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(t1['hour'].isin([22,23,0,1]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    t1.fillna(0,inplace = True)
    func = globals().get(statsf)
    t1 = func(t1,df_base,'ratio1',pre,1)
    return t1

def t31210(df,df_base): #双休日24小时出行时间占比
    fname = sys._getframe().f_code.co_name
    pre = fname
    mff[fname] = 1
    
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','hour'])['minute'].count().reset_index()
    t1 = tfc24r(t1,pre)
    return t1

def t31211(df,df_base): #双休日24小时出行时间占比，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname
    mff[fname] = 1
    
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = tfc24r(t1,pre,1)
    return t1

def t11200(df,df_base):#周一至周五出行不同时点数
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['TRIP_ID'].count().reset_index() 
    t1 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci) 

def t11201(df,df_base):#周一至周五出行不同时点数，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1 

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['TRIP_ID'].count().reset_index() 
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index() 
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci) 

def t11210(df,df_base):#双休日出行不同时点数
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['TRIP_ID'].count().reset_index() 
    t1 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci) 

def t11211(df,df_base):#双休日出行不同时点数，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1 

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['TRIP_ID'].count().reset_index() 
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index() 
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci) 

def t11401(df,df_base):#周一至周五出行时刻按天sum值stats1值 test:0.00858
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour','minute'])['second'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].sum().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def t11601(df,df_base):#周一至周五，按天统计最早及最晚出行时刻,le
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'hminm'
    l2 = pre + 'hminstd'
    l3 = pre + 'hmaxm'
    l4 = pre + 'hmaxstd'

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].agg({'hmin':'min','hmax':'max'}).reset_index()
    t1 = t1.groupby(['TERMINALNO']).agg({'hmin':{l1:'mean',l2:'std'},'hmax':{l3:'mean',l4:'std'}}) 
    levels = t1.columns.levels
    labels = t1.columns.labels
    t1.columns = levels[1][labels[1]]
    t1 = t1.reset_index()
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def t41201(df,df_base):#周一至周五，每天出行次数，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','TRIP_ID'])['hour'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['TRIP_ID'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'TRIP_ID',pre,tci)

def t41211(df,df_base):#双休日每天出行次数，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','TRIP_ID'])['hour'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['TRIP_ID'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'TRIP_ID',pre,tci)

#海拔特征
def h11000(df,df_base):#周一至周五出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11010(df,df_base):#周六、日出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11101(df,df_base):#周一至周五平均海拔，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11111(df,df_base):#双休日平均海拔，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11301(df,df_base):#周一至周五按天出行海拔 STD值,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11311(df,df_base):#双休日按天出行海拔 STD值,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def h11202(df,df_base):#周一至周五按天出行海拔,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['HEIGHT'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)


#速度特征
def s11000(df,df_base):#周一至周五出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11010(df,df_base):#周六、日出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11101(df,df_base):#周一至周五出行速度stats1值,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['SPEED'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11301(df,df_base):#周一至周五出行速度std,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['SPEED'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11103(df,df_base):#周一至周五出行速度,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['SPEED'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11323(df,df_base):#出行速度std,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[df['SPEED']>0]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['SPEED'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11111(df,df_base):#双休日出行速度stats1值,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['SPEED'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11311(df,df_base):#双休日出行速度std,按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['SPEED'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11113(df,df_base):#双休日出行速度,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['SPEED'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def s11313(df,df_base):#双休日出行速度std,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['SPEED'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)


#经纬度特征
def l11001(df,df_base):#周一至周五,LONGITUDE极差，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','month','day'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l11000(df,df_base):#周一至周五,LONGITUDE极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l11003(df,df_base):#周一至周五,LONGITUDE极差，按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l21001(df,df_base):#周一至周五,LATITUDE极差，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','month','day'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

def l21000(df,df_base):#周一至周五,LATITUDE极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

def l21003(df,df_base):#周一至周五,LATITUDE极差,按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

def l11011(df,df_base):#双休日,LONGITUDE极差，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','month','day'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l11010(df,df_base):#双休日,LONGITUDE极差 test:-0.03743
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l11013(df,df_base):#双休日,LONGITUDE极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LONGITUDE',pre,lci)

def l21011(df,df_base):#双休日,LATITUDE极差，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','month','day'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

def l21010(df,df_base):#双休日,LATITUDE极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

def l21013(df,df_base):#双休日,LATITUDE极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]

    t1 = t1.groupby(['TERMINALNO','TRIP_ID'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'LATITUDE',pre,lci)

#通话状态特征
def c11020(df,df_base):#行车cs占比统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[df['CALLSTATE']>0]
    t1 = t1.groupby(['TERMINALNO','CALLSTATE'])['hour'].agg({'c1':'count'}).reset_index()
    t1 = cfcr(t1,pre)
    return t1

def c11021(df,df_base):#行车cs占比统计，按天统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[df['CALLSTATE']>0]
    t1 = t1.groupby(['TERMINALNO','month','day','CALLSTATE'])['hour'].agg({'c1':'count'}).reset_index()
    t1 = cfcr(t1,pre,1)
    return t1

def c11022(df,df_base):#行车cs占比统计，按trip统计
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[df['CALLSTATE']>0]
    t1 = t1.groupby(['TERMINALNO','TRIP_ID','CALLSTATE'])['hour'].agg({'c1':'count'}).reset_index()
    t1 = cfcr(t1,pre,2)
    return t1


def transdata1(df):
    # df['TERMINALNO'] = df['TERMINALNO'].astype(np.uint16)
    # df['TIME'] = df['TIME'].astype(np.uint32)
    # df['TRIP_ID'] = df['TRIP_ID'].astype(np.uint16)
    # df['LONGITUDE'] = df['LONGITUDE'].astype(np.float32)
    # df['LATITUDE'] = df['LATITUDE'].astype(np.float32)
    # df['DIRECTION'] = df['DIRECTION'].astype(np.uint16)
    # df['HEIGHT'] = df['HEIGHT'].astype(np.float32)
    # df['SPEED'] = df['SPEED'].astype(np.float32)
    # df['CALLSTATE'] = df['CALLSTATE'].astype(np.uint8)
    df.fillna(0, inplace = True)
    df['DIRECTION'] = df['DIRECTION'].astype(np.int16)

    df['ftime'] = df['TIME'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))
    df['ftime'] = pd.to_datetime(df['ftime'])
    df['hour'] = df['ftime'].dt.hour.astype(np.uint8)
    df['minute'] = df['ftime'].dt.minute.astype(np.uint8)
    df['second'] = df['ftime'].dt.second.astype(np.uint8)
    df['weekday'] = df['ftime'].dt.weekday.astype(np.uint8)
    df['wno'] = df['ftime'].dt.weekofyear.astype(np.uint8)
    df['day'] = df['ftime'].dt.day.astype(np.uint8)
    df['month'] = df['ftime'].dt.month.astype(np.uint8)

    df = df[(df['month'] != 10) | ((df['month'] == 10) & (df['day'] > 9))]#排除国庆长假
    df = df[(df['month'] != 9) | ((df['month'] == 9) & ((df['day'] > 18) | (df['day'] < 15)))]#排除中秋

    df = df.sort_values(by = ['TERMINALNO','TIME'],axis = 0).reset_index(drop=True)
    df['stime'] = df['TIME'].shift(1)
    df['TRIP_ID'] = df['TIME'] - df['stime']
    ts = df['TRIP_ID'].values
    tslv1 = len(ts)
    ts[0] = 1
    for i1 in range(1,tslv1):
        if ts[i1]>300:
            ts[i1] = ts[i1-1] + 1
        else:
            ts[i1] = ts[i1-1]
    df['TRIP_ID'] = ts
    df['TRIP_ID'] = df['TRIP_ID'].astype(np.uint32)
    df = df.drop(['stime','ftime'], axis=1)
    
def output(df):
    temp = df[['TERMINALNO', 'Y']].sort_values(by='Y', axis=0, ascending=True)
    temp.rename(columns={'TERMINALNO': 'Id', 'Y': 'Pred'}, inplace=True)
    temp.to_csv('./model/output.csv', index=False)

def ds(X,y,random):
    t = pd.concat([X,y], axis=1)
    df_pay1 = t[ (t['Y'] > 0) & (t['Y'] <= 0.8)]
    df_pay2 = t[ (t['Y'] > 0.8)]
    df_npay = t[t['Y'] <= 0]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df_pay1.iloc[:,:-1], df_pay1['Y'], random_state=random, test_size= 0.15)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df_npay.iloc[:,:-1], df_npay['Y'], random_state=random, test_size= 0.15)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(df_pay2.iloc[:,:-1], df_pay2['Y'], random_state=random, test_size= 0.15)
    X_train = pd.concat([X_train1,X_train2,X_train3], axis=0)
    X_test = pd.concat([X_test1,X_test2,X_test3], axis=0)
    y_train = pd.concat([y_train1,y_train2,y_train3],axis=0)
    y_test = pd.concat([y_test1,y_test2,y_test3],axis=0)
    return X_train,X_test,y_train,y_test

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2
    return giniSum / len(actual)

def pse1(actual, pred): 
    assert (len(actual) == len(pred))
    return 2*roc_auc_score(actual, pred)-1

def pse2(actual, pred): 
    assert (len(actual) == len(pred))
    return (abs(actual - pred) * (pred/pred.sum()*10 + np.rint(pred+0.2)) ).sum()

gini_score = make_scorer(gini, greater_is_better=True)
pse1_score = make_scorer(pse1, greater_is_better=True)
r2_score1 = make_scorer(r2_score, greater_is_better=True)
evs_score1 = make_scorer(explained_variance_score, greater_is_better=True)

def makefeature1(fs,df):
    if "Y" in df.columns.tolist():
        df_owner = df[['TERMINALNO','Y','TRIP_ID']].groupby(['TERMINALNO','Y']).count().reset_index()
        df_owner['Y1'] = pd.cut(df_owner['Y'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
        18,19,20,25,30,35,40,45,50,60,70,80,90,100,150,200,500],labels = [1,2,3,4,5,6,7,8,9,10,11,12,
        13,14,15,16,17,18,19,20,25,30,35,40,45,50,60,70,80,90,100,150,200,500]).astype(np.float32)
        df_owner.fillna(0, inplace = True)
        df_owner['Y1'] = df_owner['Y1'].astype(np.uint16)
        df_owner['Y2'] = pd.cut(df_owner['Y'],[0,3,400],labels = [1,2]).astype(np.float32).fillna(0).astype(np.uint8)
    else:
        df_owner = df[['TERMINALNO','TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
    for i1,v1 in fs.items():
        for i2,v2 in enumerate(v1):
            sl1 = v2.split('_')
            if sl1[0] not in mff:
                func = globals().get(sl1[0])
                dfn = func(df,df_owner.iloc[:,:1])
                df_owner = pd.merge(df_owner, dfn, how='left', on=['TERMINALNO'])
    logging.debug(df_owner.columns)
    #以下使数值特征正态化
    # df_owner['Y'] = boxcox1p(df_owner['Y'] , boxcox_normmax(df_owner['Y']+1))
    # numeric_feats = df_owner.dtypes[df_owner.dtypes != "object"].index
    # skewed_feats = df_owner[numeric_feats].apply(lambda x: skew(x.dropna()))
    # skewed_feats = skewed_feats[skewed_feats > 0.75]
    # skewed_feats = skewed_feats.index
    # df_owner[skewed_feats] = np.log1p(df_owner[skewed_feats])
    return df_owner

def selectfeature1(fs,df_feature): 
    first = False
    starti = 0
    endi = -1
    t = df_feature.iloc[:,:1]
    colums = list(df_feature.columns)
    for v in fs:
        for i1,v1 in enumerate(colums):
            if (('_' in v) & (v1.startswith(ｖ))) | (('_' not in v) & (v1.startswith(ｖ+'_'))):
                if first == False:
                    starti=i1
                    first = True
            else:
                if first == True and endi == -1:
                    endi = i1
        if endi == -1:
            endi = i1 + 1
        t = pd.concat([t,df_feature.iloc[:,starti : endi]], axis=1)
        first = False
        starti = 0
        endi = -1
    logging.debug(t.columns)
    return t

def gs1(X,y):
    if mn == 0:
        X.fillna( 0, inplace = True )
        param1 = {'max_depth':range(3,7,1), 'min_samples_split':range(2,7,1)}
        model = GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.01, random_state = 1976)
    if mn == 1:
        param1 = {'max_depth':range(3,7,1), 'min_child_weight':range(1,6,1)}
        model = xgb.XGBRegressor(**mparams[mn])
    gs = GridSearchCV(model,param1,scoring = gini_score,cv = 10)
    gs.fit(X,y)
    logging.info(gs.best_params_)
    logging.info(gs.best_score_)

def predict1(X_train,y_train,X_test):
    # skewed_features = X_train.skew(axis = 0)
    # skewed_features = skewed_features[np.abs(skewed_features)>0.5]
    # X_train[skewed_features.index] = np.log1p(X_train[skewed_features.index])
    # skewed_features = X_test.skew(axis = 0)
    # skewed_features = skewed_features[np.abs(skewed_features)>0.5]
    # X_test[skewed_features.index] = np.log1pscore1X_test[skewed_features.index])
    # y_train = np.log1p(y_train)
    # y_train.fillna(0, inplace = True)
    if mn == 0:
        # X_train.fillna(0, inplace = True)
        # X_test.fillna(0, inplace = True)
        # np.nan_to_num(X_train)
        # np.nan_to_num(X_test)
        model = GradientBoostingRegressor(**mparams[mn])
    elif mn == 1:
        # model = xgb.XGBRegressor(**mparams[mn])
        model = XGBClassifier(**xgb_params)
    elif mn == 2:
        # model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
        model = LassoCV(cv=10)
        # model = SelectFromModel(model, threshold=0.25)
        #以下采用两种方式进行数值数据标准化
        y_train = boxcox1p(y_train , boxcox_normmax(y_train+1))
        # numeric_feats = X_train.dtypes[X_train.dtypes != "object"].index
        # skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna()))
        # skewed_feats = skewed_feats[skewed_feats > 0.75]
        # skewed_feats = skewed_feats.index
        # X_train[skewed_feats] = np.log1p(X_train[skewed_feats])
        # X_test[skewed_feats] = np.log1p(X_test[skewed_feats])
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())

        ss_X = StandardScaler()
        ss_y = StandardScaler()
        X_train = ss_X.fit_transform(X_train)
        X_test = ss_X.transform(X_test)
        # y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    elif mn == 3:
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        model = RandomForestRegressor(**mparams[mn])
    else:
        # model = lgb.LGBMRegressor(objective='regression',num_leaves=20,learning_rate=0.01, n_estimators=720,reg_lambda=0.005,silent = True)
        model = LGBMClassifier(**lgb_params)
    if (istrain == 1) & (cv > 0):
        # pred = cross_val_predict(model,X_train.values,y_train,cv=10)
        skf = StratifiedKFold(n_splits = cv,shuffle = True,random_state = 1976)
        # kf = KFold(n_splits = cv,shuffle = True,random_state = 1976)
        pred = cross_val_score(model,X_train,y_train,scoring = 'f1_macro',cv=skf,n_jobs = -1)
    elif (istrain == 1) & (cv < 0):
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)
    else:
        model.fit(X_train, y_train)
        # pred = model.predict(X_test)
        pred = model.predict_proba(X_test)
    # if mn == 2:
    #     coef = pd.Series(model.coef_, index = X_train.columns)
    #     logging.info(coef.sort_values().head(38)) 
    return pred

def fsbymodel1(X_train,y_train):
    X_train.fillna(0, inplace = True)
    if mn == 0:
        model = GradientBoostingRegressor(**mparams[mn])
    elif mn == 4:
        model = lgb.LGBMRegressor(objective='regression',num_leaves=20,
                              learning_rate=0.01, n_estimators=720,reg_lambda=0.005,silent = True)
    model.fit(X_train.values, y_train)
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(-feature_importance)
    feature_name = X_train.columns
    t1 = []
    c = 0
    for i in sorted_idx:
        if c > 38:
            break
        # logging.info(feature_name[i]+':'+str(feature_importance[i]))
        print(feature_name[i])
        t1.append(feature_name[i])
        c = c+1
    return t1

def score1(X,y,y1,y2):
    X.fillna(0,inplace = True)
    y.fillna(0,inplace = True)
    X = X.values
    y = y.values
    y1 = y1.values
    y2 = y2.values
    if cv > 0:
        pred = predict1(X,y2,X)
        return pred.mean(),pred.std(),0
    # elif cv == 2:
    #     pred = predict1(X,y,X)
    #     y_test = y
        # X_train,X_test,y_train,y_test = ds(X,y,76)
        # pred = predict1(X_train, y_train, X_test)
        # giniv1 = gini(y_test,pred)
        # X_train,X_test,y_train,y_test = ds(X,y,11)
        # pred = predict1(X_train, y_train, X_test)
        # giniv2 = gini(y_test,pred)
        # X_train,X_test,y_train,y_test = ds(X,y,20)
        # pred = predict1(X_train, y_train, X_test)
        # giniv3 = gini(y_test,pred)
        # return (giniv1+giniv2+giniv3)/3,0
    else:
        # skf = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 1976)
        kf = KFold(n_splits = 5,shuffle = True,random_state = 1976)
        giniv1 = np.zeros(5)
        mse1 = np.zeros(5)
        # for i1, (train_index, test_index) in enumerate(skf.split(X, y1)):
        for i1, (train_index, test_index) in enumerate(kf.split(X)):
            pred = predict1(X[train_index], y2[train_index], X[test_index])
            pred[:,1] = pred[:,1] * 2
            pred[:,2] = pred[:,2] * 6
            giniv1[i1] = gini(y[test_index],pred.sum(axis = 1))
            mse1[i1] = mean_squared_error(y[test_index],pred.sum(axis = 1))
    return giniv1.mean(),giniv1.std(),mse1.mean()

def testfeature1(afl,fl,hs,df_feature):
    hscorep = hs
    hscorec = 0
    hsfl = fl.copy()
    cfl = fl.copy()
    cit = len(fl)
    len1 = len(afl)
    lf = 'dan'
    if cit > 0:
        lf = cfl.pop()
        cit = len(cfl)
   
    for i1 in range(len1):
        if i1 > cit:
            break
        if i1 == cit:
            for i2 in afl:
                if (lf != 'dan') & (i2 != lf):
                    continue
                lf = 'dan'
                if i2 not in cfl:
                    str1 = i2.split('_',1)
                    if str1[0] in cfl:
                        continue
                    cfl.append(i2)
                    df_sf = selectfeature1(cfl,df_feature)
                    hscorec,mse1,pse1 = score1(df_sf.iloc[:,1:],df_feature['Y'],df_feature['Y1'],df_feature['Y2'])
                    logging.info("g:%f,m:%f,f:%s"%(hscorec,mse1,','.join(cfl)))
                    if metrics == 0:
                        if hscorec >= hscorep:
                            hscorep = hscorec
                            hsfl = cfl.copy()
                            logging.info("h:g:%f,m:%f,f:%s"%(hscorec,mse1,','.join(hsfl)))
                    elif metrics == 1:
                        if mse1 <= hscorep:
                            hscorep = mse1
                            hsfl = cfl.copy()
                            logging.info("h:g:%f,m:%f,f:%s"%(hscorec,mse1,','.join(hsfl)))
                    elif metrics == 2:
                        if pse1 <= hscorep:
                            hscorep = pse1
                            hsfl = cfl.copy()
                            logging.info("h:g:%f,m:%f,f:%s"%(hscorec,mse1,','.join(hsfl)))
                    cfl.pop()
            logging.info("hiscore:%f,feature:%s"%(hscorep,','.join(hsfl)))
            cfl = hsfl
            cit = len(hsfl)

def dataexp1(fl,df):
    fbi1 = ['tfc26','tfc21','sfc2','tfc18','tfc15','tfc19','tfc31']
    fl['p1'] = fbi1
    t1 = makefeature1(fl,df)
    t2 = t1.copy()
    t4 = selectfeature1(fbi1,t2)
    t3 = t2[t2['Y'] > 0]
    i1 = 0
    i2 = 0
    giniv1 = 0
    while i1 < 8:
        while i2 < i1:
            t2 = pd.concat([t2, t3], axis=0)
            i2 = i2 + 1
        i1 = i1 + 1

        # t4 = selectfeature1(afl,t2)
        # X_train2 = t4.iloc[:,1:]
        # y_train2 = t2['Y']
        # fbi1 = fsbymodel1(X_train2,y_train2)
        
        t4 = selectfeature1(fbi1,t2)
        X_train2 = t4.iloc[:,1:]
        y_train2 = t2['Y']

        X_train2.fillna(0, inplace = True)
        y_train2.fillna(0, inplace = True)
        model = GradientBoostingRegressor(**mparams[0])
        model.fit(X_train2, y_train2)

        t4 = selectfeature1(fbi1,t1)
        X_train1 = t4.iloc[:,1:]
        y_train1 = t1['Y']
        X_train1.fillna(0, inplace = True)
        y_train1.fillna(0, inplace = True)
        kf1 = KFold(n_splits=10, shuffle=True, random_state=0)
        for train, test in kf1.split(X_train1,y_train1):
            giniv1 = giniv1 + gini(model.predict(X_train1.values[test]),y_train1[test])
        giniv1 = giniv1 / 10
        logging.info("adddata:%d,gini:%f,feature:%s"%(i1-1,giniv1,','.join(fbi1)))

def process():
    df = pd.read_csv(path_train,dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                            LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.float32,
                                            HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8,
                                            Y=np.float16))
    # df = pd.read_csv(path_train)
    global mn,fl
    transdata1(df)
    # df = df[df['Y'] < 120] # 排除异常点
    logging.info("train:%d,cv:%d,mn:%d,ensemble:%d,adddata:%d,metrics:%d"%(istrain,cv,mn,isensemble,isadddata,metrics))
    if istrain == 1:
        fl['p1'] = afl
        # dataexp1(fl,df)
        # return
        df_feature = makefeature1(fl,df)
        del df
        # print(df_feature.columns.values)
        #以下为生成多份理赔样本
        if isadddata > 0:
            t1 = df_feature[df_feature['Y'] > 0]
            # t2 = df_feature[df_feature['Y'] > 1]
            i1 = 0
            while i1 < isadddata:
                df_feature = pd.concat([df_feature, t1], axis=0)
                i1 = i1 + 1
        elif isadddata < 0:
            t1 = df_feature[df_feature['Y'] > 0]
            t2 = df_feature[df_feature['Y'] <= 0]
            t2,_ = train_test_split(t2,  random_state=1976, test_size= abs(isadddata)/10)
            df_feature = pd.concat([t1, t2], axis=0)
        # df_sf = selectfeature1(fl['p1'],df_feature) 
        # X_train = df_sf.iloc[:,1:]
        # y_train = df_feature['Y']
        # gs1(X_train,y_train)
        # fsbymodel1(X_train,y_train)
        testfeature1(afl,[],0,df_feature)
        # for i1,v1 in fl.items():
        #     df_sf = selectfeature1(v1,df_feature)
        #     # print(df_sf.columns)
        #     logging.debug(df_sf.columns.tolist())
        #     giniv1,mse1,pse1 = score1(df_sf.iloc[:,1:],df_feature['Y'],df_feature['Y1'],df_feature['Y2'])
        #     logging.info("gini:%f,mse:%f,pse:%f,feature:%s"%(giniv1,mse1,pse1,','.join(v1))) 
    else:
        fl = {}
        f = ['t31200','h11202_mean','h11301_mean','c11022__3','t21002_mean','s11323_mean'] # gbr特征
        logging.info("feature:%s"%(','.join(f))) 
        fl['p1'] = f
        df_feature = makefeature1(fl,df)

        df_sf = selectfeature1(f,df_feature)
        
        X_train = df_sf.iloc[:,1:]
        y_train = df_feature['Y2']
        mff.clear()

        if isensemble == 1:
            fl1 = {}
            f1 = ['tfc9','tfc10','tfc11','tfc4','hfc7','tfc1','tfc21','tfc24','tfc30','hfc8','sfc5','sfc6','tfc27','tfc23','tfc25'] # rf特征
            logging.info("rf feature:%s"%(','.join(f1))) 
            fl1['p1'] = f1
            df_feature = makefeature1(fl1,df)
            #以下为生成多份理赔样本
            if isadddata > 0:
                t1 = df_feature[df_feature['Y'] > 0]
                # t2 = df_feature[df_feature['Y'] > 1]
                i1 = 0
                while i1 < isadddata:
                    df_feature = pd.concat([df_feature, t1], axis=0)
                    i1 = i1 + 1
            df_sf = selectfeature1(f1,df_feature)
            X_train1 = df_sf.iloc[:,1:]
            y_train1 = df_feature['Y']
            mff.clear()

        df = pd.read_csv(path_test, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                               LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.float32,
                                               HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8))
        # df = pd.read_csv(path_test)
        transdata1(df)
        df_feature = makefeature1(fl,df)
        logging.debug(df_feature.columns) 
        df_sf = selectfeature1(f,df_feature)
        X_test = df_sf.iloc[:,1:]
        logging.debug(X_test.columns)
        X_train.fillna(0,inplace=True)
        X_test.fillna(0,inplace=True)
        pred1 = predict1(X_train.values,y_train.values,X_test.values)
        pred1[:,1] = pred1[:,1] * 2
        pred1[:,2] = pred1[:,2] * 6
        mff.clear()
        if isensemble == 1:
            mn = 3
            df_feature = makefeature1(fl1,df)
            df_sf = selectfeature1(f1,df_feature)
            X_test1 = df_sf.iloc[:,1:]
            pred2 = predict1(X_train1,y_train1,X_test1)
            pred1 = (pred1 + pred2)/2
        df_feature["Y"] = pred1.sum(axis = 1)
        output(df_feature)

if __name__ == "__main__":
    # 程序入口
    process()
    # detection()

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res