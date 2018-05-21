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
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,make_scorer
import xgboost as xgb
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_val_predict,KFold
from scipy.special import boxcox1p
from scipy.stats import skew,boxcox_normmax
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,format='%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
pay = 0.3 #赔付率阈值
cv = 0

mparams = [{'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls','random_state':1976},
           {'max_depth':4, 'learning_rate':0.005, 'n_estimators':1000, 'silent':True, 'objective':'reg:linear'},
           {'boosting_type': "dart" , 'learning_rate':0.0021},
           {'n_estimators' : 1300, 'max_features' : 0.5, 'min_samples_leaf' : 30,'random_state' : 1}]
mff = {}#是否已生成特征
fl = {'p1':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','lfc5'],#adddata=0,0.13455
      'p2':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','tfc27'],#adddata=0,0.14707
      'p3':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc1','sfc1'],#adddata=0,0.10581
      'p4':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','tfc21'],#adddata=0,0.13099,adddata=2,0.14909
      'p5':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','tfc21','tfc24','tfc30'],#adddata=2,0.16267
      'p6':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','cfc4'],#adddata=0,0.12327
      'p7':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','hfc1'],
      'p8':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','cfc3'],#adddata=0,0.14537
      'p9':['tfc1','tfc4','tfc9','tfc10','tfc11','hfc7','sfc1','sfc4']}
# fl = {'p1':['tfc9','tfc10','tfc11','tfc4','hfc3','tfc1']}
afl = ['tfc1_mean','tfc1_std','tfc2_mean','tfc2_std'
        ,'tfc3_mean','tfc3_std','tfc4_mean','tfc4_std','tfc5_mean','tfc5_std'
        ,'tfc6_mean','tfc6_std','tfc7_mean','tfc7_std','tfc8_mean','tfc8_std'
        ,'tfc9_mean','tfc9_std','tfc10_mean','tfc10_std','tfc11_mean','tfc11_std'
        ,'tfc12__0','tfc12__1','tfc12__2','tfc12__3','tfc12__4','tfc12__5'
        ,'tfc12__6','tfc12__7','tfc12__8','tfc12__9','tfc12__10','tfc12__11'
        ,'tfc12__12','tfc12__13','tfc12__14','tfc12__15','tfc12__16','tfc12__17'
        ,'tfc12__18','tfc12__19','tfc12__20','tfc12__21','tfc12__22','tfc12__23'
        ,'tfc13_11_mean','tfc13_11_std','tfc13_12_mean','tfc13_12_std'
        ,'tfc13_13_mean','tfc13_13_std','tfc13_21_mean','tfc13_21_std'
        ,'tfc13_22_mean','tfc13_22_std','tfc14__0','tfc14__1','tfc14__2'
        ,'tfc14__3','tfc14__4','tfc14__5','tfc14__6','tfc14__7','tfc14__8'
        ,'tfc14__9','tfc14__10','tfc14__11','tfc14__12','tfc14__13','tfc14__14'
        ,'tfc14__15','tfc14__16','tfc14__17','tfc14__18','tfc14__19','tfc14__20'
        ,'tfc14__21','tfc14__22','tfc14__23','tfc15_mean','tfc15_std','tfc16_mean'
        ,'tfc16_std','tfc17_mean','tfc17_std','tfc18_mean','tfc18_std'
        ,'tfc19_mean','tfc19_std','tfc20_hc','tfc21_mean','tfc21_std','tfc22_mean'
        ,'tfc22_std','tfc23_tr','tfc24_tr','tfc25_tr','tfc26_hminm'
        ,'tfc26_hminstd','tfc26_hmaxm','tfc26_hmaxstd','tfc27_mean','tfc27_std'
        ,'tfc28_mean','tfc28_std','tfc29_mean','tfc29_std','tfc30_mean'
        ,'tfc30_std','tfc31_mean','tfc31_std','hfc1_mean','hfc1_std','hfc2_max'
        ,'hfc2_mean','hfc2_min','hfc2_std','hfc2_pc1','hfc2_pc2','hfc3_max'
        ,'hfc3_mean','hfc3_min','hfc3_std','hfc3_pc1','hfc3_pc2','hfc4__0'
        ,'hfc4__1','hfc4__2','hfc4__3','hfc4__4','hfc4__5','hfc4__6','hfc4__7'
        ,'hfc4__8','hfc4__9','hfc4__10','hfc4__11','hfc4__12','hfc4__13'
        ,'hfc4__14','hfc4__15','hfc4__16','hfc4__17','hfc4__18','hfc4__19'
        ,'hfc4__20','hfc4__21','hfc4__22','hfc4__23','hfc5_mean','hfc5_std'
        ,'hfc6_mean','hfc6_std','hfc7_mean','hfc7_std','sfc1_mean','sfc1_std'
        ,'sfc2_max','sfc2_mean','sfc2_min','sfc2_std','sfc2_pc1','sfc2_pc2'
        ,'sfc3_mean','sfc3_std','sfc4_mean','sfc4_std','lfc1_mean','lfc1_std'
        ,'lfc2_mean','lfc2_std','lfc3_mean','lfc3_std','lfc4_mean','lfc4_std'
        ,'cfc1_cs1','cfc2_cs1','cfc3_cs3','cfc4_cscmax','cfc4_cscstd'
        ,'cfc4_csrmax','cfc4_csrstd']
afl = ['tfc1','tfc2','tfc3','tfc4','tfc5','tfc6','tfc7','tfc8','tfc9','tfc10',
       'tfc11','tfc12','tfc13','tfc14','tfc15','tfc16','tfc17','tfc18','tfc19',
       'tfc20','tfc21','tfc22','tfc23','tfc24','tfc25','tfc26','tfc27','tfc28',
       'tfc29','tfc30','tfc31','hfc1','hfc2','hfc3','hfc4','hfc5','hfc6','hfc7',
       'sfc1','sfc2','sfc3','sfc4','lfc1','lfc2','lfc3','lfc4','cfc1','cfc2','cfc3','cfc4']
isensemble = 0
istrain = 0
isadddata = 0
mn = 0
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

def la1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO'])[col].agg({'count':'count'}).reset_index()
    t1 = df.groupby(['TERMINALNO'])['count'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def lb1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO','month'])[col].agg({'mean':'count'}).reset_index()
    else:
        t1 = df.groupby(['TERMINALNO','month'])[col].agg({'mean':np.mean}).reset_index()
    t1 = t1.groupby(['TERMINALNO'])['mean'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def lc1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO','weekday'])[col].agg({'mean':'count'}).reset_index()
    else:
        t1 = df.groupby(['TERMINALNO','weekday'])[col].agg({'mean':np.mean}).reset_index()
    t1 = t1.groupby(['TERMINALNO'])['mean'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def ld1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO','wno'])[col].agg({'mean':'count'}).reset_index()
    else:
        t1 = df.groupby(['TERMINALNO','wno'])[col].agg({'mean':np.mean}).reset_index()
    t1 = t1.groupby(['TERMINALNO'])['mean'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def le1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO','month','day'])[col].agg({'mean':'count'}).reset_index()
    else:
        t1 = df.groupby(['TERMINALNO','month','day'])[col].agg({'mean':np.mean}).reset_index()
    t1 = t1.groupby(['TERMINALNO'])['mean'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def lf1(df,df_base,col,pre,type=0):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    if type == 1:
        t1 = df.groupby(['TERMINALNO','TRIP_ID'])[col].agg({'mean':'count'}).reset_index()
    else:
        t1 = df.groupby(['TERMINALNO','TRIP_ID'])[col].agg({'mean':np.mean}).reset_index()
    t1 = t1.groupby(['TERMINALNO'])['mean'].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return df_temp

def cal_ratio(df,col,pre):
    df_temp1 = df[['TERMINALNO',col,'TRIP_ID']].groupby(['TERMINALNO',col]).count().reset_index()
    df_temp1.rename(columns={'TRIP_ID':'count1'}, inplace = True)
    df_temp2 = df[['TERMINALNO','TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
    df_temp2.rename(columns={'TRIP_ID':'count2'}, inplace = True)
    df_result = pd.merge(df_temp1, df_temp2, how='left', on=['TERMINALNO'])
    df_result['ratio'] = df_result['count1']/df_result['count2']
    df_temp1 = df_result[['TERMINALNO',col,'ratio']]
    df_temp2 = pd.get_dummies(df_temp1[col], prefix= pre)
    df_temp1 = pd.concat([df_temp1, df_temp2], axis=1)
    startcol = df_temp2.columns.values[0]
    df_temp1.loc[:,startcol:] = df_temp1.loc[:,startcol:].mul(df_temp1['ratio'],axis=0)
    df_temp1 = df_temp1.groupby('TERMINALNO').sum().reset_index()
    return df_temp1

# 时间特征:
'''
feature:'tfc9','tfc10','tfc11','tfc4','hfc3','tfc1',gini:0.11162,本次成绩采用stat2,stat1:0.09978
feature:'tfc9','tfc10','tfc11','tfc4','hfc1','tfc1',gini:0.13679,tfc1为强特征,统计量均采用stat2
feature:'tfc9','tfc10','tfc11','tfc4','hfc7','tfc1',gini:0.14528
feature:'tfc9','tfc10','tfc11','tfc4','hfc7','tfc1','tfc21','tfc24','tfc30', gini:0.17171
'''
def tfc1(df,df_base):#周一至周五出行时刻stats1值,la
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc2(df,df_base):#周六、日出行时刻tats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc3(df,df_base):#周一和周五出行时刻stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'].isin([0,4]))]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc4(df,df_base):#周一至周五按天出行时长stats值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc5(df,df_base):#周六、日出行时长tats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc6(df,df_base):#周一和周五出行时长stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'].isin([0,4]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc7(df,df_base):#周一至周五出行频率stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc8(df,df_base):#周六、日出行频率stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc9(df,df_base):#周一至周五按天上班高峰期（7,8，9，17,18）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([7,8,9,17,18]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc10(df,df_base):#凌晨（0,1，2,3,4）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t2 = df.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = df[(df['hour'].isin([0,1,2,3,4]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc11(df,df_base):#晚餐后（20,21,22,23）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t2 = df.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = df[(df['hour'].isin([20,21,22,23]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc12(df,df_base):#周一至周五24小时出行时间占比
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    # func = globals().get(statsf)
    t1 = cal_ratio(t1,'hour',pre)
    t1 = t1.drop(['hour','ratio'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def tfc13(df,df_base):#按每次trip_id进行统计
    fname = sys._getframe().f_code.co_name
    mff[fname] = 1
    head = fname + '_'
    t1 = df.groupby(['TERMINALNO','TRIP_ID']).agg({'LONGITUDE':{'flo':'first','llo':'last'},
                                              'LATITUDE':{'fla':'first','lla':'last'},
                                              'HEIGHT':{'hmean':np.mean,'hstd':np.std,'hmax':np.max,'hmin':np.min},
                                              'TIME':{'ft':'first','lt':'last'},
                                              'DIRECTION':{'dmean':np.mean,'dstd':np.std},
                                              'SPEED':{'smean':np.mean,'sstd':np.std}})
    levels = t1.columns.levels
    labels = t1.columns.labels
    t1.columns = levels[1][labels[1]]
    t1 = t1.reset_index()
    t1['lo'] = ((t1['fla'] - t1['lla']).abs() + (t1['flo'] - t1['llo']).abs())*100
    t1['tspan'] = (t1['lt'] - t1['ft'])/60
    t1['hsub'] = t1['hmax'] - t1['hmin']
    t1[['hmean', 'hstd','dmean','dstd','smean','sstd','lo','hsub']] = t1[['hmean', 'hstd','dmean','dstd','smean','sstd','lo','hsub']].astype(np.float32)
    t1[['tspan','TRIP_ID']] = t1[['tspan','TRIP_ID']].astype(np.uint32)
    t1['lt'] = t1['lt'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))
    t1['lt'] = pd.to_datetime(t1['lt'])
    t1['hour'] = t1['lt'].dt.hour.astype(np.uint8)
    t1['weekday'] = t1['lt'].dt.weekday.astype(np.uint8)
    t1['day'] = t1['lt'].dt.day.astype(np.uint8)
    t1['month'] = t1['lt'].dt.month.astype(np.uint8)
    t1 = t1.drop(['flo','llo','fla','lla','ft','lt','hmax','hmin'], axis=1)

    func = globals().get(statsf)
    pre = head + '11_' #周一至周五海拔极差
    t2 = t1[(t1['weekday'] < 5)]
    df_11 = func(t2,df_base,'hsub',pre,hci)

    pre = head + '12_' #周六、周日海拔极差
    t2 = t1[(t1['weekday'] >= 5)]
    df_12 = func(t2,df_base,'hsub',pre,hci)

    pre = head + '13_'#周一至周五上班高峰期（7,8，9，17,18）海拔极差
    t2 = t1[(t1['weekday'] < 5)]
    t2 = t2[(df['hour'].isin([7,8,9,17,18]))]
    df_13 = func(t2,df_base,'hsub',pre,hci)

    pre = head + '21_' #周一至周五连续行驶时间（5分钟之内的间隔）
    t2 = t1[(t1['weekday'] < 5)]
    df_21 = func(t2,df_base,'tspan',pre,tci)

    pre = head + '22_' #周六、周日连续行驶时间（5分钟之内的间隔）
    t2 = t1[(t1['weekday'] >= 5)]
    df_22 = func(t2,df_base,'tspan',pre,tci)

    t1 = pd.merge(df_base, df_11, how='left', on=['TERMINALNO'])
    t1 = pd.merge(t1, df_12, how='left', on=['TERMINALNO'])
    t1 = pd.merge(t1, df_13, how='left', on=['TERMINALNO'])    
    t1 = pd.merge(t1, df_21, how='left', on=['TERMINALNO'])
    t1 = pd.merge(t1, df_22, how='left', on=['TERMINALNO'])
    # print(t1.columns)
    return t1

def tfc14(df,df_base):#周一至周五24小时出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['TRIP_ID'].agg({'ts':'count'}).reset_index()
    t2 = pd.get_dummies(t1['hour'], prefix= pre)
    t1 = pd.concat([t1, t2], axis=1)
    startcol = t2.columns.values[0]
    t1.loc[:,startcol:] = t1.loc[:,startcol:].mul(t1['ts'],axis=0)
    t1 = t1.groupby('TERMINALNO').sum().reset_index()
    t1 = t1.drop(['hour','ts'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def tfc15(df,df_base):#周一至周五上下班高峰时间出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['hour'].isin([7,8,9,17,18]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].agg({'ts':'count'}).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'ts',pre,tci)

def tfc16(df,df_base):#周一和周五上下班高峰时间出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'].isin([0,4]))]
    t1 = t1[(t1['hour'].isin([7,8,9,17,18]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].agg({'ts':'count'}).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'ts',pre,tci)

def tfc17(df,df_base):#凌晨出行时长
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['hour'].isin([0,1,2,3,4]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].agg({'ts':'count'}).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'ts',pre,tci)

def tfc18(df,df_base):#周一至周五按天出行加权时刻
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','month','day','hour'])['TRIP_ID'].agg({'ts1':'count'}).reset_index()
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].agg({'ts2':'count'}).reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['hr']=t1['hour']*(t1['ts1']/t1['ts2'])
    func = globals().get(statsf)
    return func(t1,df_base,'hr',pre,tci)

def tfc19(df,df_base):#周一至周五出行时刻按天mean值stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc20(df,df_base):#周一至周五出行不同时点数stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'hc'

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','hour'])['TRIP_ID'].count().reset_index() 
    t1 = t1.groupby(['TERMINALNO'])['hour'].agg({l1:'count'}).reset_index() 
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def tfc21(df,df_base):#周一至周五出行时刻按天sum值stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].sum().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc22(df,df_base):#周一至周五出行时刻加权时刻
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','hour'])['TRIP_ID'].agg({'ts1':'count'}).reset_index()
    t3 = t1.groupby(['TERMINALNO'])['hour'].agg({'ts2':'count'}).reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1['hr']=t1['hour']*(t1['ts1']/t1['ts2'])
    func = globals().get(statsf)
    return func(t1,df_base,'hr',pre,tci)

def tfc23(df,df_base):#周一至周五上班高峰期（7,8，9，17,18）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'tr'
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([7,8,9,17,18]))]
    t3 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['hour_y']/t1['hour_x']
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1[['TERMINALNO',l1]]

def tfc24(df,df_base):#凌晨（0,1，2,3,4）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'tr'

    t2 = df.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = df[(df['hour'].isin([0,1,2,3,4]))]
    t3 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['hour_y']/t1['hour_x']
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1[['TERMINALNO',l1]]

def tfc25(df,df_base):#晚餐后（20,21,22,23）出行占比stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'tr'

    t2 = df.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = df[(df['hour'].isin([20,21,22,23]))]
    t3 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['hour_y']/t1['hour_x']
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1[['TERMINALNO',l1]]

def tfc26(df,df_base):#周一至周五，按天统计最早及最晚出行时刻,le
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

def tfc27(df,df_base):#周一至周五出行时刻按天std值stats1值,le
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].std().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc28(df,df_base):#周一至周五凌晨（0,1，2,3,4）出行占比stats1值,le
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([0,1,2,3,4]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc29(df,df_base):#周一至周五晚餐后（20,21,22,23）出行占比stats1值,le
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([20,21,22,23]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc30(df,df_base):#按月计算周一至周五出行时刻stats1值,lb
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month'])['hour'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc31(df,df_base):#按周计算周一至周五出行时刻stats1值,ld
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','wno'])['hour'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc32(df,df_base):#计算周一至周五出行时长与总时长的占比
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'tr'


    t2 = df.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = df[(df['weekday'] < 5)]
    t3 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['hour_y']/t1['hour_x']
    t1 = t1.drop(['hour_x','hour_y'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def tfc33(df,df_base):#按周计算周一和周五出行时长与总时长的占比
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'tr'

    t2 = df.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = df[(df['weekday'].isin([0,4]))]
    t3 = t1.groupby(['TERMINALNO'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['hour_y']/t1['hour_x']
    t1 = t1.drop(['hour_x','hour_y'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

#海拔特征
def hfc1(df,df_base):#周一至周五出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def hfc2(df,df_base):#周六、日出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'HEIGHT',pre,hci)

def hfc3(df,df_base):#出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df
    # func = globals().get(statsf)
    return stats1(t1,df_base,'HEIGHT',pre,hci)

def hfc4(df,df_base):#周一至周五24小时海拔std
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','hour'])['HEIGHT'].agg(np.std).reset_index()
    t3 = pd.get_dummies(t2['hour'], prefix= pre)
    t1 = pd.concat([t2, t3], axis=1)
    startcol = t3.columns.values[0]
    t1.loc[:,startcol:] = t1.loc[:,startcol:].mul(t1['HEIGHT'],axis=0)
    t1 = t1.groupby('TERMINALNO').sum().reset_index()
    t1 = t1.drop(['hour','HEIGHT'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def hfc5(df,df_base):#按天计算海拔极差,再按车主计算极差的均值和std
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df.groupby(['TERMINALNO','month','day'])['HEIGHT'].agg(lambda x:max(x)-min(x)).reset_index()
    # t1 = t1.groupby(['TERMINALNO'])['HEIGHT'].agg({l1:np.mean,l2:np.std}).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,lci)

def hfc6(df,df_base):#周一至周五，上班高峰时间海拔极差
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['hour'].isin([7,8,9,17,18]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].agg(lambda x:max(x)-min(x)).reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

def hfc7(df,df_base):#周一至周五按天出行海拔stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].mean().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'HEIGHT',pre,hci)

#速度特征
def sfc1(df,df_base):#周一至周五出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['SPEED'] > 0)]
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def sfc2(df,df_base):#周六、日出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['SPEED'] > 0)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'SPEED',pre,sci)

def sfc3(df,df_base):#出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['SPEED'] > 0)]
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,sci)

def sfc4(df,df_base):#凌晨出行速度stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['hour'].isin([0,1,2,3,4]))]
    t1 = t1[(t1['SPEED'] > 0)]
    func = globals().get(statsf)
    return func(t1,df_base,'SPEED',pre,tci)

#经纬度特征
def lfc1(df,df_base):#周一至周五出行区域stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    func = globals().get(statsf)
    return func(t1,df_base,'area1',pre,lci)

def lfc2(df,df_base):#周六、日出行区域stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    func = globals().get(statsf)
    return func(t1,df_base,'area1',pre,lci)

def lfc3(df,df_base):#出行区域stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    
    t1 = df[(df['LONGITUDE'] < 136) & (df['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    func = globals().get(statsf)
    return func(t1,df_base,'area1',pre,lci)

def lfc4(df,df_base):#按天计算经纬极差,再按车主计算极差的stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1

    t1 = df[(df['LONGITUDE'] < 136) & (df['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t2 = t1.groupby(['TERMINALNO','month','day'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    t3 = t1.groupby(['TERMINALNO','month','day'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['area1'] = t1['LONGITUDE']*100 + t1['LATITUDE']*100
    func = globals().get(statsf)
    return func(t1,df_base,'area1',pre,lci)

def lfc5(df,df_base):#周一至周五出行区域stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 =  pre + 'area1'

    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t1 = t1.groupby(['TERMINALNO']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:l1}, inplace = True)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def lfc6(df,df_base):#双休日出行区域stats1值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 =  pre + 'area1'

    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(t1['LONGITUDE'] < 136) & (t1['LONGITUDE'] > 72)]
    t1 = t1[(t1['LATITUDE'] < 54) & (t1['LATITUDE'] > 3)]
    t1 = t1.groupby(['TERMINALNO']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:l1}, inplace = True)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

#通话状态特征
def cfc1(df,df_base):#根据样本数据状态不为0,4为筛选值
    # t = df[(df['Y'] == 0) &  (df['CALLSTATE'] == 3)].groupby(['TERMINALNO']).count().reset_index()
    # print('cs3 nopay:' + str(t['TERMINALNO'].count()))
    # t = df[(df['Y'] > 0) &  (df['CALLSTATE'] == 3)].groupby(['TERMINALNO']).count().reset_index()
    # print('cs3 pay:' + str(t['TERMINALNO'].count()))
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'cs1'
    t1 = df[(df['CALLSTATE'] != 0)]
    t2 = t1.groupby(['TERMINALNO'])['CALLSTATE'].count().reset_index()
    t1 = df[(df['CALLSTATE'].isin([1,2,3]))]
    t3 = t1.groupby(['TERMINALNO'])['CALLSTATE'].count().reset_index()
    t1 = pd.merge(df_base, t2, how='left', on=['TERMINALNO'])
    t1 = pd.merge(t1, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['CALLSTATE_y']/t1['CALLSTATE_x']
    t1 = t1.drop(['CALLSTATE_y','CALLSTATE_x'], axis=1)
    return t1

def cfc2(df,df_base):#根据样本数据状态为3为筛选值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'cs1'
    t1 = df[(df['CALLSTATE'] != 0)]
    t2 = t1.groupby(['TERMINALNO'])['CALLSTATE'].count().reset_index()
    t1 = df[(df['CALLSTATE'] == 3)]
    t3 = t1.groupby(['TERMINALNO'])['CALLSTATE'].count().reset_index()
    t1 = pd.merge(df_base, t2, how='left', on=['TERMINALNO'])
    t1 = pd.merge(t1, t3, how='left', on=['TERMINALNO'])
    t1[l1] = t1['CALLSTATE_y']/t1['CALLSTATE_x']
    t1 = t1.drop(['CALLSTATE_y','CALLSTATE_x'], axis=1)
    return t1

def cfc3(df,df_base):#根据样本数据状态为3为筛选值
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'cs3'
    t1 = df[(df['CALLSTATE'] == 3)]
    t1[l1] = 1
    t1 = t1.groupby(['TERMINALNO'])[l1].mean().reset_index()
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def cfc4(df,df_base):#按tripid统计callstate的时长，比率
    fname = sys._getframe().f_code.co_name
    pre = fname + '_'
    mff[fname] = 1
    l1 = pre + 'cscmax'
    l2 = pre + 'cscstd'
    l3 = pre + 'csrmax'
    l4 = pre + 'csrstd'
    l5 = pre + '_1'
    l6 = pre + '_3'

    t1 = df.groupby(['TERMINALNO','TRIP_ID','CALLSTATE'])['TRIP_ID'].agg({'cscount':'count'}).reset_index()
    t2 = pd.get_dummies(t1['CALLSTATE'], prefix= pre)
    t1 = pd.concat([t1, t2], axis=1)
    startcol = t2.columns.values[0]
    t1.loc[:,startcol:] = t1.loc[:,startcol:].mul(t1['cscount'],axis=0)
    t1 = t1.groupby(['TERMINALNO','TRIP_ID']).sum().reset_index()
    t1['csc'] = t1.loc[:,l5:l6].sum(axis=1)
    t1['csr'] = t1['csc']/t1['cscount']
    t1 = t1.groupby(['TERMINALNO']).agg({'csc':{l1:np.max,l2:np.mean},'csr':{l3:np.max,l4:np.mean}})
    levels = t1.columns.levels
    labels = t1.columns.labels
    t1.columns = levels[1][labels[1]]
    t1 = t1.reset_index()
    t1[l2] = t1[l1]/t1[l2]
    t1[l4] = t1[l3]/t1[l4]
    t1 = pd.merge(df_base, t1[['TERMINALNO',l1,l2,l3,l4]], how='left', on=['TERMINALNO'])
    logging.debug(t1.columns) 
    return t1

def transdata1(df):
    df['ftime'] = df['TIME'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))
    df['ftime'] = pd.to_datetime(df['ftime'])
    df['hour'] = df['ftime'].dt.hour.astype(np.uint8)
    df['minute'] = df['ftime'].dt.minute.astype(np.uint8)
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
    df = df.drop(['stime'], axis=1)
    df['TRIP_ID'] = ts
    
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

gini_score = make_scorer(gini, greater_is_better=True)

def makefeature1(fs,df):
    if "Y" in df.columns.tolist():
        df_owner = df[['TERMINALNO','Y','TRIP_ID']].groupby(['TERMINALNO','Y']).count().reset_index()
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
        param1 = {'max_depth':range(3,8,1), 'min_samples_split':range(200,801,200)}
        model = GradientBoostingRegressor(n_estimators = 500,learning_rate = 0.01, random_state = 1976)
        gs = GridSearchCV(model,param1,scoring = gini_score,cv = 5)
        gs.fit(X,y)
        logging.info(gs.best_params_)
        logging.info(gs.best_score_)

def predict1(X_train,y_train,X_test):
    # skewed_features = X_train.skew(axis = 0)
    # skewed_features = skewed_features[np.abs(skewed_features)>0.5]
    # X_train[skewed_features.index] = np.log1p(X_train[skewed_features.index])
    # skewed_features = X_test.skew(axis = 0)
    # skewed_features = skewed_features[np.abs(skewed_features)>0.5]
    # X_test[skewed_features.index] = np.log1p(X_test[skewed_features.index])
    # y_train = np.log1p(y_train)
    # y_train.fillna(0, inplace = True)
    if mn == 0:
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        model = GradientBoostingRegressor(**mparams[mn])
    elif mn == 1:
        model = xgb.XGBRegressor(**mparams[mn])
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
        model = lgb.LGBMRegressor(**mparams[mn])
    if cv == 1:
        pred = cross_val_predict(model,X_train,y_train,cv=10)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    # if mn == 2:
    #     coef = pd.Series(model.coef_, index = X_train.columns)
    #     logging.info(coef.sort_values().head(38)) 
    return pred

def fsbymodel1(X_train,y_train):
    X_train.fillna(0, inplace = True)
    model = GradientBoostingRegressor(**mparams[mn])
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
        # print(feature_name[i])
        t1.append(feature_name[i])
        c = c+1
    return t1

def score1(X,y):
    if cv == 1:
        pred = predict1(X,y,X)
        y_test = y
    elif cv == 2:
        X_train,X_test,y_train,y_test = ds(X,y,76)
        pred = predict1(X_train, y_train, X_test)
        giniv1 = gini(y_test,pred)
        X_train,X_test,y_train,y_test = ds(X,y,11)
        pred = predict1(X_train, y_train, X_test)
        giniv2 = gini(y_test,pred)
        X_train,X_test,y_train,y_test = ds(X,y,20)
        pred = predict1(X_train, y_train, X_test)
        giniv3 = gini(y_test,pred)
        return (giniv1+giniv2+giniv3)/3
    else:
        X_train,X_test,y_train,y_test = ds(X,y,76)
        pred = predict1(X_train, y_train, X_test)
    giniv1 = gini(y_test,pred)
    mse1 = mean_squared_error(y_test,pred)
    return giniv1,mse1

def testfeature1(afl,fl,hs,df_feature):
    hscorep = hs
    hscorec = 0
    hsfl = fl.copy()
    cfl = fl.copy()
    cit = len(fl)
    len1 = len(afl)
    for i1 in range(len1):
        if i1 > cit:
            break
        if i1 == cit:
            for i2 in afl:
                if i2 not in cfl:
                    cfl.append(i2)
                    df_sf = selectfeature1(cfl,df_feature)
                    hscorec,mse1 = score1(df_sf.iloc[:,1:],df_feature['Y'])
                    if hscorec > hscorep:
                        hscorep = hscorec
                        hsfl = cfl.copy()
                        logging.info("gini:%f,mse:%f,feature:%s"%(hscorep,mse1,','.join(hsfl)))
                    cfl.pop()
            logging.info("gini score:%f,feature:%s"%(hscorep,','.join(hsfl)))
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
    df = pd.read_csv(path_train, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                            LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                            HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8,
                                            Y=np.float16))
    transdata1(df)
    global mn,fl
    logging.info("hci:%f,tci:%f,lci:%f,sci:%f,train:%d,cv:%d,mn:%d,ensemble:%d,adddata:%d"%(hci,tci,lci,sci,istrain,cv,mn,isensemble,isadddata))
    if istrain == 1:
        # fl['p1'] = afl
        # dataexp1(fl,df)
        # return
        df_feature = makefeature1(fl,df)
        #以下为生成多份理赔样本
        if isadddata > 0:
            t1 = df_feature[df_feature['Y'] > 0]
            t2 = df_feature[df_feature['Y'] > 1]
            i1 = 0
            while i1 < isadddata:
                df_feature = pd.concat([df_feature, t1, t2], axis=0)
                i1 = i1 + 1
        # df_sf = selectfeature1(afl,df_feature) 
        # X_train = df_sf.iloc[:,1:]
        # y_train = df_feature['Y']
        # # gs1(X_train,y_train)
        # fsbymodel1(X_train,y_train)
        # testfeature1(afl,['tfc26','tfc21','sfc2','tfc18','tfc15','tfc19','tfc31'],0.1608,df_feature)
        for i1,v1 in fl.items():
            df_sf = selectfeature1(v1,df_feature)
            logging.debug(df_sf.columns.tolist())
            giniv1,mse1 = score1(df_sf.iloc[:,1:],df_feature['Y'])
            logging.info("gini:%f,mse:%f,feature:%s"%(giniv1,mse1,','.join(v1))) 
    else:
        fl = {}
        f = ['tfc9','tfc10','tfc11','tfc4','hfc7','tfc1','tfc21','tfc24','tfc30'] # gbr特征
        logging.info("gbr feature:%s"%(','.join(f))) 
        fl['p1'] = f
        df_feature = makefeature1(fl,df)

        #以下为生成多份理赔样本
        if isadddata > 0:
            t1 = df_feature[df_feature['Y'] > 0]
            t2 = df_feature[df_feature['Y'] > 1]
            i1 = 0
            while i1 < isadddata:
                df_feature = pd.concat([df_feature, t1, t2], axis=0)
                i1 = i1 + 1

        df_sf = selectfeature1(f,df_feature)
        X_train = df_sf.iloc[:,1:]
        y_train = df_feature['Y']
        
        mff.clear()
        if isensemble == 1:
            fl1 = {}
            f1 = ['tfc2','tfc5','tfc7','tfc8','tfc12','tfc13','tfc14','hfc2','sfc1','sfc2','lfc1','lfc2','cfc4'] # rf特征
            logging.info("rf feature:%s"%(','.join(f1))) 
            fl1['p1'] = f1
            df_feature = makefeature1(fl1,df)
            df_sf = selectfeature1(f1,df_feature)
            X_train1 = df_sf.iloc[:,1:]
            y_train1 = df_feature['Y']
            mff.clear()

        df = pd.read_csv(path_test, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                               LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                               HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8))
        transdata1(df)
        df_feature = makefeature1(fl,df)
        logging.debug(df_feature.columns) 
        df_sf = selectfeature1(f,df_feature)
        X_test = df_sf.iloc[:,1:]
        logging.debug(X_test.columns)
        pred1 = predict1(X_train,y_train,X_test)
        if isensemble == 1:
            mn = 3
            df_feature = makefeature1(fl1,df)
            df_sf = selectfeature1(f1,df_feature)
            X_test1 = df_sf.iloc[:,1:]
            pred2 = predict1(X_train1,y_train1,X_test1)
            pred1 = (pred1 + pred2)/2
        df_feature["Y"] = pred1
        output(df_feature)

if __name__ == "__main__":
    # 程序入口
    process()
    # detection()
