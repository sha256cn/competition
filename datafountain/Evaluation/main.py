# -*- coding:utf8 -*-
import pandas as pd
import sys
from pandas import Series,DataFrame
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import  feature_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore")

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
pay = 0.3 #赔付率阈值
cv = 0
fcl = {'tfc' : 13, 'sfc' : 3, 'hfc' : 5, 'lfc' : 4, 'cfc' : 3} #特征选择参数
fs = {'p1':['tfc6', 'tfc12', 'tfc13', 'hfc3'],
      'p2':['tfc6', 'tfc12', 'tfc13', 'hfc3','sfc3'],
      'p3':['tfc6', 'tfc12', 'tfc13', 'hfc3','cfc3']}#特征方案
fs = {'p1' : ['tfc1','tfc6','tfc13_1','tfc12__4','lfc3']}
afl = ['tfc1_mean', 'tfc1_std', 'tfc2_mean', 'tfc2_std', 'tfc3_mean', 'tfc3_std', 'tfc4_mean', 
       'tfc4_std', 'tfc5_mean', 'tfc5_std', 'tfc6_mean', 'tfc6_std', 'tfc7_max', 'tfc7_mean', 
       'tfc7_min', 'tfc7_std', 'tfc7_pc1', 'tfc7_pc2', 'tfc8_mean', 'tfc8_std', 'tfc9_mean', 
       'tfc9_std', 'tfc10_mean', 'tfc10_std', 'tfc11_mean', 'tfc11_std', 'tfc12__0', 'tfc12__1', 
       'tfc12__2', 'tfc12__3', 'tfc12__4', 'tfc12__5', 'tfc12__6', 'tfc12__7', 'tfc12__8', 
       'tfc12__9', 'tfc12__10', 'tfc12__11', 'tfc12__12', 'tfc12__13', 'tfc12__14', 'tfc12__15', 
       'tfc12__16', 'tfc12__17', 'tfc12__18', 'tfc12__19', 'tfc12__20', 'tfc12__21', 'tfc12__22', 
       'tfc12__23', 'sfc1_max', 'sfc1_mean', 'sfc1_min', 'sfc1_std', 'sfc1_pc1', 'sfc1_pc2', 
       'sfc2_max', 'sfc2_mean', 'sfc2_min', 'sfc2_std', 'sfc2_pc1', 'sfc2_pc2', 'sfc3_max', 
       'sfc3_mean', 'sfc3_min', 'sfc3_std', 'sfc3_pc1', 'sfc3_pc2', 'hfc1_max', 'hfc1_mean', 
       'hfc1_min', 'hfc1_std', 'hfc1_pc1', 'hfc1_pc2', 'hfc2_max', 'hfc2_mean', 'hfc2_min', 
       'hfc2_std', 'hfc2_pc1', 'hfc2_pc2', 'hfc3_max', 'hfc3_mean', 'hfc3_min', 'hfc3_std', 
       'hfc3_pc1', 'hfc3_pc2', 'hfc4__0', 'hfc4__1', 'hfc4__2', 'hfc4__3', 'hfc4__4', 'hfc4__5', 
       'hfc4__6', 'hfc4__7', 'hfc4__8', 'hfc4__9', 'hfc4__10', 'hfc4__11', 'hfc4__12', 'hfc4__13', 
       'hfc4__14', 'hfc4__15', 'hfc4__16', 'hfc4__17', 'hfc4__18', 'hfc4__19', 'hfc4__20', 'hfc4__21', 
       'hfc4__22', 'hfc4__23', 'hfc5_max', 'hfc5_mean', 'hfc5_min', 'hfc5_std', 'hfc5_pc1', 'hfc5_pc2', 
       'lfc1_max', 'lfc1_mean', 'lfc1_min', 'lfc1_std', 'lfc1_pc1', 'lfc1_pc2', 'lfc2_max', 'lfc2_mean', 
       'lfc2_min', 'lfc2_std', 'lfc2_pc1', 'lfc2_pc2', 'lfc3_max', 'lfc3_mean', 'lfc3_min', 'lfc3_std', 
       'lfc3_pc1', 'lfc3_pc2', 'lfc4_max', 'lfc4_mean', 'lfc4_min', 'lfc4_std', 'lfc4_pc1', 'lfc4_pc2',
       'cfc1_cs1', 'cfc2_cs1', 'cfc3_cs3']
# afl = ['tfc4', 'tfc12__14', 'tfc12__0', 'tfc12__14', 'tfc12__22', 'tfc12__21', 'hfc3']
istrain = 1
mn = 'gbr'
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

    # tn = df.groupby(['TERMINALNO'])[col].agg(
    #     {l1:np.max,l4:np.mean,l2:np.min,l3:np.std,
    #      l5:lambda x: stats.norm.interval(ci,np.mean(x),np.std(x))[1] 
    #         - stats.norm.interval(ci,np.mean(x),np.std(x))[0] }).reset_index()
    df_temp = pd.merge(df_base, tn, how='left', on=['TERMINALNO'])
    return df_temp

def stats2(df,df_base,col,pre,ci):
    l1 = pre + 'mean'
    l2 = pre + 'std'
    tn = df
    tn = df.groupby(['TERMINALNO'])[col].agg({l1:np.mean,l2:np.std}).reset_index()
    df_temp = pd.merge(df_base, tn, how='left', on=['TERMINALNO'])
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

# 时间特征
def tfc1(df,df_base):#周一至周五出行时间stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc2(df,df_base):#周六、日出行时间tats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc3(df,df_base):#周一和周五出行时间stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'].isin([0,4]))]
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc4(df,df_base):#周一至周五出行时长stats值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc5(df,df_base):#周六、日出行时长tats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc6(df,df_base):#周一和周五上班高峰出行时长stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'].isin([0,4]))]
    t1 = t1[(df['hour'].isin([7,8,9,17,18]))]
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc7(df,df_base):#周一至周五出行频率stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return stats1(t1,df_base,'hour',pre,tci)

def tfc8(df,df_base):#周六、日出行频率stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day','hour'])['minute'].count().reset_index()
    t1 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    func = globals().get(statsf)
    return func(t1,df_base,'hour',pre,tci)

def tfc9(df,df_base):#上班高峰期（7,8，9，17,18）出行概率stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([7,8,9,17,18]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc10(df,df_base):#凌晨（0,1，2,3,4）出行概率stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([0,1,2,3,4]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc11(df,df_base):#晚餐后（20,21,22,23）出行概率stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t2 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = t1[(df['hour'].isin([20,21,22,23]))]
    t3 = t1.groupby(['TERMINALNO','month','day'])['hour'].count().reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['ratio1'] = t1['hour_y']/t1['hour_x']
    func = globals().get(statsf)
    return func(t1,df_base,'ratio1',pre,tci)

def tfc12(df,df_base):#周一至周五24小时出行比率
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    # func = globals().get(statsf)
    t1 = cal_ratio(t1,'hour',pre)
    t1 = t1.drop(['hour','ratio'], axis=1)
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def tfc13(df,df_base):#按每次trip_id进行统计
    head = sys._getframe().f_code.co_name + '_'
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
    



#海拔特征
def hfc1(df,df_base):#周一至周五出行海拔stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'HEIGHT',pre,hci)

def hfc2(df,df_base):#周六、日出行海拔stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'HEIGHT',pre,hci)

def hfc3(df,df_base):#出行海拔stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    # func = globals().get(statsf)
    return stats1(t1,df_base,'HEIGHT',pre,hci)

def hfc4(df,df_base):#周一至周五24小时海拔std
    pre = sys._getframe().f_code.co_name + '_'
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
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t1 = t1.groupby(['TERMINALNO','month','day'])['HEIGHT'].agg(lambda x:max(x)-min(x)).reset_index()
    # t1 = t1.groupby(['TERMINALNO'])['HEIGHT'].agg({l1:np.mean,l2:np.std}).reset_index()
    return stats1(t1,df_base,'HEIGHT',pre,lci)

#速度特征
def sfc1(df,df_base):#周一至周五出行速度stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    t1 = t1[(df['SPEED'] > 0)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'SPEED',pre,sci)

def sfc2(df,df_base):#周六、日出行速度stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1[(df['SPEED'] > 0)]
    func = globals().get(statsf)
    return stats1(t1,df_base,'SPEED',pre,sci)

def sfc3(df,df_base):#出行速度stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t1 = t1[(df['SPEED'] > 0)]
    # func = globals().get(statsf)
    return stats1(t1,df_base,'SPEED',pre,sci)

#经纬度特征
def lfc1(df,df_base):#周一至周五出行区域stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] < 5)]
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    func = globals().get(statsf)
    return stats1(t1,df_base,'area1',pre,lci)

def lfc2(df,df_base):#周六、日出行区域stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df[(df['weekday'] >= 5)]
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    func = globals().get(statsf)
    return stats1(t1,df_base,'area1',pre,lci)

def lfc3(df,df_base):#出行区域stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t1 = t1.groupby(['TERMINALNO','month','day']).apply(
        lambda x: (np.max(x['LONGITUDE']) - np.min(x['LONGITUDE']))*100 + 
        (np.max(x['LATITUDE']) - np.min(x['LATITUDE']))*100).reset_index()
    t1.rename(columns={0:'area1'}, inplace = True)
    # func = globals().get(statsf)
    return stats1(t1,df_base,'area1',pre,lci)

def lfc4(df,df_base):#按天计算经纬极差,再按车主计算极差的stats1值
    pre = sys._getframe().f_code.co_name + '_'
    t1 = df
    t2 = t1.groupby(['TERMINALNO','month','day'])['LONGITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    t3 = t1.groupby(['TERMINALNO','month','day'])['LATITUDE'].agg(lambda x:max(x)-min(x)).reset_index()
    t1 = pd.merge(t2, t3, how='left', on=['TERMINALNO','month','day'])
    t1['area1'] = t1['LONGITUDE']*100 + t1['LATITUDE']*100
    return stats1(t1,df_base,'area1',pre,lci)

#通话状态特征
def cfc1(df,df_base):#根据样本数据状态不为4为筛选值
    # t = df[(df['Y'] == 0) &  (df['CALLSTATE'] == 3)].groupby(['TERMINALNO']).count().reset_index()
    # print('cs3 nopay:' + str(t['TERMINALNO'].count()))
    # t = df[(df['Y'] > 0) &  (df['CALLSTATE'] == 3)].groupby(['TERMINALNO']).count().reset_index()
    # print('cs3 pay:' + str(t['TERMINALNO'].count()))
    pre = sys._getframe().f_code.co_name + '_'
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
    pre = sys._getframe().f_code.co_name + '_'
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
    pre = sys._getframe().f_code.co_name + '_'
    l1 = pre + 'cs3'
    t1 = df[(df['CALLSTATE'] == 3)]
    t1[l1] = 1
    t1 = t1.groupby(['TERMINALNO'])[l1].mean().reset_index()
    t1 = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    return t1

def data_transformation(df):
    df['ftime'] = df['TIME'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))
    df['ftime'] = pd.to_datetime(df['ftime'])
    df['hour'] = df['ftime'].dt.hour.astype(np.uint8)
    df['minute'] = df['ftime'].dt.minute.astype(np.uint8)
    df['weekday'] = df['ftime'].dt.weekday.astype(np.uint8)
    df['day'] = df['ftime'].dt.day.astype(np.uint8)
    df['month'] = df['ftime'].dt.month.astype(np.uint8)
    # df['rs'] = np.rint(df['SPEED']/10).astype(np.uint8)
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

def tuning(X_train,y_train):
    param = {
        'n_estimators': range(30, 50, 2),
        'max_depth': range(2, 7, 1)
    }
    if mn == 'gbr':
        X_train.fillna(0, inplace = True)
        params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    else:
        model = xgb.XGBRegressor(learning_rate=0.01,n_estimators=500, max_depth=4, silent=True, objective='reg:gamma')
    # clf = GridSearchCV(estimator = model, param_grid = param, scoring='r2', cv=10)
    # clf.fit(X_train, y_train)
    # print(clf.grid_scores_, clf.best_params_, clf.best_score_)

    percentiles = range(1, 100, 2)
    results = []
    X_train.fillna(0, inplace = True)
    for i in percentiles:
        fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = i)
        X_train_fs = fs.fit_transform(X_train, y_train)
        scores = cross_val_score(model, X_train_fs, y_train, cv=5)
        results = np.append(results, scores.mean())
    print(results)
    opt = np.where(results == results.max())[0]
    print(percentiles[int(opt)])
    

def predicate(X_train,y_train,X_test):
    if mn == 'gbr':
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif mn == 'lgb':
        model = lgb.LGBMRegressor(boosting_type = "dart" , learning_rate=0.0021 )
    else:
        model = xgb.XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=500, silent=True, objective='reg:linear')
        # model = xgb.XGBRegressor(learning_rate=0.1, max_depth=2, silent=True, objective='reg:linear')        
    # print(X_train.columns.tolist)
    model.fit(X_train.values, y_train)
    if mn != 'xgboost': 
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(-feature_importance)
        feature_name = X_train.columns
        # for i in sorted_idx:
        #     if i > 2:
        #         break
        #     print(feature_name[i]+':'+str(feature_importance[i]))
        # print(feature_name[sorted_idx[0]]+"|"+feature_name[sorted_idx[1]]+"|"+feature_name[sorted_idx[2]]+"|"+feature_name[sorted_idx[3]]+"|"+feature_name[sorted_idx[4]])
    # return cross_val_score(model,X_train,y_train,scoring = ginival, cv=10)
    return model.predict(X_test.values)

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

def train(X,y):
    # print(X.columns)
    if cv == 1:
        X.fillna(0, inplace = True)
        params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
        pred = cross_val_predict(model,X,y,cv=10)
        return -1,gini(y,pred)
    X_train,X_test,y_train,y_test = ds(X,y,76)
    pred = predicate(X_train, y_train, X_test)
    r2v1 = r2_score(y_test,pred)
    giniv1 = gini(y_test,pred)

    X_train,X_test,y_train,y_test = ds(X,y,11)
    pred = predicate(X_train, y_train, X_test)
    r2v2 = r2_score(y_test,pred)
    giniv2 = gini(y_test,pred)

    X_train,X_test,y_train,y_test = ds(X,y,20)
    pred = predicate(X_train, y_train, X_test)
    r2v3 = r2_score(y_test,pred)
    giniv3 = gini(y_test,pred)

    return (r2v1+r2v2+r2v3)/3,(giniv1+giniv2+giniv3)/3

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def ginival(estimator,X, y):
    pred = estimator.predict(X)
    actual = y
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2
    return giniSum / len(actual)

def makefeature(fcl,df,df_base,df_owner):
    i = 1
    df_temp = df_owner
    for key,value in fcl.items():
        while i <= value:
            func = key + str(i)
            dfn = 'df_' + func
            func = globals().get(key + str(i))
            dfn = func(df,df_base)
            df_temp = pd.merge(df_temp, dfn, how='left', on=['TERMINALNO']) 
            i = i + 1
        i = 1
    return df_temp

def makefeature1(fs,df,df_base,df_owner):
    df_temp = df_owner
    for index,value in fs.items():
        for i,v in enumerate(value):
            dfn = 'df_' + v
            func = globals().get(v)
            dfn = func(df,df_base)
            df_temp = pd.merge(df_temp, dfn, how='left', on=['TERMINALNO'])
    # print(df_temp.columns)
    return df_temp

def featureselected(fs,df_base,df_feature): 
    # print(fs)
    first = False
    starti = 0
    endi = -1
    t = df_base
    colums = list(df_feature.columns)
    for index,value in fs.items():
        for i,v in enumerate(value):
            for i1,v1 in enumerate(colums):
                if (('_' in v) & (v1 == v)) | (('_' not in v) & (v1.startswith(ｖ+'_'))):
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
    return t

def score(fs,df_base,df_feature):
    first = False
    starti = 0
    endi = -1
    t = df_base
    # print(df_feature.columns.tolist)
    colums = list(df_feature.columns)
    for index,value in fs.items():
        for i,v in enumerate(value):
            for i1,v1 in enumerate(colums):
                if (('_' in v) & (v1.startswith(ｖ))) | (('_' not in v) & (v1.startswith(ｖ+'_'))):
                    if first == False:
                        starti=i1
                        first = True
                        # print('starti'+str(starti))
                        # print('endi'+str(endi))
                else:
                    if first == True and endi == -1:
                        endi = i1
                        # print('endi'+str(endi))
            if endi == -1:
                endi = i1 + 1
                # print('endi'+str(endi))
            t = pd.concat([t,df_feature.iloc[:,starti : endi]], axis=1)
            # print(t.columns)
            first = False
            starti = 0
            endi = -1
        r2v,giniv = train(t.iloc[:,1:],df_feature['Y'])
        print('feature:' + ','.join(value) + ',giniscore:' + str(giniv) + ',r2score:' + str(r2v))
        # tuning(t.iloc[:,1:],df_feature['Y'])
        t = df_base
    return r2v,giniv

def make2fs(afl,fl,hs,df_base,df_feature):
    hscorep = hs
    hscorec = 0
    hsfl = fl.copy()
    cfl = fl.copy()
    cfs = {'f1':['tfc1']}
    cfs['f1'] = cfl
    len1 = len(afl)
    for i1 in range(len1):
        cfl.append(afl[i1])
        for i2 in range(len1):
            if i2 > i1:
                cfl.append(afl[i2])
                cfs['f1'] = cfl
                r2sv ,hscorec = score(cfs,df_base,df_feature)
                if hscorec > hscorep:
                    hscorep = hscorec
                    hsfl = cfl.copy()
                    print('feature:' + ','.join(cfl) + ',giniscore:' + str(hscorec) + ',r2score:' + str(r2sv))
                cfl.pop()
        cfl.pop()
    print('hs feature:' + ','.join(hsfl)+',score:' + str(hscorep))

def makefs(afl,fl,hs,df_base,df_feature):
    hscorep = hs
    hscorec = 0
    hsfl = fl.copy()
    cfl = fl.copy()
    cit = len(fl)
    len1 = len(afl)
    cfs = {'f1':['tfc1']}
    cfs['f1'] = cfl
    for i1 in range(len1):
        if i1 > cit:
            break
        if i1 == cit:
            for i2 in afl:
                if i2 not in cfl:
                    cfl.append(i2)
                    cfs['f1'] = cfl
                    hsr2v , hscorec = score(cfs,df_base,df_feature)
                    if hscorec > hscorep:
                        hscorep = hscorec
                        hsfl = cfl.copy()
                        print('feature:' + ','.join(cfl) + ',giniscore:' + str(hscorec) + ',r2score:' + str(hsr2v))
                    cfl.pop()
            print('hs feature:' + ','.join(hsfl)+',score:' + str(hscorep))
            cfl = hsfl
            cit = len(hsfl)

def process():
    # print("lineno:", sys._getframe().f_lineno)
    df = pd.read_csv(path_train, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                  LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                  HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8,
                                                  Y=np.float16))
    # df = df.sort_values(by = ['TERMINALNO','TIME'],axis = 0).reset_index(drop=True)
    data_transformation(df)
    # df = df.drop(['ctime','tshift','cday','dshift'], axis=1)
    df_owner = df[['TERMINALNO','Y','TRIP_ID']].groupby(['TERMINALNO','Y']).count().reset_index()
    # print(df_owner.shape[0],df_pay.shape[0])
    df_owner.rename(columns={'TRIP_ID':'counts'}, inplace = True)
    df_base = df_owner.iloc[:,:1]
    print("hci"+str(hci)+",tci"+str(tci)+",lci"+str(lci)+",sci"+str(sci),",cv"+str(cv))
    if istrain == 1:
        df_feature = makefeature(fcl,df,df_base,df_owner)
        # print(df_feature.columns.tolist())
        # makefs(afl,['hfc2_pc2','tfc4_std','tfc12__14','tfc12__22','tfc12__21','tfc10_std'],0.09367407600027881,df_base,df_feature)
        score(fs,df_base,df_feature)    
    else:
        f = {'p1':['sfc2_std','tfc12__14','tfc4_std','hfc3_mean','tfc5_std','lfc2_min','cfc1_cs1','tfc9_std','tfc12__0','hfc3_pc1','cfc3_cs3']}
        df_feature = makefeature(fcl,df,df_base,df_owner)
        t = featureselected( f ,df_base,df_feature)
        X_train = t.iloc[:,1:]
        y_train = df_feature.iloc[:,1]
        df = pd.read_csv(path_test, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                    LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                    HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8))
        # df = df.sort_values(by = ['TERMINALNO','TIME'],axis = 0).reset_index(drop=True)
        data_transformation(df)
        df_owner = df[['TERMINALNO', 'TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
        df_owner.rename(columns={'TRIP_ID':'counts'}, inplace = True)
        df_base = df_owner.iloc[:,:1]
        df_feature = makefeature(fcl,df,df_base,df_owner)
        t = featureselected(f,df_base,df_feature)
        X_test = t.iloc[:,1:]
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        df_feature["Y"] = predicate(X_train,y_train,X_test)
        output(df_feature)

if __name__ == "__main__":
    # 程序入口
    process()
    # detection()
